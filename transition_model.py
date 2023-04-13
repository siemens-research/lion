"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

DEFAULT_CONFIG = {
    # model
    "batchsize": 512,
    "epochs": 300,
    "patience": 10,
    "hidden_layers": [32, 24],
    "output_type": "state",  # delta or state

    # optimizer
    "optim_config": {
        "weight_decay": 0.001,
        "lr": 0.01, 
        "lr_lambda": lambda epoch: 0.99
    },

    # other
    "result_dir": "models",
    "envname": "simpleenv"
}


class MLPTransitionModel(pl.LightningModule):
    def __init__(self, stats, hidden_dims, output_type, optim_config, envtype="simpleenv"):
        super(MLPTransitionModel, self).__init__()
        self.envtype = envtype

        # build model
        input_dim = stats["state_dim"] + stats["action_dim"]
        aug_hidden_dims = [input_dim] + hidden_dims
        self.hiddens = torch.nn.ModuleList()
        for i in range(len(aug_hidden_dims)-1):
            self.hiddens.append(torch.nn.Linear(aug_hidden_dims[i], aug_hidden_dims[i+1]))
        self.out_layer_mean = torch.nn.Linear(aug_hidden_dims[-1], stats["out_dim"])
        self.out_layer_std = torch.nn.Linear(aug_hidden_dims[-1], stats["out_dim"])

        # normalization / clipping constants
        # -> register as buffer so they are saved or can be moved to GPU
        # but the optimizer will not try to train them.
        self.register_buffer('state_mean', stats['state_mean'])
        self.register_buffer('state_std', stats['state_std'])

        self.register_buffer('output_mean', stats['output_mean'])
        self.register_buffer('output_std', stats['output_std'])

        self.register_buffer('min_output', stats['min_output'])
        self.register_buffer('max_output', stats['max_output'])

        self.register_buffer('global_state_min', stats['global_state_min'])
        self.register_buffer('global_state_max', stats['global_state_max'])

        self.output_type = output_type  # TODO: can I save this permanently?
        self.optim_config = optim_config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.optim_config["weight_decay"], lr=self.optim_config["lr"])
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=self.optim_config["lr_lambda"])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, state, action):
        # normalize and concat
        norm_state = (state - self.state_mean) / self.state_std
        h = torch.cat([norm_state, action], 1)

        # perform actual forward pass
        for hidden in self.hiddens:
            h = F.relu(hidden(h))

        norm_output_mean = self.out_layer_mean(h)
        norm_output_std = self.out_layer_std(h.detach())  # std works w/ features provided -> only learn last layer
        norm_output_std = torch.exp(norm_output_std)  # learn in log space for numerical stability
        return norm_output_mean, norm_output_std

    def predict(self, state, action):
        # call forward to calculate normalized rewards
        norm_output_mean, norm_output_std = self.forward(state, action)

        # unnormalize delta / state
        unnorm_output_mean = norm_output_mean * self.output_std + self.output_mean
        unnorm_output_std = norm_output_std * self.output_std  # TODO: correct?

        # clip delta / state
        unnorm_output_mean = torch.min(unnorm_output_mean, self.max_output)
        unnorm_output_mean = torch.max(unnorm_output_mean, self.min_output)
        unnorm_output_std = torch.min(unnorm_output_std, self.max_output - self.min_output)  # std should be below span
        unnorm_output_std = torch.max(unnorm_output_std, torch.zeros_like(unnorm_output_std))  # but above zero

        # move to state if output was delta
        state_std = unnorm_output_std
        if self.output_type == "delta":
            state_mean = state + unnorm_output_mean
        elif self.output_type == "state":
            state_mean = unnorm_output_mean
        else:
            print(f"unknown output type: {self.output_type}")

        # extract reward
        reward_mean = state_mean[:, 2]
        reward_std = state_std[:, 2]
        state_mean = state_mean[:, :2]
        state_std = state_std[:, :2]

        # compile return dict
        return_dict = {"state_mean": state_mean, "state_std": state_std,
                       "reward_mean": reward_mean, "reward_std": reward_std}
        return return_dict

    def training_step(self, batch, batch_idx):
        state, action, next_state = batch  # unpack batch

        # get prediction
        norm_output_mean, norm_output_std = self.forward(state, action)

        # get target (no gradient required)
        with torch.no_grad():
            true_output_norm = self._get_target(state, next_state)

        # calculate mse
        mse = F.mse_loss(norm_output_mean, true_output_norm)

        # stop gradient flow from negative log likelihood over mean -> mean should be trained only by mse
        with torch.no_grad():
            norm_output_mean_nograd = 1 * norm_output_mean
        nll = -torch.distributions.normal.Normal(norm_output_mean_nograd, norm_output_std).log_prob(true_output_norm).mean()

        # final loss and reporting
        loss = mse + nll
        self.log("Train/Loss", loss.item(), on_epoch=True)
        self.log("Train/MSE", mse.item(), on_epoch=True)
        self.log("Train/NLL", nll.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state, action, next_state = batch  # unpack batch

        # no gradient required at all during validation
        with torch.no_grad():
            # get prediction & target
            norm_output_mean, norm_output_std = self.forward(state, action)
            true_output_norm = self._get_target(state, next_state)

            # calculate mse & negative log likelihood
            mse = F.mse_loss(norm_output_mean, true_output_norm)
            nll = -torch.distributions.normal.Normal(norm_output_mean, norm_output_std).log_prob(true_output_norm).mean()
            loss = mse + nll

            self.log("Val/Loss", loss.item(), on_epoch=True)
            self.log("Val/MSE", mse.item(), on_epoch=True)
            self.log("Val/NLL", nll.item(), on_epoch=True)

        return loss

    def _get_target(self, state, next_state):
        # get target
        if self.output_type == "delta":
            true_output = next_state - state
        elif self.output_type == "state":
            true_output = next_state
        else:
            print(f"unknown output type: {self.output_type}")
        true_output_norm = (true_output - self.output_mean) / self.output_std
        return true_output_norm


def get_dummy_stats(nsdim = 3):
    int_keys = ["state_dim", "action_dim", "out_dim"]
    tensor_keys = ["min_output", "max_output", "state_mean", "state_std", "output_mean", "output_std"]
    trainstats = {key: torch.randn(2) for key in tensor_keys}
    for key in int_keys: trainstats[key] = 2
    for key in ["global_state_max", "global_state_min"]: trainstats[key] = torch.randn(1)
    for key in ["next_state_mean", "next_state_std"]: trainstats[key] = torch.randn(nsdim)
    for key in ["state_min", "state_max"]: trainstats[key] = torch.randn(nsdim)
    return trainstats


def load_models(config, datasetname, trainstats):
    modeldir_trans = config["result_dir"] + "/" + datasetname + "/"
    checkpointdir_trans = modeldir_trans + os.listdir(modeldir_trans)[0] + "/checkpoints/"
    transition_models = []

    for chkp in os.listdir(checkpointdir_trans):
        checkpoint_dict = torch.load(checkpointdir_trans + chkp)
        trainstats = augment_simple(trainstats)

        tmodel = MLPTransitionModel(trainstats, config["hidden_layers"], config["output_type"], config["optim_config"], envtype=config["envname"])
        tmodel.load_state_dict(checkpoint_dict["state_dict"])
        transition_models.append(tmodel)

    return transition_models


def single_training(config, datasetname, trainstats, train_loader, val_loader):
    callbacks = []
    if config["patience"] > 0:
        early_stopping_callback = EarlyStopping(monitor="avg_val_loss", patience=config["patience"], mode="min")
        callbacks.append(early_stopping_callback)

    expname = "batchsize-{}_epochs-{}_patience-{}".format(config["batchsize"], config["epochs"], config["patience"])
    root_dir = "{}/{}/{}/".format(config["result_dir"], datasetname, expname)

    checkpoint_callback = ModelCheckpoint(dirpath=root_dir + "checkpoints/", monitor="Val/Loss", mode="min")
    callbacks.append(checkpoint_callback)

    # augment train stats
    trainstats = augment_simple(trainstats)

    transition_model = MLPTransitionModel(trainstats, config["hidden_layers"], config["output_type"], config["optim_config"], envtype=config["envname"])
    transition_trainer = pl.Trainer(max_epochs=config["epochs"], default_root_dir=root_dir, callbacks=callbacks)
    transition_trainer.fit(transition_model, train_loader, val_loader)


def augment_simple(trainstats):
    trainstats["output_mean"] = trainstats["next_state_mean"]
    trainstats["output_std"] = trainstats["next_state_std"]
    trainstats["min_output"] = trainstats["state_min"]
    trainstats["max_output"] = trainstats["state_max"]
    trainstats["out_dim"] = trainstats["state_dim"] + 1
    trainstats["global_state_min"] = torch.Tensor(np.array([-1000.]))
    trainstats["global_state_max"] = torch.Tensor(np.array([1000.]))
    return trainstats


if __name__ == "__main__":
    import os
    import copy
    from load_data import get_simpleenv_dataset, get_dataloaders
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["patience"] = 0

    path = "datasets/either_0.1"

    simpleenv_train_ds, simpleenv_val_ds, datasetname = get_simpleenv_dataset(path, add_reward=True)
    simpleenv_train_loder, simpleenv_val_loader = get_dataloaders((simpleenv_train_ds, simpleenv_val_ds), num_workers=4,
                                                    batchsize=config["batchsize"])
    simpleenv_train_stats = simpleenv_train_ds.get_stats()

    single_training(config, datasetname, simpleenv_train_stats, simpleenv_train_loder, simpleenv_val_loader)