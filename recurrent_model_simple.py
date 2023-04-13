"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import torch
import torch.nn.functional as F
from math import pi, sin
import numpy as np
import pytorch_lightning as pl
import os

EPOCH_LR = 0.01

DEFAULT_CONFIG = {
    # model
    "batchsize": 1024,
    "epochs": 3000,
    "patience": 128,
    "hidden_dim": 30,
    "output_type": "state",  # delta or state
    "model_type": "rnn",
    "self_input": (True, False),

    # optimizer
    "optim_config": {
        "weight_decay": 0.001,
        "lr": EPOCH_LR/80,  # lr=0.001 is standard
        "lr_lambda": lambda epoch: 0.99
    },

    # other
    "result_dir": "models/",
    "envname": "IB"
}

class CombinedModel(object):
    def __init__(self, fatigue_model, consumption_model):
        super(CombinedModel, self).__init__()
        self.fatigue_model = fatigue_model
        self.consumption_model = consumption_model

        self.f_fac = 3.
        self.c_fac = 1.

    def init_hidden(self, batch_size, obs=None):
        return self.fatigue_model.init_hidden(batch_size), self.consumption_model.init_hidden(batch_size)

    def calc_obs(self, old_obs, action):
        with torch.no_grad():
            new_obs = self.fatigue_model.apply_steerings(action, old_obs)
        return new_obs

    def predict(self, obs, past_obs, hidden, actions=None):
        if len(obs.shape) == 3:
            old_obs = obs[:, 1:, :]
            obs = obs[:, -1, :4]
        else:
            old_obs = None

        if actions is not None:
            obs = self.fatigue_model.apply_steerings(actions, obs)

        if past_obs.shape[1] == 6:
            past_prediction_f = past_obs[:, 4].view(-1, 1)
            past_prediction_c = past_obs[:, 5].view(-1, 1)
        else:  # 2
            past_prediction_f = past_obs[:, 0].view(-1, 1)
            past_prediction_c = past_obs[:, 1].view(-1, 1)

        f_prediction = self.fatigue_model.predict(obs, past_prediction_f, hidden[0])
        c_prediction = self.consumption_model.predict(obs, past_prediction_c, hidden[1])

        f_mean = f_prediction["output_mean_clip"]
        f_std = f_prediction["output_std"]
        c_mean = c_prediction["output_mean_clip"]
        c_std = c_prediction["output_std"]
        reward = - self.f_fac * f_mean - self.c_fac * c_mean
        reward_std = self.f_fac * f_std + self.c_fac * c_std

        new_hidden = (f_prediction["hidden"], c_prediction["hidden"])

        resdict = {"hidden": new_hidden, "reward_mean": reward, "reward_std": reward_std, "components": torch.cat([f_mean, c_mean], 1)}
        if old_obs is not None:
            resdict["state_mean"] = torch.cat([old_obs, torch.cat([obs[:, :4], resdict["components"]], 1).view((obs.shape[0], 1, -1))], 1)
        return resdict


class RecurrentTransitionModel(pl.LightningModule):
    def __init__(self, stats, hidden_dim=30, target=None, self_input=True, type="rnn", optimizer_config={}):
        super(RecurrentTransitionModel, self).__init__()
        # build model
        self.hidden_dim = hidden_dim
        self.self_input = self_input
        self.optimizer_config = optimizer_config
        obs_dim = stats["obs_dim"] - 1 - (0 if self_input else 1)

        # type differentiation
        assert type == "rnn" or type == "lstm", "type must be either 'rnn' or 'lstm'"
        self.type = type
        if self.type == "rnn":
            self.embedding = torch.nn.RNNCell(obs_dim, hidden_dim)
        else:
            self.embedding = torch.nn.LSTM(obs_dim, hidden_dim)

        # final output layers
        self.output_mean = torch.nn.Linear(hidden_dim, 1)
        self.output_std = torch.nn.Linear(hidden_dim, 1)

        # target differentiation
        assert target == "fatigue" or target == "consumption", "target must be either 'fatigue' or 'consumption'"
        if target == "fatigue":
            self.t_index = 4
        else:
            self.t_index = 5

        # normalization, clipping, and transition constants
        # -> register as buffer so they are saved or can be moved to GPU
        # but the optimizer will not try to train them.
        self.obs_indices = torch.LongTensor([0, 1, 2, 3])

        self.register_buffer('obs_mean', stats['state_mean'][self.obs_indices])
        self.register_buffer('obs_std', stats['state_std'][self.obs_indices])

        self.register_buffer('prediction_max', stats['state_max'][self.t_index])
        self.register_buffer('prediction_min', stats['state_min'][self.t_index])

        self.register_buffer('prediction_mean', stats['state_mean'][self.t_index])
        self.register_buffer('prediction_std', stats['state_std'][self.t_index])

        self.register_buffer('v_fac', torch.FloatTensor(np.array([1.])))
        self.register_buffer('g_fac', torch.FloatTensor(np.array([10.])))
        self.register_buffer('h_fac', torch.FloatTensor(np.array([20 * sin(15 * pi / 180) / 0.9])))

        self.register_buffer('global_state_min', torch.FloatTensor(np.array([0.])))
        self.register_buffer('global_state_max', torch.FloatTensor(np.array([100.])))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_config["lr"], 
                                    weight_decay=self.optimizer_config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Val/MSE"} 

    def init_hidden(self, batch_size):
        if self.type == "rnn":
            return torch.zeros((batch_size, self.hidden_dim)).to(next(self.parameters()).device)
        else:  # lstm has h & c
            return (torch.zeros((1, batch_size, self.hidden_dim)).to(next(self.parameters()).device),
                    torch.zeros((1, batch_size, self.hidden_dim)).to(next(self.parameters()).device))

    def forward(self, norm_obs, hidden):
        # recurrent cell
        if self.type == "rnn":
            rec_hidden = self.embedding(norm_obs, hidden)
            rec_output = rec_hidden
        else:
            norm_obs = norm_obs.view(1, *norm_obs.shape)
            rec_output, rec_hidden = self.embedding(norm_obs, hidden)
            rec_output = rec_output[0, :, :]

        # calculate mean and stddev of output
        out_mean = self.output_mean(rec_output)
        out_std = self.output_std(rec_output.detach())
        out_std = torch.exp(out_std)

        return out_mean, out_std, rec_hidden

    def apply_steerings(self, action, obs):
        last_v, last_g, last_h = obs[:, 1], obs[:, 2], obs[:, 3]
        next_v = torch.max(self.global_state_min, torch.min(self.global_state_max, last_v + self.v_fac * action[:, 0]))
        next_g = torch.max(self.global_state_min, torch.min(self.global_state_max, last_g + self.g_fac * action[:, 1]))
        next_h = torch.max(self.global_state_min, torch.min(self.global_state_max, last_h + self.h_fac * action[:, 2]))
        next_obs = torch.cat([x.view(-1, 1) for x in [obs[:, 0], next_v, next_g, next_h]], 1)
        return next_obs

    def predict(self, obs, past_prediction, hidden):
        # call forward to calculate normalized rewards
        obs = (obs - self.obs_mean) / self.obs_std
        pp = (past_prediction - self.prediction_mean) / self.prediction_std
        norm_obs = torch.cat([obs, pp], 1) if self.self_input else obs
        norm_output_mean, norm_output_std, new_hidden = self.forward(norm_obs, hidden)

        # map back to unnormalized state space
        output_mean = (norm_output_mean * self.prediction_std) + self.prediction_mean
        output_std = (norm_output_std * self.prediction_std)  # add mean or no?

        # clip unnormalized state
        output_mean_clip = torch.min(output_mean, self.prediction_max)
        output_mean_clip = torch.max(output_mean_clip, self.prediction_min)

        # compile return dict
        return_dict = {"norm_output_mean": norm_output_mean, "norm_output_std": norm_output_std,
                       "output_mean": output_mean, "output_std": output_std,
                       "output_mean_clip": output_mean_clip, "hidden": new_hidden}

        return return_dict

    def training_step(self, batch, batch_idx):
        (new_obs, past_targets, targets), (new_obs_norm, past_targets_norm, targets_norm) = self.unpack_batch(batch)

        hidden = self.init_hidden(new_obs.shape[0])  # initialize hidden with correct batch size

        # history steps: build hidden state
        for i in range(past_targets.shape[1]):
            input_obs = torch.cat([new_obs_norm[:, i, :], past_targets_norm[:, i, :]], 1) if self.self_input else new_obs_norm[:, i, :]
            norm_output_mean, norm_output_std, hidden = self.forward(input_obs, hidden)

        # future steps: get predictions
        norm_output_means, norm_output_stds = [], []
        for i in range(past_targets.shape[1], new_obs.shape[1]):
            input_obs = torch.cat([new_obs_norm[:, i, :], norm_output_mean], 1) if self.self_input else new_obs_norm[:, i, :]
            norm_output_mean, norm_output_std, hidden = self.forward(input_obs, hidden)

            norm_output_means.append(norm_output_mean)
            norm_output_stds.append(norm_output_std)

        # extract only the targets to train against (the ones on the future branch)
        targets = targets[:, past_targets.shape[1]:, :]
        targets_norm = targets_norm[:, past_targets.shape[1]:, :]

        # concat
        norm_output_means = torch.cat(norm_output_means, 1).unsqueeze(2)
        unnorm_output_means = (norm_output_means * self.prediction_std) + self.prediction_mean
        norm_output_stds = torch.cat(norm_output_stds, 1).unsqueeze(2)

        # calculate mse & negative log likelihood
        mse = F.mse_loss(targets_norm, norm_output_means)
        norm_output_means_nograd = norm_output_means.detach()

        nll = -torch.distributions.normal.Normal(norm_output_means_nograd, norm_output_stds).log_prob(targets_norm).mean()

        loss = mse + nll
        self.log("Train/Loss", loss.item(), on_epoch=True)
        self.log("Train/MSE", mse.item(), on_epoch=True)
        self.log("Train/NLL", nll.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (new_obs, past_targets, targets), (new_obs_norm, past_targets_norm, targets_norm) = self.unpack_batch(batch)

        hidden = self.init_hidden(new_obs.shape[0])  # initialize hidden with correct batch size

        with torch.no_grad():
            # history steps: build hidden state
            for i in range(past_targets.shape[1]):
                input_obs = torch.cat([new_obs_norm[:, i, :], past_targets_norm[:, i, :]], 1) if self.self_input else new_obs_norm[:, i, :]
                norm_output_mean, norm_output_std, hidden = self.forward(input_obs, hidden)

            # future steps: get predictions
            norm_output_means, norm_output_stds = [], []
            for i in range(past_targets.shape[1], new_obs.shape[1]):
                input_obs = torch.cat([new_obs_norm[:, i, :], norm_output_mean], 1) if self.self_input else new_obs_norm[:, i, :]
                norm_output_mean, norm_output_std, hidden = self.forward(input_obs, hidden)

                norm_output_means.append(norm_output_mean)
                norm_output_stds.append(norm_output_std)

            # extract only the targets to train against (the ones on the future branch)
            targets = targets[:, past_targets.shape[1]:, :]
            targets_norm = targets_norm[:, past_targets.shape[1]:, :]

            # concat
            norm_output_means = torch.cat(norm_output_means, 1).unsqueeze(2)
            unnorm_output_means = (norm_output_means * self.prediction_std) + self.prediction_mean
            norm_output_stds = torch.cat(norm_output_stds, 1).unsqueeze(2)

            # calculate mse & negative log likelihood
            mse = F.mse_loss(targets_norm, norm_output_means)
            nll = -torch.distributions.normal.Normal(norm_output_means, norm_output_stds).log_prob(targets_norm).mean()
            mae = F.l1_loss(targets, unnorm_output_means)
            mae_check = F.l1_loss(targets_norm, norm_output_means) * self.prediction_std
            loss = mse + nll

            self.log("Val/Loss", loss.item(), on_epoch=True)
            self.log("Val/MSE", mse.item(), on_epoch=True)
            self.log("Val/NLL", nll.item(), on_epoch=True)
            self.log("Val/MAE", mae.item(), on_epoch=True)
            self.log("Val/MAEcheck", mae_check.item(), on_epoch=True)

        return loss

    def unpack_batch(self, batch):
        past_obs, new_obs, _ = batch  # unpack batch # added actions...
        past_obs, new_obs = past_obs[0], new_obs[0]
        with torch.no_grad():
            past_targets = past_obs[:, :, self.t_index].view(*past_obs.shape[:2], 1)
            targets = new_obs[:, :, self.t_index].view(*new_obs.shape[:2], 1)
            new_obs = new_obs[:, :, :4]

            pt_norm = (past_targets - self.prediction_mean) / self.prediction_std
            t_norm = (targets - self.prediction_mean) / self.prediction_std
            no_norm = (new_obs - self.obs_mean) / self.obs_std
        return (new_obs, past_targets, targets), (no_norm, pt_norm, t_norm)


def single_training(config, trainstats, train_loader, val_loader, target):
    callbacks = []
    if config["patience"] > 0:
        early_stopping_callback = EarlyStopping(monitor="Val/MSE", patience=config["patience"], mode="min")
        callbacks.append(early_stopping_callback)

    expname = "batchsize-{}_epochs-{}_patience-{}".format(*[config[x] for x in ["batchsize", "epochs", "patience"]])
    root_dir = "{}/{}/{}/{}/".format(config["result_dir"], config["datasetname"], expname, target)

    checkpoint_callback = ModelCheckpoint(dirpath=root_dir + "checkpoints/", monitor="Val/MSE", mode="min")  # not entire loss, just mse...
    callbacks.append(checkpoint_callback)

    single_self_input = config["self_input"][0 if target=="fatigue" else 1]
    transition_model = RecurrentTransitionModel(trainstats, config["hidden_dim"], target=target, 
                                        self_input=single_self_input, optimizer_config=config["optim_config"])

    transition_trainer = pl.Trainer(max_epochs=config["epochs"], default_root_dir=root_dir, callbacks=callbacks)
    transition_trainer.fit(transition_model, train_loader, val_loader)


def load_rec_transition_models(config, datasetname, trainstats):
    modeldir_trans = config["result_dir"] + datasetname + "/"
    modeldir_trans = modeldir_trans + os.listdir(modeldir_trans)[0] + "/"

    print(modeldir_trans)
    checkpointsdir_fatigue = modeldir_trans + "fatigue/checkpoints/"
    checkpointsdir_consumption = modeldir_trans + "consumption/checkpoints/"
    checkpoints_f = [checkpointsdir_fatigue + ch for ch in sorted(os.listdir(checkpointsdir_fatigue))]
    checkpoints_c = [checkpointsdir_consumption + ch for ch in sorted(os.listdir(checkpointsdir_consumption))]
    transition_models = []
    for chkp_f, chkp_c in zip(checkpoints_f, checkpoints_c):
        checkpoint_dict_f = torch.load(chkp_f)
        checkpoint_dict_c = torch.load(chkp_c)
        f_model = RecurrentTransitionModel(trainstats, config["hidden_dim"], target="fatigue", self_input=config["self_input"][0])
        c_model = RecurrentTransitionModel(trainstats, config["hidden_dim"], target="consumption", self_input=config["self_input"][1])
        try:
            f_model.load_state_dict(checkpoint_dict_f["state_dict"])
            c_model.load_state_dict(checkpoint_dict_c["state_dict"])
            combined_model = CombinedModel(f_model, c_model)
            transition_models.append(combined_model)
        except:
            print("CAUTION: model loading from\n", chkp_f, "\n", chkp_c, "\nfailed!")
    return transition_models


def double_training(config, trainstats, train_loader, val_loader):
    print("## !! ## Fatigue ## !! ##")
    single_training(config, trainstats, train_loader, val_loader, "fatigue")
    print("## !! ## Consumption ## !! ##")
    single_training(config, trainstats, train_loader, val_loader, "consumption")


if __name__ == "__main__":
    import copy
    from load_data import all_ib_dataset_generator, get_ib_dataset, get_dataloaders, all_ib_dataset_generator
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    config = copy.deepcopy(DEFAULT_CONFIG)

    path = "datasets/global_1.0_100x1000_30frames.pickle"
    lsy_train_ds, lsy_val_ds, datasetname = get_ib_dataset("lsy", lsy_path=path, seq="iter", bsize=config["batchsize"])
    lsy_train_loader, lsy_val_loader = get_dataloaders((lsy_train_ds, lsy_val_ds), num_workers=0,
                                                    batchsize=1, shuffle=False)
    config["datasetname"] = datasetname

    lsy_train_stats = lsy_train_ds.get_stats()
    double_training(config, lsy_train_stats, lsy_train_loader, lsy_val_loader)