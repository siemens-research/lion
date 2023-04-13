"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import torch
import numpy as np
import logging
import torch.nn.functional as F
from load_data import extract_lsy_name, get_simpleenv_dataset, get_dataloaders, get_ib_dataset
from transition_model import load_models
from transition_model import DEFAULT_CONFIG as default_model_config
from recurrent_model_simple import load_rec_transition_models
from recurrent_model_simple import DEFAULT_CONFIG as default_rec_model_config
from plot_policy import plot_policy

import copy
import pickle

from collections import defaultdict
from test_policy import eval_multi_lamda
from policy import Policy, Actor
from critic import Critic

from generate_data import IB_opt, IB_mediocre, IB_bad


DEFAULT_CONFIG = {
    "hidden_dim": 1024,
    "hidden_dim_behavioral": 20,
    "weight_decay": 0.0,
    "lr": 3e-4,
    "updates_per_step": 200,
    "max_lamda": 1.,
    "use_lamda": 1,
    "shuffle": True,
    "rollout_step_length": 100,
    "epochs": 25,
    "fixed_lamda": None,
    "lamda_step": 0.05,
    "sampled_states": "starting_only",

    "eta": 0.1,

    "envname": None,
    "discount": 0.97,
    "warmup_behavior": 20,

    # parameters for model-free experiments
    "use_value": False,
    "tau": 0.005,
    "policy_noise": 0.2,
	"noise_clip": 0.5, # 0.5
	"policy_freq": 2,
	"max_action": 1.
}

model_free_config = copy.deepcopy(DEFAULT_CONFIG)
model_free_config["use_value"] = True
model_free_config["epochs"] = 100
model_free_config["updates_per_step"] = 100
model_free_config["hidden_dim"] = 256

simpleenv_config = copy.deepcopy(DEFAULT_CONFIG)
simpleenv_config["epochs"] = 230
simpleenv_config["warmup_behavior"] = 200
simpleenv_config["rollout_step_length"] = 25
simpleenv_config["hidden_dim"] = 256
simpleenv_config["hidden_dim_behavioral"] = 40
simpleenv_config["discount"] = 0.99

def behavior_chooser(envname):
    if "bad" in envname:
        return IB_bad
    elif "mediocre" in envname:
        return IB_mediocre
    elif "optimized" in envname:
        return IB_opt
    else:
        return None

class LION(object):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self._load_data(self.config["datapath"])
        self._load_models()

        mean_step_return = self.train_ds.get_mean_return(self.config["rollout_step_length"], self.config["discount"])
        self.mean_step_return_mag = np.abs(mean_step_return)

        self.policy = Policy(self.stats, config["hidden_dim"], config["use_lamda"], config["max_lamda"])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), weight_decay=self.config["weight_decay"],
                                                 lr=self.config["lr"])

        self.behavioral = Policy(self.stats, config["hidden_dim_behavioral"], config["use_lamda"], config["max_lamda"])
        self.behavioral_optimizer = torch.optim.Adam(self.behavioral.parameters(), weight_decay=self.config["weight_decay"],
                                                 lr=self.config["lr"])

        if config["use_value"]:
            self._init_value()

        if self.train_ds.data["obs"].shape[0] / config["batchsize"] < self.config["updates_per_step"]:
            self.config["updates_per_step"] = int(self.train_ds.data["obs"].shape[0] / config["batchsize"])

    def _load_data(self, datapath):
        if "IB" in self.config["envname"]:
            train_ds, _, datasetname = get_ib_dataset("lsy", datapath, seq="iter", bsize=config["batchsize"], rand=False, splitsize=1.) # config["model"]["batchsize"]
            train_loader, _ = get_dataloaders((train_ds, None), num_workers=4, shuffle=False, batchsize=1)
        else:
            train_ds, _, datasetname = get_simpleenv_dataset(datapath, splitsize=0., add_reward=True if config["envname"] == "simpleenv" else False)
            train_loader, _ = get_dataloaders((train_ds, None), num_workers=4, shuffle=self.config["shuffle"], batchsize=self.config["model"]["batchsize"])
        
        train_stats = train_ds.get_stats()

        self.data_loader = train_loader
        self.stats = train_stats
        self.train_ds = train_ds
        self.dsname = datasetname

    def _load_models(self):
        self.model_type = self.config["model"].get("model_type", "mlp")
        if self.model_type == "mlp":
            self.models = load_models(self.config["model"], self.dsname, self.stats)
        elif self.model_type == "rnn":
            self.models = load_rec_transition_models(self.config["model"], self.dsname, self.stats)
        else:
            raise NotImplementedError()

    def _init_value(self):
        self.actor = Actor(self.stats["state_dim"], self.stats["action_dim"], self.config["max_action"], self.config["use_lamda"], hidden_dim=self.config["hidden_dim"])
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])

        self.critic = Critic(self.stats["state_dim"], self.stats["action_dim"], self.config["hidden_dim"], self.config["use_lamda"])
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.total_it = 0

    def save(self, savepath):
        if self.config["use_value"]:
            torch.save(self.actor.state_dict(), savepath)
        else:
            torch.save(self.policy.state_dict(), savepath)

    def train_value(self, fix_lamda):
        """
        The code for this function originates from the TD3+BC algorithm (https://github.com/sfujim/TD3_BC)
        and has been adapted as outlined in the LION paper (https://openreview.net/forum?id=a4COps0uokg)
        in order to obtain a model-free baseline that conditiones on the trade-off hyperparameter. In the
        paper, we refer to the derivative method as lambda-TD3+BC.
        Copyright (c) 2021 Scott Fujimoto
        Copyright (c) 2022 Phillip Swazinna (Siemens AG)
        SPDX-License-Identifier: MIT
        """
        metrics = defaultdict(list)

        for i in range(self.config["updates_per_step"]):
            self.total_it += 1

            # collect data
            indices = np.random.choice(100000,size=self.config["batchsize"])
            states = self.train_ds.data["obs"][indices, :]
            behavior_actions = self.train_ds.data["action"][indices, :]

            states = torch.FloatTensor(states)
            behavior_actions = torch.FloatTensor(behavior_actions)

            next_states = self.train_ds.data["next_obs"][indices, :]
            reward = self.train_ds.data["reward"][indices, :]

            next_states = torch.FloatTensor(next_states)
            reward = torch.FloatTensor(reward)

            # sample lamda
            lamda = torch.zeros(states.shape[0], 1)
            if fix_lamda is not None:
                lamda += fix_lamda  # possibility to train for a fixed parameter
            else:
                lamda += np.random.beta(0.1, 0.1, size=(states.shape[0], 1)).astype(np.float32)

            # value training
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(behavior_actions) * self.config["policy_noise"]
                ).clamp(-self.config["noise_clip"], self.config["noise_clip"])
                
                next_action = (
                    self.actor_target(next_states, lamda) + noise # , lamda
                ).clamp(-self.config["max_action"], self.config["max_action"])

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_states, next_action, lamda=lamda)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + self.config["discount"] * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, behavior_actions, lamda=lamda)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.config["policy_freq"] == 0:
                # Compute actor loss
                pi = self.actor(states, lamda)

                Q = self.critic.Q1(states, pi, lamda=lamda)
                normalizer = (1/2.5) * Q.abs().mean().detach()

                value = (-lamda * Q / normalizer).mean()
                penalty = ((1-lamda)*(pi - behavior_actions)**2).mean()
                actor_loss = value + penalty

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.config["tau"] * param.data + (1 - self.config["tau"]) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.config["tau"] * param.data + (1 - self.config["tau"]) * target_param.data)

                # logging
                metrics["actor_loss"].append(actor_loss.item())
                metrics["critic_loss"].append(critic_loss.item())

        # aggregate all metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key])

        return metrics["actor_loss"], metrics["critic_loss"], None


    def train(self, fix_lamda = None, behavior_only = False):
        metrics = defaultdict(list)

        for i in range(self.config["updates_per_step"]):

            # pull behavioral training data separately
            behavioral_states = self.train_ds.data["obs"][self.config["batchsize"]*i:self.config["batchsize"]*(i+1), :]
            behavioral_actions = self.train_ds.data["action"][self.config["batchsize"]*i:self.config["batchsize"]*(i+1), :]
            if "IB" in self.config["envname"]:
                behavioral_states = np.stack(reversed([behavioral_states[:, 6*j:6*(j+1)] for j in range(int(180/6))]), 1)  # other format
            behavioral_states = torch.FloatTensor(behavioral_states)
            behavioral_actions = torch.FloatTensor(behavioral_actions)

            if behavior_only:
                bc_actions = self.behavioral.act(behavioral_states, lamda=torch.zeros((behavioral_states.shape[0], 1)))
                bc_loss = F.mse_loss(bc_actions, behavioral_actions)
                self.behavioral_optimizer.zero_grad()
                bc_loss.backward()
                self.behavioral_optimizer.step()
                metrics["bc_loss"].append(bc_loss.item())
            
            else:
                if self.config["sampled_states"] == "all":
                    states = self.train_ds.data["obs"][self.config["batchsize"]*i:self.config["batchsize"]*(i+1), :]
                    if "IB" in self.config["envname"]:
                        states = np.stack(reversed([states[:, 6*j:6*(j+1)] for j in range(int(180/6))]), 1)  # other format

                elif self.config["sampled_states"] == "starting_only":
                    if "IB" in self.config["envname"]:
                        init_indices = np.hstack([np.arange(30) + x for x in np.arange(0, 100000, 1000)])
                    else:
                        init_indices = np.hstack([np.arange(3) + x for x in np.arange(0, 1000, 25)])
                    states = self.train_ds.data["obs"][np.random.choice(init_indices, size=200)]

                    if "IB" in self.config["envname"]:
                        states = np.stack(reversed([states[:, 6*j:6*(j+1)] for j in range(int(180/6))]), 1)  # other format
                    self.config["batchsize"] = states.shape[0]
                
                states = torch.FloatTensor(states)

                # sample lamda
                lamda = torch.zeros(states.shape[0], 1)
                if fix_lamda is not None:
                    lamda += fix_lamda  # possibility to train for a fixed parameter
                else:
                    lamda += np.random.beta(0.1, 0.1, size=(states.shape[0], 1)).astype(np.float32)

                # perform rollouts -> get return estimate
                loss_normalized, loss_noscale, penalty_loss = self._rollout(states, lamda, steps=self.config["rollout_step_length"], discount=self.config["discount"])

                # calculate data penalty additionally
                policy_actions = self.policy.act(behavioral_states, lamda=torch.zeros((behavioral_states.shape[0], 1)))              
                policy_bc_data_loss = ((behavioral_actions - policy_actions)**2).mean(1).view(-1, 1)
                
                # combine to get loss
                eta = self.config["eta"]
                combined_loss = (lamda * loss_normalized + (1-lamda) * penalty_loss).mean() + eta * policy_bc_data_loss.mean()
                combined_loss_noscale = (lamda * loss_noscale + penalty_loss.detach()).mean()
                
                # optimize
                self.policy_optimizer.zero_grad()
                combined_loss.backward()
                grad_mag = self.policy.l1.weight.grad.abs().mean()
                self.policy_optimizer.step()

                # logging
                metrics["return_loss_normalized"].append(loss_normalized.mean().item())
                metrics["return_loss_noscale"].append(loss_noscale.mean().item())
                metrics["combined_loss_normalized"].append(combined_loss.item())
                metrics["combined_loss_noscale"].append(combined_loss_noscale.item())
                metrics["weight_magnitude_l1"].append(self.policy.l1.weight.abs().mean().item())
                metrics["behavior_loss"].append(penalty_loss.mean().item())
                metrics["gradient_magnitude_l1"].append(grad_mag.item())

        # aggregate all metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key])

        return metrics["return_loss_noscale"], metrics["behavior_loss"], metrics["bc_loss"]

    def _rollout(self, states, lamda=None, steps=5, discount=0.99):
        rewards = []
        penalties = []

        if self.model_type != "mlp":
            hiddens, last_predictions = self._init_recurrent(states)
            last_predictions = states[:, -1, :]

        for i in range(steps):
            if lamda is not None:
                actions = self.policy.act(states, lamda)
            else:
                actions = self.policy.act(states)

            bc_actions = self.behavioral.act(states, lamda=torch.zeros((self.config["batchsize"], 1)))
            squared_distances = ((bc_actions - actions)**2).mean(1).view(-1, 1) * (discount ** i)
            penalties.append(squared_distances)

            if self.model_type != "mlp":
                predictions = [model.predict(states, last_predictions, hiddens[j], actions=actions) for j, model in enumerate(self.models)]
            else:
                predictions = [model.predict(states, actions) for model in self.models]

            reward = torch.stack([e["reward_mean"] for e in predictions])
            if len(reward.shape) != 3:
                reward = reward[:, :, None]
            
            min_reward, min_indices = torch.min(reward, 0)
            min_reward = torch.unsqueeze(min_reward, 1)

            states = torch.stack([e["state_mean"] for e in predictions])
            states = states[min_indices[:, 0], torch.arange(min_reward.shape[0])]

            if self.model_type != "mlp":
                last_predictions = torch.stack([e["components"] for e in predictions])[min_indices[:, 0], torch.arange(min_reward.shape[0])]
                hiddens = [e["hidden"] for e in predictions]

            rewards.append(min_reward * (discount ** i))

        # return
        loss = - torch.cat(rewards, 1)
        loss = loss.sum(1).view(-1, 1)
        loss_noscale = loss.detach()
        loss_normalized = loss / self.mean_step_return_mag

        penalty_loss = torch.cat(penalties, 1).sum(1).view(-1, 1) / steps

        return loss_normalized, loss_noscale, penalty_loss

    def _init_recurrent(self, obs):
        hiddens = []
        last_predictions = []
        for m in self.models:
            hidden = m.init_hidden(obs.shape[0])  # init hidden state
            for i in range(1, obs.shape[1]):  # then init history
                res = m.predict(obs[:, i, :4], obs[:, i-1, :], hidden)
                hidden = res["hidden"]

                if i == obs.shape[1] - 1:
                    last_predictions.append(res["components"])

            hiddens.append(hidden)
        
        last_predictions = torch.stack(last_predictions).mean(0)
        return hiddens, last_predictions


def single_training(config):
    algo = LION(config)

    train_returns = []
    train_losses = []

    fixstr = "" if config["fixed_lamda"] is None else f"fixed-{config['fixed_lamda']}_"
    vstr = "" if config["use_value"] is False else "VALUE"
    basename = f"{algo.dsname}_{fixstr}{vstr}"

    for i in range(config["epochs"]):
        if config["use_value"]:
            print(i, algo.total_it)
            ret, loss, _ = algo.train_value(fix_lamda=config["fixed_lamda"])
        else:
            ret, loss, bc = algo.train(fix_lamda=config["fixed_lamda"], behavior_only=i < config["warmup_behavior"])
            print(bc)

        train_returns.append(ret)
        train_losses.append(loss)

        if i % 1 == 0 and i >= config["warmup_behavior"] - 1:
            print(i, ")", ret, loss)

    evaluation_actor = algo.actor if config["use_value"] else algo.policy
    if config["envname"] == "simpleenv":
        savename = basename + "_visualization_all"
        savepath_policy_visualization = "results/" + savename
        if config["fixed_lamda"] is None:
            lamda_steps = np.arange(0., config["max_lamda"] + config["lamda_step"], config["lamda_step"])
            #lamda_steps = np.array([0.0, 0.4, 0.6, 0.65, 0.7, 0.85, 1.0])
            plot_policy(evaluation_actor, savepath_policy_visualization, 3, 7, lamda_steps, envname=config["envname"])
            #plot_policy(evaluation_actor, savepath_policy_visualization, 1, 7, lamda_steps, envname=config["envname"])

    evaluation_actor = algo.actor if config["use_value"] else algo.policy
    behavior_actor = behavior_chooser(config["envname"])

    reps = 50 if config["envname"] == "simpleenv" else 10
    if config["fixed_lamda"] is not None:
        return eval_multi_lamda(evaluation_actor, np.array([config["fixed_lamda"]]), reps=reps,
            envname=config["envname"], discount=config["discount"], behavioral=behavior_actor, learned_b=algo.behavioral, ds=algo.train_ds.data)[config["fixed_lamda"]]
    else:
        return eval_multi_lamda(evaluation_actor, np.arange(0., config["max_lamda"] + config["lamda_step"], config["lamda_step"]), reps=reps,
            envname=config["envname"], discount=config["discount"], behavioral=behavior_actor, learned_b=algo.behavioral, ds=algo.train_ds.data)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    config = {
        "model": default_model_config, #default_model_config for simpleenv, default_rec_model_config for IB
        **simpleenv_config # DEFAULT_CONFIG for LION on IB, model_free_config for lambda-TD3+BC on IB, simpleenv_config for LION on simpleenv
    }
    if config["use_value"]:
        config["batchsize"] = 256
    else:
        config["batchsize"] = 512
    config["model"]["optim_config"] = None  # unpicklable things here

    #config["datapath"] = "datasets/global_1.0_100x1000_30frames.pickle"
    #config["envname"] = "IB_" + extract_lsy_name(config["datapath"])

    # activate these two lines for simpleenv
    config["datapath"] = "datasets/either_0.1"
    config["envname"] = "simpleenv"

    lress = single_training(config.copy())
    with open(f"results/simpleenv_training_repeat_lion.p", "wb") as filehandle:
        pickle.dump(lress, filehandle)