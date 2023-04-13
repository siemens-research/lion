"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import torch
import numpy as np
from math import floor

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(SimpleDataset, self).__init__()
        self.data = data
        self._calc_stats()

    def __getitem__(self, item):
        return self.data["obs"][item, :], self.data["action"][item, :], self.data["next_obs"][item, :]

    def __len__(self):
        return self.data["obs"].shape[0]

    def sample_start_states(self, size=-1, random=False):
        if size == -1:
            size = self.data["index"].shape[0]
            random = False

        if random:
            indices = np.random.choice(self.data["index"].astype(np.int), size)
        else:
            indices = self.data["index"].astype(np.int)
        start_states = self.data["obs"][indices]

        return start_states

    def _calc_stats(self):
        smean = np.mean(self.data["obs"], axis=0)
        sstd = np.std(self.data["obs"], axis=0)
        sstd[sstd == 0.] = 1.

        nsmean = np.mean(self.data["next_obs"], axis=0)
        nsstd = np.std(self.data["next_obs"], axis=0)
        nsstd[nsstd == 0.] = 1.

        smax = np.max(self.data["next_obs"], axis=0)
        smin = np.min(self.data["next_obs"], axis=0)

        amean = np.mean(self.data["action"], axis=0)
        astd = np.std(self.data["action"], axis=0)
        astd[astd == 0.] = 1.

        amax = np.max(self.data["action"], axis=0)
        amin = np.min(self.data["action"], axis=0)

        try:
            delta_targets = self.data["next_obs"] - self.data["obs"]
        except:
            delta_targets = self.data["next_obs"][:, :2] - self.data["obs"]

        delta_mean = np.mean(delta_targets, axis=0)
        delta_std = np.std(delta_targets, axis=0)
        delta_std[delta_std == 0.] = 1.

        dmax = np.max(delta_targets, axis=0)
        dmin = np.min(delta_targets, axis=0)

        stats = {"state_mean": smean, "state_std": sstd, "state_min": smin, "state_max": smax,
                 "action_mean": amean, "action_std": astd, "action_max": amax, "action_min": amin,
                 "delta_mean": delta_mean, "delta_std": delta_std, "delta_min": dmin, "delta_max": dmax,
                 "next_state_mean": nsmean, "next_state_std": nsstd}

        self.stats = {key: torch.FloatTensor(stats[key]) for key in stats.keys()}
        self.stats["state_dim"] = self.data["obs"].shape[1]
        self.stats["action_dim"] = self.data["action"].shape[1]
        print(self.stats)

    def get_stats(self):
        return self.stats

    def get_mean_return(self, step_length, discount, from_start_only=False):
        rewards = self.data["obs"][:, -1]
        window_discount = discount ** np.arange(step_length)
        window_returns = []

        if from_start_only:
            for index in self.data["index"]:
                index = int(index)
                ret = (rewards[index:index+step_length] * window_discount).sum()
                window_returns.append(ret)
        else:
            for i in range(rewards.shape[0] - step_length):
                ret = (rewards[i:i+step_length] * window_discount).sum()
                window_returns.append(ret)
        return np.mean(window_returns)


class IBDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(IBDataset, self).__init__()
        self.data = data
        self._calc_stats()

    def __getitem__(self, item):
        return self.data["obs"][item, :6], self.data["action"][item, :], self.data["next_obs"][item, :6]

    def __len__(self):
        return self.data["obs"].shape[0]

    def sample_start_states(self, size=-1, random=False):
        if size == -1:
            size = self.data["index"].shape[0]
            random = False

        if random:
            indices = np.random.choice(self.data["index"].astype(np.int), size) + 29
        else:
            indices = self.data["index"].astype(np.int) + 29
        start_states = self.data["obs"][indices]
        start_actions = np.hstack([self.data["action"][indices - i] for i in range(30)])

        return start_states, start_actions

    def _calc_stats(self):
        smean = np.mean(self.data["obs"], axis=0)
        sstd = np.std(self.data["obs"], axis=0)
        sstd[sstd == 0.] = 1.

        smax = np.max(self.data["next_obs"], axis=0)
        smin = np.min(self.data["next_obs"], axis=0)

        amean = np.mean(self.data["action"], axis=0)
        astd = np.std(self.data["action"], axis=0)
        astd[astd == 0.] = 1.

        amax = np.max(self.data["action"], axis=0)
        amin = np.min(self.data["action"], axis=0)

        delta_targets = self.data["next_obs"][:, :6] - self.data["obs"][:, :6]

        delta_mean = np.mean(delta_targets, axis=0)
        delta_std = np.std(delta_targets, axis=0)
        delta_std[delta_std == 0.] = 1.

        dmax = np.max(delta_targets, axis=0)
        dmin = np.min(delta_targets, axis=0)

        stats = {"state_mean": smean, "state_std": sstd, "state_min": smin, "state_max": smax,
                 "action_mean": amean, "action_std": astd, "action_max": amax, "action_min": amin,
                 "delta_mean": delta_mean, "delta_std": delta_std, "delta_min": dmin, "delta_max": dmax}

        self.stats = {key: torch.FloatTensor(stats[key]) for key in stats.keys()}
        self.stats["state_dim"] = self.data["obs"].shape[1]
        self.stats["obs_dim"] = int(self.stats["state_dim"] / 30)
        self.stats["action_dim"] = self.data["action"].shape[1]

    def get_mean_return(self, step_length, discount, from_start_only=False):
        reward_dims = self.data["obs"][:, 4:6]
        rewards = reward_dims[:, 0] * (-3) + reward_dims[: , 1] * (-1)
        window_discount = discount ** np.arange(step_length)
        window_returns = []

        if from_start_only:
            for index in self.data["index"]:
                index = int(index)
                ret = (rewards[index:index+step_length] * window_discount).sum()
                window_returns.append(ret)
        else:
            for i in range(rewards.shape[0] - step_length):
                ret = (rewards[i:i+step_length] * window_discount).sum()
                window_returns.append(ret)
        return np.mean(window_returns)

    def get_stats(self):
        return self.stats


class IBSeqDataset(IBDataset):
    def __init__(self, data, windowsize_past, windowsize_future):
        super(IBSeqDataset, self).__init__(data)
        self.windowsize_past = windowsize_past
        self.windowsize_future = windowsize_future
        self.calc_len()

    def __getitem__(self, item):
        trajectory = floor(item / self.trajectory_length_bordered)
        index = int(item % self.trajectory_length_bordered)
        real_index = int(trajectory * self.trajectory_length + index)
        obs = self.data["obs"][real_index: real_index + self.windowsize_past, :6]
        tar = self.data["next_obs"][real_index: real_index + self.windowsize_past + self.windowsize_future, :6]
        return obs, tar

    def calc_len(self):
        self.trajectory_length = self.data["index"][1] - self.data["index"][0]
        assert ((self.data["index"][1:] - self.data["index"][:-1]) == self.trajectory_length).all(), \
            "all trajectory lengths should be equal"
        self.trajectory_length_bordered = self.trajectory_length - (self.windowsize_past + self.windowsize_future)
        self.sample_count = int(self.trajectory_length_bordered * self.data["index"].shape[0])

    def __len__(self):
        return self.sample_count


class IBiterDataset(torch.utils.data.IterableDataset, IBSeqDataset):
    def __init__(self, data, windowsize_past, windowsize_future, batchsize, random):
        torch.utils.data.IterableDataset.__init__(self)
        IBSeqDataset.__init__(self, data, windowsize_past, windowsize_future)
        self.batchsize = batchsize
        self.random = random

    def __iter__(self):
        self.iter_indices = np.arange(0, self.sample_count)
        if self.random:
            np.random.shuffle(self.iter_indices)
        return self

    def __len__(self):
        print(np.ceil(self.sample_count / self.batchsize))
        return int(np.ceil(self.sample_count / self.batchsize))

    def __next__(self):
        if self.iter_indices.shape[0] > 0:
            current_is = self.iter_indices[:self.batchsize]
            if self.iter_indices.shape[0] > self.batchsize:
                self.iter_indices = self.iter_indices[self.batchsize:]
            else:
                self.iter_indices = np.array([])
            real_is = self.convert_indices(current_is)
            batch = self.batch_access(real_is)
            return batch
        else:
            raise StopIteration

    def convert_indices(self, indices):
        trajectory = np.floor(indices / self.trajectory_length_bordered)
        index = (indices % self.trajectory_length_bordered).astype(np.int32)
        real_index = (trajectory * self.trajectory_length + index).astype(np.int32)
        return real_index

    def batch_access(self, real_indices):
        indexarray_past = np.array([list(range(ri, ri + self.windowsize_past)) for ri in real_indices])
        indexarray_future = np.array([list(range(ri, ri + self.windowsize_past + self.windowsize_future)) for ri in real_indices])
        return self.data["obs"][indexarray_past, :6], self.data["next_obs"][indexarray_future, :6], self.data["action"][indexarray_past, :]