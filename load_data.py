"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import os
import torch
import pickle
import numpy as np
from analyse_data import flatten
from datasets import SimpleDataset, IBDataset, IBSeqDataset, IBiterDataset

def get_simpleenv_data(path, splitsize=0.9, random=False, add_reward=False):
    data = {}
    data_tuples = pickle.load(open(path, "rb"))
    data_tuples = flatten(data_tuples)
    data["obs"] = np.array([d[0].reshape((-1,)) for d in data_tuples])
    if add_reward:
        data["next_obs"] = np.array([np.hstack([d[3].reshape((-1,)), d[2].reshape((-1,))]) for d in data_tuples])
    else:
        data["next_obs"] = np.array([d[3].reshape((-1,)) for d in data_tuples])
    data["action"] = np.array([d[1].reshape((-1,)) for d in data_tuples])
    dones = np.array([d[4] for d in data_tuples])
    if np.sum(dones) == 0:
        data["index"] = np.arange(0, 10000, 100)
    else:
        data["index"] = np.nonzero(dones)[0]

    if splitsize == 0. or splitsize == 1.:
        data_train = data
        data_val = None
    else:
        data_train, data_val = split(data, splitsize, random)
    return data_train, data_val

def get_lsy_data(path, splitsize=0.9, random=False):
    data = {}
    data_tuples = pickle.load(open(path, "rb"))
    data["obs"] = np.array([d[0] for d in data_tuples])
    data["next_obs"] = np.array([d[4] for d in data_tuples])
    data["action"] = np.array([d[1] for d in data_tuples])
    data["reward"] = np.array([d[2] for d in data_tuples]).reshape((-1, 1))
    data["index"] = np.nonzero((data["obs"][:, 1:4] == 50.).all(axis=1))[0]

    data_train, data_val = split(data, splitsize, random)
    return data_train, data_val

def split(datadict, splitsize, random):
    trajectory_count = datadict["index"].shape[0]
    train_traj_count = int(splitsize * trajectory_count)

    trajectory_length = datadict["index"][1] - datadict["index"][0]
    assert ((datadict["index"][1:] - datadict["index"][:-1]) == trajectory_length).all(), \
        "all trajectory lengths should be equal"

    if random:
        train_traj_indices = np.random.choice(datadict["index"], train_traj_count, replace=False)
        assert len(set(train_traj_indices)) == train_traj_indices.shape[0], "made an error and over sampled"

        train_traj_indices = set(train_traj_indices)
        val_traj_indices = set(datadict["index"]) - train_traj_indices
        train_sample_indices = np.hstack([np.arange(datadict["index"][i], datadict["index"][i]+trajectory_length)
                                          for i in train_traj_indices])
        val_sample_indices = np.hstack([np.arange(datadict["index"][i], datadict["index"][i]+trajectory_length)
                                        for i in val_traj_indices])

    else:
        train_sample_indices = np.arange(train_traj_count * trajectory_length)
        val_sample_indices = np.arange(train_traj_count * trajectory_length, trajectory_count * trajectory_length)  #  + 1 was a mistake?

    train_index = np.arange(0, train_sample_indices.shape[0], trajectory_length)
    val_index = np.arange(0, val_sample_indices.shape[0], trajectory_length)

    datadict_train, datadict_val = {}, {}
    for key in datadict.keys():
        if key == "index":
            datadict_train[key] = train_index
            datadict_val[key] = val_index
        else:
            datadict_train[key] = datadict[key][train_sample_indices, :]
            datadict_val[key] = datadict[key][val_sample_indices, :]

    return datadict_train, datadict_val

def convert_to_float32(datadict):
    for key in datadict.keys():
        datadict[key] = datadict[key].astype(np.float32)
    return datadict

def extract_simpleenv_name(path, add_reward=False):
    prefix = "se-"
    return prefix + path.split("/")[-1]

def extract_lsy_name(path):
    name = "lsy"
    for setting in ["bad", "mediocre", "optimized", "global"]:
        if setting in path:
            if setting == "global":
                name += "-" + "mediocre" # legacy
            else:
                name += "-" + setting

    for eps in ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]:
        if eps in path:
            name += "-" + eps

    return name

def get_simpleenv_dataset(path, seq="no", return_dicts=False, bsize=1024, rand=True, splitsize=0.9, add_reward=False):
    datadict_train, datadict_val = get_simpleenv_data(path, splitsize, add_reward=add_reward)
    name = extract_simpleenv_name(path, add_reward)

    datadict_train = convert_to_float32(datadict_train)
    if datadict_val is not None:
        datadict_val = convert_to_float32(datadict_val)

    if seq == "no":
        dataset_train = SimpleDataset(datadict_train)
        if datadict_val is not None:
            dataset_val = SimpleDataset(datadict_val)
        else:
            dataset_val = None
    else:
        print(f"unknown seq type: {seq}")

    if return_dicts:
        return dataset_train, dataset_val, name, (datadict_train, datadict_val)
    return dataset_train, dataset_val, name

def get_ib_dataset(origin, lsy_path=None, neo_config=None, seq="no", return_dicts=False, bsize=1024, rand=True, splitsize=0.9):
    assert origin in ["lsy", "neo"], "origin must be either lsy or neo"
    assert (origin == "lsy" and lsy_path is not None and neo_config is None) or \
           (origin == "neo" and neo_config is not None and lsy_path is None), \
        "origin=lsy needs lsypath; origin=neo needs neoconfig"

    if origin == "lsy":
        datadict_train, datadict_val = get_lsy_data(lsy_path, splitsize=splitsize)
        print(datadict_train["obs"].shape, datadict_val["obs"].shape)
        name = extract_lsy_name(lsy_path)
    else:
        return None

    datadict_train = convert_to_float32(datadict_train)
    datadict_val = convert_to_float32(datadict_val)

    dataset_val = None
    if seq == "iter":
        dataset_train = IBiterDataset(datadict_train, 30, 50, bsize, rand)
        if splitsize != 1.:
            dataset_val = IBiterDataset(datadict_val, 30, 50, bsize, False)
    elif seq == "seq": # this used to be simply if seq
        dataset_train = IBSeqDataset(datadict_train, 30, 50)
        if splitsize != 1.:
            dataset_val = IBSeqDataset(datadict_val, 30, 50)
    else:
        dataset_train = IBDataset(datadict_train)
        if splitsize != 1.:
            dataset_val = IBDataset(datadict_val)

    if return_dicts:
        return dataset_train, dataset_val, name, (datadict_train, datadict_val)
    return dataset_train, dataset_val, name

def get_dataloaders(datasets, batchsize=512, shuffle=True, num_workers=0):
    train_dl = torch.utils.data.DataLoader(datasets[0], batch_size=batchsize, shuffle=shuffle, num_workers=num_workers)
    if datasets[1] is not None:
        val_dl = torch.utils.data.DataLoader(datasets[1], batch_size=batchsize, shuffle=False, num_workers=num_workers)
    else: val_dl = None
    return train_dl, val_dl