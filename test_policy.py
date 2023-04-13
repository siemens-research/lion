"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import torch
import numpy as np
from policy import Actor, Policy
from generate_data import env_creator


def get_dataset_eval(policy, dataset, lamda, batchsize=512):
    if dataset is None:
        return np.identity(3), None, None

    # first calculate covariance matrix
    all_actions = dataset["action"]
    action_covariances = np.cov(all_actions, rowvar=False)
    iv = np.linalg.inv(action_covariances)
    lamda = lamda.repeat(512, 1)

    dists, dists_maha = [], []
    for i in range(int(dataset["action"].shape[0] / batchsize)):
        states = torch.FloatTensor(dataset["obs"][i*batchsize: (i+1)*batchsize])
        actions = dataset["action"][i*batchsize: (i+1)*batchsize]
        b_actions = policy(states, lamda).detach().numpy()

        diff = actions - b_actions
        euclid = np.sqrt((diff**2).sum(axis=1))
        maha = np.diag( np.dot( np.dot(diff, iv), diff.T) )

        dists.append(euclid.mean())
        dists_maha.append(maha.mean())
    
    return iv, dists, dists_maha


def evaluate(policy, lamda, discount=0.97, behavioral=None, behavioral_learned=None, check_unknown=True, envname="simpleenv", action_matrix=None):
    #lamda = torch.Tensor(lamda.reshape((1,1)))
    env = env_creator(envname)
    done = False
    state = env.reset()
    ret = 0.
    i = 0
    dists, learned_dists, dists_maha, learned_dists_maha = [], [], [], []
    unknown_counter = 0
    state_dims = []

    while not done and i < 100:
        if type(policy) == Policy:
            if len(state.shape) == 1:
                state = state.reshape((1, -1))
            action = policy.act(torch.Tensor(state), lamda).detach().numpy()
        elif  type(policy) == Actor:
            if len(state.shape) == 1:
                state = state.reshape((1, -1))
                
            if lamda is not None:
                action = policy(torch.Tensor(state), lamda).detach().numpy()
            else:
                action = policy(torch.Tensor(state)).detach().numpy()
        else:
            action = policy(state)

        state_dims.append(state[0, 1:4])

        if behavioral is not None:
            s = state.reshape(30, 6)
            b_action = behavioral(s)

            diff = action - b_action
            euclid = np.sqrt((diff**2).sum(axis=1))
            maha = np.diag( np.dot( np.dot(diff, action_matrix), diff.T) )

            dists.append(euclid.mean())
            dists_maha.append(maha.mean())
        
        if behavioral_learned is not None:
            b_action = behavioral_learned.act(torch.Tensor(state), lamda).detach().numpy()

            diff = action - b_action
            euclid = np.sqrt((diff**2).sum(axis=1))
            maha = np.diag( np.dot( np.dot(diff, action_matrix), diff.T) )

            learned_dists.append(euclid.mean())
            learned_dists_maha.append(maha.mean())

        if check_unknown and state[0, 1] >= 4:
            unknown_counter += 1

        if len(action.shape) == 2 and "IB" in envname:
            action = action.reshape((-1,))
        state, reward, done, _ = env.step(action)
        ret += reward * discount ** i
        i += 1
    
    state_dims = np.vstack(state_dims)

    ret_tuple = [ret]
    if behavioral is not None: 
        ret_tuple.append(np.mean(dists))
        ret_tuple.append(np.mean(dists_maha))
    if behavioral_learned is not None: 
        ret_tuple.append(np.mean(learned_dists))
        ret_tuple.append(np.mean(learned_dists_maha))
    if check_unknown is not None: ret_tuple.append(unknown_counter)

    return ret_tuple, state_dims

def eval_many(policy, lamda, discount=1.0, reps=10, envname="simpleenv", behavioral=None, learned_b=None, ds=None):
    lamda = torch.Tensor(lamda.reshape((1,1)))
    action_matrix, ds_dists, ds_dists_maha = get_dataset_eval(policy, ds, lamda)

    results = [evaluate(policy, lamda, discount, envname=envname, behavioral=behavioral, behavioral_learned=learned_b, action_matrix=action_matrix) for _ in range(reps)]
    state_results = [x[1] for x in results]
    results = [x[0] for x in results]
    ret_tuple = []
    for i in range(len(results[0])):
        ret_tuple.append(np.mean([x[i] for x in results]))
        ret_tuple.append(np.std([x[i] for x in results]))

    ret_tuple.append(np.mean(ds_dists))
    ret_tuple.append(np.mean(ds_dists_maha))

    state_results = np.vstack(state_results)
    state_agg = np.mean(state_results, axis=0), np.std(state_results, axis=0), np.min(state_results, axis=0), np.max(state_results, axis=0)
    ret_tuple.append(state_agg)

    return ret_tuple

def eval_multi_lamda(policy, lamdas, discount=1.0, reps=10, envname="simpleenv", behavioral=None, learned_b=None, ds=None):
    res = {}
    for l in lamdas:
        res[l] = eval_many(policy, l, discount, reps, envname=envname, behavioral=behavioral, learned_b=learned_b, ds=ds)
        print(l, res[l][0])
    return res