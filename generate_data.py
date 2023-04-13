"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import pickle
import numpy as np
from simpleenv import SimpleEnv
from industrial_benchmark_python.IBGym import IBGym

IB_MEANS = np.array([55.0, 48.75, 50.53, 49.45, 37.51, 166.33])
IB_STDDEVS = np.array([28.72, 12.31, 29.91, 29.22, 31.17, 139.44])

# industrial benchmark baseline policies
def IB_opt(s):
    s = (s - IB_MEANS) / IB_STDDEVS
    return np.array([-s[5, 1] - 0.91, 2 * s[3, 4] - s[0, 0] + 1.43, -3.48 * s[3, 3] - s[4, 3] + 2 * s[0, 0] + 0.81]).clip(-1., 1.)

def IB_mediocre(s):
    s = (s - IB_MEANS) / IB_STDDEVS
    return np.array([25 - s[0, 1], 25 - s[0, 2], 25 - s[0, 3]]).clip(-1., 1.)

def IB_bad(s):
    s = (s - IB_MEANS) / IB_STDDEVS
    return np.array([100 - s[0, 1], 100 - s[0, 2], 100 - s[0, 3]]).clip(-1., 1.)


# baseline in simpleenv
def go_either(state):
    goal1 = np.array([[2.5, 2.5]])
    goal2 = np.array([[7.5, 7.5]])
    distance1 = np.sqrt(np.sum((goal1 - state)**2, axis=1))
    distance2 = np.sqrt(np.sum((goal2 - state)**2, axis=1))
    go1 = (distance1 <= distance2).reshape((-1, 1))
    go2 = (distance2 < distance1).reshape((-1, 1))

    action1 = goal1 - state
    oversize1 = np.max(np.ceil(np.abs(action1)), axis=1).reshape((-1, 1))
    oversize1[oversize1 == 0.] = 1.
    out1 = action1 / oversize1
    action2 = goal2 - state
    oversize2 = np.max(np.ceil(np.abs(action2)), axis=1).reshape((-1, 1))
    oversize2[oversize2 == 0.] = 1.
    out2 = action2 / oversize2

    action = go1 * out1 + go2 * out2
    return action

def augment_controller(controller, epsilon=0.1):
    def new_controller(x):
        if np.random.uniform() < epsilon:
            return np.random.uniform(-1., 1., size=(1, 2))
        return controller(x)
    return new_controller

def generate_data(savepath, env, controller, num_trajectories=100, trajectory_length=100):
    data = []
    for i in range(num_trajectories):
        state = env.reset()
        data.append([])
        for _ in range(trajectory_length):
            action = controller(state)
            next_state, reward, done, info = env.step(action)
            data[i].append((state, action, reward, next_state, done))
            state = next_state
        
    pickle.dump(data, open(savepath, "wb"))

def env_creator(estr):
    if estr == "simpleenv":
        return SimpleEnv()
    elif "IB" in estr:
        return IBGym(70, "classic", "continuous", "include_past")


if __name__ == "__main__":
    mysavepath = "datasets/either_0.1"
    env = env_creator("simpleenv")
    con = augment_controller(go_either)
    generate_data(mysavepath, env, con, trajectory_length=25, num_trajectories=400)