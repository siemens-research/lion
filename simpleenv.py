"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import gym
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self, size=(10, 10), goal=(3., 6.), max_action=1.,
     batchsize=1, maxlen=25, rew_sigma=1.5):
        super().__init__()
        self.position = None
        self.step_counter = None
        self.sizex, self.sizey = size
        self.goal = np.array(goal).reshape((1, -1))
        self.max_action = max_action
        self.batchsize = batchsize
        self.max_trajectory_len = maxlen
        self.rew_sigma = rew_sigma
        self.max_reward = gaussian_pdf(self.goal, self.goal, self.rew_sigma)[0, 0]
    
    def reset(self):
        xpos = np.random.uniform(0, self.sizex, size=self.batchsize)
        ypos = np.random.uniform(0, self.sizey, size=self.batchsize)
        self.position = np.hstack([xpos, ypos]).reshape((1, -1))
        self.step_counter = 0
        self.done = False
        return self.position.copy()

    def step(self, action):
        assert not self.done, "already done - please reset environment"
        if len(action.shape) == 1:
            action = action.reshape((1, -1))
        action = action.clip(-self.max_action, self.max_action)

        new_pos = self.position + action

        self.position = new_pos
        reward = self._reward()
        self.step_counter += 1
        self.done = self.step_counter == self.max_trajectory_len

        return self.position.copy(), reward, self.done, {}

    def _reward(self):
        reward = gaussian_pdf(self.position, self.goal, self.rew_sigma)
        scaled = reward * (1. / self.max_reward)
        return scaled

def gaussian_pdf(x, mu, sig):
    factor = 1. / (sig * np.sqrt(2 * np.pi))
    exponent = -0.5 * (np.sqrt(np.sum((x - mu)**2, axis=1).reshape((-1, 1))) / sig) ** 2
    res = factor * np.exp(exponent)
    return res