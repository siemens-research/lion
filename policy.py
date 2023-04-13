"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Policy(torch.nn.Module):
    def __init__(self, stats, hidden_dim=20, use_lamda=False, max_lamda=50):
        super(Policy, self).__init__()
        self.single_layer = hidden_dim == 20
        self.l1 = torch.nn.Linear(stats["state_dim"] + use_lamda, hidden_dim)
        if not self.single_layer:
            self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, stats["action_dim"])

        self.register_buffer("state_mean", stats["state_mean"])
        self.register_buffer("state_std", stats["state_std"])

        self.register_buffer("lamda_mean", torch.Tensor([(max_lamda - 0.) / 2.]))
        self.register_buffer("lamda_std", torch.Tensor([np.sqrt((max_lamda - 0.)**2. / 12.)]))

    def forward(self, state, lamda=None):
        s = (state - self.state_mean) / self.state_std
        if lamda is not None:
            l = (lamda - self.lamda_mean) / self.lamda_std
            s = torch.cat([s, l], 1)

        h = F.relu(self.l1(s))
        if not self.single_layer:
            h = F.relu(self.l2(h))
        action = torch.tanh(self.out(h))

        return action

    def act(self, state, lamda=None):
        if len(state.shape) == 3:
            state = torch.cat([state[:, i, :] for i in reversed(range(state.shape[1]))], 1)

        action = self.forward(state, lamda)

        return action

class Actor(nn.Module):
    """
    The code for this class originates from the TD3+BC algorithm (https://github.com/sfujim/TD3_BC)
    and has been adapted as outlined in the LION paper (https://openreview.net/forum?id=a4COps0uokg)
    in order to obtain a model-free baseline that conditiones on the trade-off hyperparameter. In the
    paper, we refer to the derivative method as lambda-TD3+BC.
    Copyright (c) 2021 Scott Fujimoto
    Copyright (c) 2022 Phillip Swazinna (Siemens AG)
    SPDX-License-Identifier: MIT
    """
    def __init__(self, state_dim, action_dim, max_action, use_lamda=False, hidden_dim=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim + use_lamda, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state, lamda=None):
        if lamda is not None:
            state = torch.cat([state, lamda], 1)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))