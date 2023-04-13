"""
Copyright (c) 2022 Phillip Swazinna (Siemens AG)
SPDX-License-Identifier: MIT
"""

import torch
from torch import nn
import torch.nn.functional as F

class Critic(nn.Module):
	"""
    The code for this class originates from the TD3+BC algorithm (https://github.com/sfujim/TD3_BC)
    and has been adapted as outlined in the LION paper (https://openreview.net/forum?id=a4COps0uokg)
    in order to obtain a model-free baseline that conditiones on the trade-off hyperparameter. In the
    paper, we refer to the derivative method as lambda-TD3+BC.
    Copyright (c) 2021 Scott Fujimoto
    Copyright (c) 2022 Phillip Swazinna (Siemens AG)
    SPDX-License-Identifier: MIT
    """
	def __init__(self, state_dim, action_dim, hidden_dim=256, use_lamda=False):
		super(Critic, self).__init__()

		# Q1 architecture
		print("use lamda", use_lamda)
		self.l1 = nn.Linear(state_dim + action_dim + use_lamda, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim + use_lamda, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)


	def forward(self, state, action, lamda=None):
		if lamda is not None:
			sa = torch.cat([state, action, lamda], 1)
		else:
			sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action, lamda=None):
		if lamda is not None:
			sa = torch.cat([state, action, lamda], 1)
		else:
			sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1