"""Network for Actor and Critic with DDPG algorithm. 

Adapted from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model"""
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 fc1_units: int = 64,
                 fc2_units: int = 32):
        """Initilize parameters and build model.
        Params
        =====
            state_size: dimensions of each state
            action_size: dimensions of each action
            seed: random seed
            fc1_units: number of nodes in the first hidden layer
            fc2_units: number of nodes in the 2nd hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model"""
    def __init__(self,
                 num_agents: int,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 fcs1_units: int = 512,
                 fc2_units: int = 512,
                 ):
        """Initialize parameters and build model.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(num_agents * state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + num_agents * action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (states, actions) -> Q values. """
        xs = F.relu(self.fcs1(states))
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc2(x))
        return self.dropout(self.fc3(x))
