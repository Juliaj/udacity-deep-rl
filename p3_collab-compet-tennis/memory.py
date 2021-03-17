"""Replay Buffer to store experience from agents

Adapted from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
"""

# import copy
from collections import namedtuple, deque
import numpy as np
import random
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object
        Params
        =====
            action_size: dimension of action
            buffer_size: max size of buffer
            batch_size: size of each training batch
            seed: random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = torch.manual_seed(seed)
        self.experience = namedtuple('Experience',
                                     field_names=[
                                         'states', 'actions', 'rewards',
                                         'next_states', 'dones'
                                     ])

    def add(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_states: np.ndarray, dones: np.ndarray):
        """Add a new experience to memory.
        Params:
        =====
            states: states from all agents,  dimensions: [num_agents, state_size]
            actions: actions from all agents, dimensions: [num_agents, action_size]
            rewards: rewards from all agents [num_agents, 1]
            next_states: next states from all agents, dimensions: [num_agents, state_size]
            dones: flags on final state, dimensions: [num_agents, 1]
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory to create a batch.
        Return:
            tuples of states, actions, rewards, next_states, dones with the first dimension as batch_size
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.vstack([e.states for e in experiences
                       if e is not None])).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.actions for e in experiences
                       if e is not None])).float().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.rewards for e in experiences
                       if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_states for e in experiences
                       if e is not None])).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.dones for e in experiences
                       if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal mermory"""
        return len(self.memory)
