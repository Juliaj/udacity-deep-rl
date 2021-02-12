"""
In memory buffer for data sampling to suppor training
"""
import random
import torch
import numpy as np
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    """
    Fixed sized buffer to store state, action and reward tuples
    """
    def __init__(self, action_size: int, buffer_size: int, batch_size: int,
                 seed: int):
        """
        Initialize a ReplyBuffer object

        Params
        ======
            action_size: dimension of action space
            buffer_size: size of the reply buffer
            batch_size: size of each training batch
            seed: random seed
        """

        super(ReplayBuffer, self).__init__()
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory and return as torch tensors
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # transform to tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences
                                             ])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences
                                              ])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences
                                              ])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences
                       ]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Return the current size of memory
        """
        return len(self.memory)
