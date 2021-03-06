"""RL Agent utilizes DDPG algorithm

Adapted from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
"""

import copy
import numpy as np
import random


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self,
                 size,
                 seed,
                 mu=0.,
                 theta=0.15,
                 sigma=0.2,
                 uniform_random=True):
        """Initialize parameters and noise process
        Params:
            size: dimension of the output
            mu, theta, sigma: params for Ornstein-Uhlenbeck process
            uniform_random: True, represents Wiener process as random.random else draw from normal distribution
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.uniform_random = uniform_random
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        if self.uniform_random:
            dx = self.theta * (self.mu - x) + self.sigma * np.array(
                [random.random() for i in range(len(x))])
        else:
            dx = self.theta * (self.mu -
                               x) + self.sigma * np.random.standard_normal(
                                   self.size)
        self.state = x + dx
        return self.state
