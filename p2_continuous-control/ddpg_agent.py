"""RL Agent utilizes DDPG algorithm

Adapted from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
"""

import copy
from collections import namedtuple, deque
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

# hyper parameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of parameters
LR_ACTOR = 1e-4  # learning rate for actor
LR_CRITIC = 1e-4  # learning rate for critic
WEIGHT_DECAY = 0  # L2 weight decay

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent():
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 random_seed: int,
                 wiener_random: bool = False):
        """Initialize an Agent object
        Params
        =====
            state_size: dimension of each state
            action_size: dimension of each action
            random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size,
                                 random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size,
                                  random_seed).to(device)
        self.actor_optimzer = optim.Adam(self.actor_local.parameters(),
                                         lr=LR_ACTOR)

        # Critic network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size,
                                   random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size,
                                    random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)

        # Noise Process
        self.noise = OUNoise(action_size, random_seed)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE,
                                   random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay buffer, and use radom samples from buffer to learn
        """
        # gather experiences from all agents
        for state, action, reward, next_state, done in zip(
                states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        # learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given states as per current policy
        """
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using giving batch of experince tuples
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        # unpack
        states, actions, rewards, next_states, dones = experiences

        # ------------ updata critic -----------------#
        # get predicted actions fro next_states and calculate the Q-value
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # compute the target Q-value
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)

        # use local network to get expected Q-value and calculate loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip the gradient, how to decide the max_norm?
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ------------ update actor -----------------#
        # compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimzer.zero_grad()
        actor_loss.backward()
        self.actor_optimzer.step()

        # ------------ update target networks --------#
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau: float):
        """Soft update model parameters
        theta_target = tau * theta_local + (1 - tau)*theta_target

        Params
        =====
            local_model: model that weights will be copied from
            target_model: model that weights will be copied to
            tau: soft update factor
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self,
                 size,
                 seed,
                 mu=0.,
                 theta=0.15,
                 sigma=0.2,
                 wiener_random=False):
        """Initialize parameters and noise process
        Params:
            size: dimension of the output
            mu, theta, sigma: params for Ornstein-Uhlenbeck process
            wiener_random: True, represents Wiener process as random.random else standard_normal
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.wiener_random = wiener_random
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        if self.wiener_random:
            dx = self.theta * (self.mu - x) + self.sigma * np.array(
                [random.random() for i in range(len(x))])
        else:
            dx = self.theta * (self.mu -
                               x) + self.sigma * np.random.standard_normal(
                                   self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object
        Params
        =====
            action_size: dimension of action
            buffer_size: max size of buffer
            batch_size: size of each training batch
            seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            'Experience',
            field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences
                       if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences
                       if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences
                       if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal mermory"""
        return len(self.memory)
