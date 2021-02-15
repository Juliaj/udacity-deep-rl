"""
Agent based on DQN and DDQN training. 
    - Adopted baseline code from Udacity Deep RL course.
"""
import numpy as np
import random

from model import QNetwork
from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # mini batch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """
    interacts and learns from environment 
    """
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 algo: str = 'ddqn'):
        """
        Initialize agent
        
        Params
        ======
            state_size: dimension of each state
            action_size: dimension of action space
            seed: random seed
            algo: training algorithm
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size,
                                       seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size,
                                        seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # init time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.algo = algo

    def step(self, state, action, reward, next_state, done):
        """
        Take a learning step if there are enough samples

        Params
        ======
            state(array_like): current_state
            action(array_like): current_action
            reward(float): reward returned with current state/action
            next_state(array_like): next_state after current state/action
            done(bool): whether episode ends  
        """
        self.memory.add(state, action, reward, next_state, done)

        # learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # check whether there are eough samples in reply buffers
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Return actions for a given state per current policy

        Params
        ======
            state(array_like): current state
            eps(float): epsilon, for epsilon-greedy action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Run through Epsilon-greedy policy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update Q function parameters/weights using given batch of experience

        Params
        ======
        experiences (Tuple[torch.tensor]): tuple of (s, a, r, s', done)
        gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Q_targets
        if self.algo == 'ddqn':
            # use local network to select optimal action and use target network to evalute the value
            actions_max = self.qnetwork_local(next_states).detach().max(
                1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(
                1, actions_max)
        elif self.algo == 'dqn':
            # use target_network to select optimal action and evaluate its value
            Q_targets_next = self.qnetwork_target(next_states).detach().max(
                1)[0].unsqueeze(1)
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)

        # Q_expected
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_targets, Q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters
        𝜃_target = τ * 𝜃_local + (1-τ) * 𝜃_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from 
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)
