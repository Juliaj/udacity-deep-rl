"""Agent implementation with MADDPG algorithm

Adapted from: https://github.com/katnoria/unityml-tennis/blob/master/maddpg.py
"""

import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

import hyperparams as hp
from models import Actor, Critic
import noise
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MADDPGAgent():
    """Multi Agent DDPG Implementation.
    Alogrithm: https://arxiv.org/abs/1706.02275.
    """
    def __init__(self,
                 num_agents: int,
                 state_size: int,
                 action_size: int,
                 agent_id: int,
                 writer: utils.VisdomWriter,
                 hparams: hp.HyperParams,
                 result_dir: str = 'results',
                 print_every=1000,
                 model_path=None,
                 saved_config=None,
                 eval_mode=False):
        """Initialize an Agent object
        Params
        =====
            num_agents: number of agents in the game
            state_size: dimension of the state for each agent
            action_size: dimension of the action space for each agent
            agent_id: id(index) for current agent
            writer: for realtime training visualization
            hparams: a set of hyper parameters
            result_dir: relative_path for saving artifacts
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.writer = writer

        # random.seed(self.seed)
        self.seed = hparams.seed
        random.seed(self.seed)
        # param for critic loss calculation
        self.gamma = hparams.gamma
        # param for soft update
        self.tau = hparams.tau

        # learning rates
        self.lr_actor = hparams.lr_actor
        self.lr_critic = hparams.lr_critic
        # param for critic optimizer initialization
        self.weight_decay = hparams.weight_decay

        # Critic network
        self.critic_local = Critic(
            self.num_agents,
            self.state_size,
            self.action_size,
            self.seed,
            fcs1_units=hparams.critic_fcs1_units,
            fc2_units=hparams.critic_fc2_units).to(device)
        self.critic_target = Critic(
            self.num_agents,
            self.state_size,
            self.action_size,
            self.seed,
            fcs1_units=hparams.critic_fcs1_units,
            fc2_units=hparams.critic_fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.lr_critic,
                                           weight_decay=self.weight_decay)

        # Actor network
        self.actor_local = Actor(self.state_size,
                                 self.action_size,
                                 self.seed,
                                 fc1_units=hparams.actor_fc1_units,
                                 fc2_units=hparams.actor_fc2_units).to(device)
        self.actor_target = Actor(self.state_size,
                                  self.action_size,
                                  self.seed,
                                  fc1_units=hparams.actor_fc1_units,
                                  fc2_units=hparams.actor_fc2_units).to(device)
        self.actor_optimzer = optim.Adam(self.actor_local.parameters(),
                                         lr=self.lr_actor)

        # Noise Process for action space exploration
        self.noise = noise.OUNoise(action_size, self.seed, sigma=hparams.sigma)

        # Replay buffer
        self.buffer_size = hparams.buffer_size
        self.batch_size = hparams.batch_size

        self.learn_step = 0
        self.result_dir = result_dir

    def act(self, state: np.ndarray, add_noise=True):
        """Returns actions for given states as per current policy
        Params:
            states: game states from environment
            add_noise: whether we should apply noise. True when in training, otherwise false
        Return:
            action clipped
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, agents, experiences, gamma: float):
        """Update policy and value parameters using giving batch of experince tuples
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
    
        For MDDDPG, agent critic uses states, actions from all agents to reduce the effect from 
        non-stationary environment
        Each agent draws action based on its own state

        Params:
        =====
            agents: MADDPGAgent objects
            experiences: sampled experiences from agents
            gamma:discount factor for Q-target calculation

        Return:
            critic_loss and actor_loss
        """

        # unpack
        states, actions, rewards, next_states, dones = experiences
        # ------------ updata critic -----------------#
        # get predicted actions from all agents actor_target network
        next_pred_actions = torch.zeros(len(actions), self.num_agents,
                                        self.action_size).to(device)
        for i, agent in enumerate(agents):
            next_pred_actions[:, i] = agent.actor_target(next_states[:, i, :])

        # flatten states and actions to produce inputs to Critic network
        critic_next_states = utils.flatten(next_states)
        next_pred_actions = utils.flatten(next_pred_actions)

        Q_targets_next = self.critic_target(critic_next_states,
                                            next_pred_actions)

        # compute the target Q-value, update current agent only
        Q_targets = rewards[:, self.agent_id, :] + gamma * Q_targets_next * (
            1 - dones[:, self.agent_id, :])

        # use local network to get expected Q-value and calculate loss
        critic_states = utils.flatten(states)
        critic_actions = utils.flatten(actions)
        Q_expected = self.critic_local(critic_states, critic_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip the gradient, how to decide the max_norm?
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ------------ update actor -----------------#
        # compute actor loss
        pred_actions = torch.zeros(len(actions), self.num_agents,
                                   self.action_size)
        pred_actions.data.copy_(actions.data)
        # update action for this agent only !
        pred_actions[:,
                     self.agent_id] = self.actor_local(states[:,
                                                              self.agent_id])
        critic_states = utils.flatten(states)
        critic_pred_actions = utils.flatten(pred_actions)
        actor_loss = -self.critic_local(critic_states,
                                        critic_pred_actions).mean()
        self.actor_optimzer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimzer.step()

        # ------------ update target networks --------#
        if self.learn_step == 0:
            utils.hard_update(self.actor_local, self.actor_target)
            utils.hard_update(self.critic_local, self.critic_target)
        else:
            utils.soft_update(self.actor_local, self.actor_target, self.tau)
            utils.soft_update(self.critic_local, self.critic_target, self.tau)

        self.learn_step += 1
        # return the losses
        return actor_loss.item(), critic_loss.item()

    def check_point(self):
        """
        Save model checkpoints and configurations
        """
        critic_pth = os.path.join(self.result_dir,
                                  f'checkpoint_critic_{self.agent_id}.pth')
        actor_pth = os.path.join(self.result_dir,
                                 f'checkpoint_actor_{self.agent_id}.pth')

        torch.save(self.actor_local.state_dict(), actor_pth)
        torch.save(self.critic_local.state_dict(), critic_pth)
