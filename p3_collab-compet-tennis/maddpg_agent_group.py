"""
Manage the group of agents' interaction with environment and learning
"""

import numpy as np
import random

from memory import ReplayBuffer
import maddpg_agent as ma
import utils


class MADDPGAgentGroup:
    """Group the MADDPG agents as a single entity"""
    def __init__(
            self,
            #  env,
            state_size,
            action_size,
            num_agents,
            writer,
            hparams,
            print_every=1000,
            result_dir='results'):
        self.num_agents = num_agents
        # self.env = env
        # self.brain_name = self.env.brain_names[0]
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = hparams.batch_size
        self.buffer_size = hparams.buffer_size
        self.seed = hparams.seed
        self.update_every = hparams.update_every
        random.seed(self.seed)
        self.writer = writer
        self.result_dir = result_dir

        self.hparams = hparams
        self.agents = [
            ma.MADDPGAgent(self.num_agents,
                           self.state_size,
                           self.action_size,
                           i,
                           self.writer,
                           self.hparams,
                           result_dir=self.result_dir)
            for i in range(self.num_agents)
        ]

        self.gamma = hparams.gamma

        self.memory = ReplayBuffer(
            self.buffer_size,
            self.batch_size,
            self.hparams.seed,
        )
        self.print_every = print_every
        self.learn_step = 0
        self.critic_loss = 0.0
        self.actor_loss = 0.0

    def act(self, states, add_noise=True):
        """Executes act on all the agents
        Parameters:
            states (list): list of states, one for each agent
            add_noise (bool): whether to apply noise to the actions
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(states[i], add_noise)
            actions.append(action)
        return actions

    def reshape(self, states, actions, rewards, next_states, dones):
        """Reshape the inputs
        """
        # adding axis=0 to states, actions, and next_states
        states = np.expand_dims(states, axis=0)
        next_states = np.expand_dims(next_states, axis=0)
        assert (states.shape[0] == 1 and states.shape[1] == self.num_agents
                and states.shape[2] == self.state_size)

        actions = np.expand_dims(actions, axis=0)
        assert (actions.shape[0] == 1 and actions.shape[1] == self.num_agents
                and actions.shape[2] == self.action_size)

        # for rewards and dones, reshape then add axis=0
        rewards = np.expand_dims(np.array(rewards).reshape(
            self.num_agents, -1),
                                 axis=0)
        assert (rewards.shape[0] == 1 and rewards.shape[1] == self.num_agents
                and rewards.shape[2] == 1)
        dones = np.expand_dims(np.array(dones).reshape(self.num_agents, -1),
                               axis=0)

        return states, actions, rewards, next_states, dones

    def step(self, states, actions, rewards, next_states, dones):
        """Performs the learning step.
        """
        # store a single entry for results from all agents by adding axis=0
        states, actions, rewards, next_states, dones = self.reshape(
            states, actions, rewards, next_states, dones)
        self.memory.add(states, actions, rewards, next_states, dones)

        # Get agent to learn from experience if we have enough data/experiences in memory
        if len(
                self.memory
        ) > self.batch_size and self.learn_step % self.update_every == 0:

            experiences = self.memory.sample()
            actor_losses = []
            critic_losses = []

            for agent in self.agents:
                actor_loss, critic_loss = agent.learn(self.agents, experiences,
                                                      self.gamma)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            # Plot real-time graphs and store losses
            if self.learn_step % self.print_every == 0:
                # Save Critic loss
                utils.save_to_txt(
                    critic_losses,
                    '{}/critic_losses.txt'.format(self.result_dir))
                self.writer.text('critic loss: {}'.format(critic_losses),
                                 'Critic')
                self.writer.push(critic_losses, 'Loss(critic)')
                # Save Actor loss
                utils.save_to_txt(
                    actor_losses,
                    '{}/actor_losses.txt'.format(self.result_dir))
                self.writer.text('actor loss: {}'.format(actor_losses),
                                 'Actor')
                self.writer.push(actor_losses, 'Loss(actor)')

            self.critic_loss = np.array(critic_losses).mean()
            self.actor_loss = np.array(actor_losses).mean()
            self.learn_step += 1

        return self.critic_loss, self.actor_loss

    def reset(self):
        """Resets the noise for each agent"""
        for agent in self.agents:
            agent.reset()

    def save(self):
        """Checkpoint actor and critic models"""
        for agent in self.agents:
            agent.check_point()
