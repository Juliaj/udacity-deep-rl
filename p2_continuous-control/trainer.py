"""Trainer for Reacher.

Reacher-2 - Unity simulation environment with 20 agent 
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from collections import deque
import ddpg_agent as ddpga
import time

from unityagents import UnityEnvironment

env = UnityEnvironment(file_name='./Reacher-2.app')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print(f'Number of agents: {num_agents}')

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print(
    f'There are {states.shape[0]} agents. Each observes a state with length: {state_size}'
)
print(f'The state for the first agent looks like: {states[0]}')


def train(n_episodes=5000, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores_all = []
    i_episode = 0

    for i_episode in range(1, n_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()

        for t in range(max_t):
            # agent acts
            actions = agent.act(states)
            # receives feedback from env
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # agent learns
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        scores_deque.append(np.mean(scores))
        scores_all.append(np.mean(scores))

        print(
            f'Episode {i_episode} Reward: {np.mean(scores):.3f} ... Average Reward: {np.mean(scores_deque):.3f}'
        )

        if np.mean(scores_deque) >= 30.0:
            print(
                f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_deque):.3f}. Saving models'
            )
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       'checkpoint_critic.pth')
            break

    return scores_all


# create agent and start training
agent = ddpga.Agent(state_size, action_size, random_seed=15)
scores = train()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('ddpg-reacher.png')
plt.show()
