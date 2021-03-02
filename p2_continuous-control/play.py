"""Testing Module for a RL Agent trained with DDPG algorithm.
"""

from unityagents import UnityEnvironment
import numpy as np
import ddpg_agent as ddpga
import torch
import matplotlib.pyplot as plt
from torchsummary import summary

env = UnityEnvironment(file_name='./Reacher-2.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

# create agent from checkpoint
agent = ddpga.Agent(state_size, action_size, random_seed=15)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

rounds = 5
scores_all = []

for r in range(rounds):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations

    scores = np.zeros(num_agents)

    while True:
        actions = agent.act(
            states, add_noise=False)  # select an action (for each agent)
        env_info = env.step(actions)[
            brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            scores_all.append(np.mean(scores))
            print(
                f'Total score (averaged over agents) {r} episode: {np.mean(scores)}'
            )
            break

