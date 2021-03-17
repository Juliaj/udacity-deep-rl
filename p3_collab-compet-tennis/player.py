"""
Testing module for a trained Agent
"""
import torch
from unityagents import UnityEnvironment
import numpy as np

from hyperparams import HyperParams
import maddpg_agent_group as mag
import utils

env = UnityEnvironment(file_name='./Tennis.app')

# get default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

print(
    f'There are {num_agents} agents. Each observes a state with length {state_size}. Agent\'s action space has {action_size} dimensions.'
)

def random_play(rounds=6):
    for i in range(1, rounds):
        env_info = env.reset(train_mode=False)[brain_name]
        scores = np.zeros(num_agents)

        while True:
            actions = np.random.randn(num_agents, action_size)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # agent learns
            scores += rewards
            states = next_states
            if np.any(dones):
                break
        print(f'Score (max over agents) from episode {i}: {np.max(scores)}')


def play_tennis(rounds=10):
    hparams = HyperParams()
    writer = utils.VisWriter(vis=False)
    agent_group = mag.MADDPGAgentGroup(state_size,
                                    action_size,
                                    num_agents,
                                    writer,
                                    hparams,
                                    result_dir='./')

    for i, agent in enumerate(agent_group.agents):
        agent.actor_local.load_state_dict(torch.load(f'./runs/2021-03-16-215701/checkpoint_actor_{i}.pth'))
        agent.critic_local.load_state_dict(
            torch.load(f'./runs/2021-03-16-215701/checkpoint_critic_{i}.pth'))

    for i in range(1, rounds):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent_group.reset()

        while True:
            actions = agent_group.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # agent learns
            scores += rewards
            states = next_states
            if np.any(dones):
                break
        print(f'Score (max over agents) from episode {i}: {np.max(scores)}')


play_tennis()
env.close()
