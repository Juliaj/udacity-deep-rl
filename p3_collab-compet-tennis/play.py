"""
Testing module for a trained Agent
"""
import torch
from unityagents import UnityEnvironment
import numpy as np
import maddpg
import ddpg_agent as ddpga

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
        states = env_info.vector_observations
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


class Config:
    """Define training parameters.
    """
    update_every = 1
    batch_size = 512
    buffer_size = int(1e6)
    gamma = 0.99
    tau = 0.2
    seed = 2
    lr_actor = 1e-4
    lr_critic = 1e-4
    action_size = action_size
    state_size = state_size
    num_agents = num_agents


# random_play()
def play_tennis(rounds=10):
    config = Config()
    # agent = ddpga.Agent(state_size, action_size, random_seed=15)
    multi_agent = maddpg.MADDPG(Config())
    for player in multi_agent.agents:
        player.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        player.critic_local.load_state_dict(
            torch.load('checkpoint_critic.pth'))

    for i in range(1, rounds):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        multi_agent.reset()

        while True:
            actions = multi_agent.act(states, add_noise=False)
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
