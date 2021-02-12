"""
Module to traing RL agent.
"""

import argparse
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from unityagents import UnityEnvironment
from ddqn_agent import Agent


class TrainingParams:
    """
    Manage the training parameters

    Params
    ======
    n_episodes (int): max number of training episodes
    max_t (int): maximum number of timesteps per episode
    eps_start (float): starting value of epsilon, for epsilon-greedy policy
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    n_episodes = 2000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.99

    def __repr__(self):
        return f'number of training episodes: {self.n_episodes}, max number of timesteps per episode: {self.max_t}\n'\
        f'epsilon-greedy policy, start value: {self.eps_start}, end value: {self.eps_end}, decay rate {self.eps_decay}'


def run_training_loop(checkpoint_file: str, tp: TrainingParams, algo: str):
    """
    Setup environment for agent to interact and learn

    Params
    ======
    checkpoint_file: file name for check pointing. 
    tp: a set of parameters for training 
    """
    # setup environment
    env = UnityEnvironment(file_name="./Banana.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # get the state_size and action_size of the env
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    # create the agent
    agent = Agent(state_size, action_size, seed=10, algo=algo)

    scores = []
    scores_window = deque(maxlen=100)
    best_avg_score = 0.0
    eps = tp.eps_start

    print(f'Training parameters:\n{tp}')

    for i_episode in range(1, tp.n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0.0
        for t in range(tp.max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[
                brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(tp.eps_end, eps * tp.eps_decay)
        print(f'\rEpisode {i_episode}\tScore: {score}')
        if i_episode % 100 == 0:
            print('\rEpisode {},\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if i_episode > 100 and np.mean(scores_window) > best_avg_score:
            print(
                f'\nBest avg score improved in last 100 of {i_episode} episodes! Average Score: {np.mean(scores_window):.2f}. Check pointing.'
            )
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_file)
            best_avg_score = np.mean(scores_window)

    env.close()
    return scores


def plot_scores(scores, filename, title, rolling_window=100):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(title)
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # use predefined set of training parameters
    training_params = TrainingParams()

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', '-a', type=str, default='ddqn')
    args = parser.parse_args()

    algo = args.algo
    scores = run_training_loop(f'{algo}_model.pt', training_params, algo)
    plot_scores(scores, f'{algo}_scores.png', f'{algo} - Training scores')
