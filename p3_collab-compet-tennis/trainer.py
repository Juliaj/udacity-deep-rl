"""Run training for Tennis game
"""
import argparse
import numpy as np
import os

import matplotlib.pyplot as plt
from collections import deque

from unityagents import UnityEnvironment

import maddpg_agent_group as mag
from hyperparams import HyperParams
import utils
from icecream import ic

def train(agent_group:mag.MADDPGAgentGroup,
          env:UnityEnvironment,
          brain_name:int,
          num_agents:int,
          writer :utils.VisWriter,
          result_dir,
          num_episodes: int=5000,
          max_t: int=1000,
          print_every: int=100,
          ):
    scores_deque = deque(maxlen=print_every)
    
    i_episode = 0
    scores = []
    for i_episode in range(1, num_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        # ic('in train', states.shape)

        score = np.zeros(num_agents)
        agent_group.reset()

        critic_loss, actor_loss = 0, 0
        for _ in range(max_t):
            # agent acts
            actions = agent_group.act(states)

            # receives feedback from env
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # agent learns
            critic_loss, actor_loss = agent_group.step(states, actions,
                                                       rewards, next_states,
                                                       dones)
            states = next_states
            score += rewards
            if np.any(dones):
                break

        max_score = np.max(score)
        scores_deque.append(max_score)
        scores.append(max_score)
        avg_score = np.mean(scores_deque)
        print(
            f'Episode {i_episode}, score : {max_score:.3f}. Average score: {avg_score:.3f}. (critic_loss: {critic_loss:.7f}, actor_loss:{actor_loss:.7f})'
        )

        # Publish and save
        writer.text('Episode {}/{}: Average score(100): {}'.format(i_episode, num_episodes, avg_score), "Average 100 episodes")
        writer.push(avg_score, "Average Score")

        if len(scores) > 0:
            writer.push(scores[-1], "Score")

        if avg_score >= 0.5:
            print(
                f'\nEnvironment solved in {i_episode-100:d} games!\tAverage Score: {np.mean(scores_deque):.4f}. Saving models'
            )
            break

        # save models
        agent_group.save()
        utils.save_to_txt(scores, os.path.join(result_dir, 'scores.txt'))
    return scores


def plot(scores, title):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_episodes", type=int, default=1000, help="Total number of training episodes")
    parser.add_argument("--max_t", type=int, default=1000, help="Max timestep in a single episode")
    parser.add_argument("--vis", dest="vis", action="store_true", help="Use visdom to visualize training")
    parser.add_argument("--result_dir", type=str, default="results", help="Use visdom to visualize training")
    parser.set_defaults(vis=True)
 
    args = parser.parse_args()

    # initialize game env
    env = UnityEnvironment(file_name='./Tennis.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    # create agent group
    hparams = HyperParams()
    
    # create folder for all artifacts
    path = utils.path_gen(hparams)
    result_dir = f'{args.result_dir}/{path}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print(f'results folder {result_dir}')
    # save configuration
    utils.save_to_json(hparams.__dict__, os.path.join(result_dir, "hyperparams.json"))

    # visualizer
    writer = utils.VisWriter(vis=True)
    agent_group = mag.MADDPGAgentGroup(env, state_size, action_size,
                                       num_agents, writer, hparams, result_dir=result_dir)

    # play and train
    scores = train(agent_group, env, brain_name, num_agents, writer, result_dir, num_episodes=args.num_episodes, max_t=args.max_t)

    # plot scores
    train_algo = 'maddpg'
    plot(scores, train_algo)


if __name__ == '__main__':
    main()
