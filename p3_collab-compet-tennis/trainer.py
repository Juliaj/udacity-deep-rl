'''Run training for Tennis game
'''
import argparse
from collections import deque
import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

import maddpg_agent_group as mag
from hyperparams import HyperParams
import utils


def setup_logger(result_dir, name='maddpg'):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger.setLevel(logging.DEBUG)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    filehandler = logging.FileHandler(filename=f'{result_dir}/{name}.log')
    filehandler.setFormatter(formatter)
    filehandler.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    # Uncomment to enable console logger
    steamhandler = logging.StreamHandler()
    steamhandler.setFormatter(formatter)
    steamhandler.setLevel(logging.INFO)
    logger.addHandler(steamhandler)

    return logger


def train(
    agent_group: mag.MADDPGAgentGroup,
    env: UnityEnvironment,
    brain_name: int,
    num_agents: int,
    writer: utils.VisWriter,
    result_dir: str,
    logger,
    num_episodes: int = 10000,
    max_t: int = 5000,
    print_every: int = 100,
    passing_score: float = 0.5,
):
    scores_deque = deque(maxlen=print_every)
    max_t_deque = deque(maxlen=print_every)

    i_episode = 0
    scores = []
    current_t = 0
    best_max_t = 0

    for i_episode in range(1, num_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        score = np.zeros(num_agents)
        agent_group.reset()

        critic_loss, actor_loss = 0, 0
        t = 0
        for t in range(max_t):
            # agent acts
            actions = agent_group.act(states)

            if current_t % print_every == 0:
                for i in range(actions[0].shape[0]):
                    action_from_dim = [a[i] for a in actions]
                    writer.push(action_from_dim, f'Actions(dim-{i})')

            # receives feedback from env
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            score += rewards

            # agent explores or learns
            critic_loss, actor_loss = agent_group.step(states, actions,
                                                       rewards, next_states,
                                                       dones)

            states = next_states

            if np.any(dones):
                logger.debug('Episode {} done at t = {}'.format(i_episode, t))
                if t >= best_max_t:
                    best_max_t = t
                    max_t_deque.append(best_max_t)
                break

            current_t += 1

        max_score = np.max(score)
        scores_deque.append(max_score)
        scores.append(max_score)
        current_score = np.mean(scores_deque)

        # keep track of scores
        utils.save_to_txt(current_score, '{}/scores.txt'.format(result_dir))

        logger.info(
            f'Episode {i_episode}, score : {max_score:.3f}. Average score: {current_score:.3f}. (critic_loss: {critic_loss:.7f}, actor_loss:{actor_loss:.7f})'
        )

        # Publish and save
        writer.text(
            'Episode {}/{}: Average score(100): {}'.format(
                i_episode, num_episodes, current_score),
            'Average 100 episodes')
        writer.push(current_score, 'Average Score')
        logger.info(
            'Episode {}\tAverage Score: {:.2f}, Average max_t: {:.2f}, Best max_t: {}'
            .format(i_episode, current_score, np.mean(max_t_deque),
                    best_max_t))

        if len(scores) > 0:
            writer.push(scores[-1], 'Score')

        if current_score >= passing_score:
            logger.info(
                f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_deque):.4f}, passing score: {passing_score}. Saving models.'
            )
            break

        # save models
        agent_group.save()

    return scores


def plot(scores, title, result_dir):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(title)
    plt.savefig(f'{result_dir}/{title}_scores.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_episodes',
                        type=int,
                        default=10000,
                        help='Total number of training episodes')
    parser.add_argument('--max_t',
                        type=int,
                        default=5000,
                        help='Max timestep in a single episode')
    parser.add_argument('--vis',
                        dest='vis',
                        action='store_true',
                        help='Use visdom to visualize training')
    parser.add_argument('--hyperparams',
                        type=str,
                        help='hyperparameter yaml file')
    parser.add_argument('--result_dir',
                        type=str,
                        default='results',
                        help='Use visdom to visualize training')
    parser.set_defaults(vis=False)

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

    # create folder for all artifacts
    result_dir = utils.path_gen()
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # setup logger
    algo = 'maddpg'
    logger = setup_logger(result_dir, algo)
    print(f'run folder {result_dir}')

    hparams = HyperParams()

    # initialize hyperparameters from file
    if args.hyperparams:
        logger.info('loading hyperparameters from file %s', args.hyperparams)
        hparams.__dict__ = utils.load_from_yaml(args.hyperparams)

    # save configuration as yaml to result dir
    utils.save_to_yaml(hparams, os.path.join(result_dir, 'hyperparams.yaml'))

    # create writer to push events to Visdom
    writer = utils.VisWriter(vis=args.vis)
    agent_group = mag.MADDPGAgentGroup(state_size,
                                       action_size,
                                       num_agents,
                                       writer,
                                       hparams,
                                       result_dir=result_dir)

    # start training
    scores = train(agent_group,
                   env,
                   brain_name,
                   num_agents,
                   writer,
                   result_dir,
                   logger,
                   num_episodes=args.num_episodes,
                   max_t=args.max_t)

    # plot scores
    plot(scores, algo, result_dir)


if __name__ == '__main__':
    main()
