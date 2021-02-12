from unityagents import UnityEnvironment
import numpy as np
import torch
from ddqn_agent import Agent

TOTAL_PLAYS = 100

env = UnityEnvironment(file_name="./Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# get the state and action size from env
env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

# initialize agent from trained model
algo = 'ddqn'
agent = Agent(state_size, action_size, seed=10, algo=algo)
print(f'Create agent from trained {algo} model')
agent.qnetwork_local.load_state_dict(torch.load(f'{algo}_model.pt'))

# track game plays
scores = []
total_tries = 0

while total_tries < TOTAL_PLAYS:
    score = 0
    done = False
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    while not done:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            scores.append((total_tries, score))
            print(f'Game play, round {total_tries}, score {score}')
    total_tries += 1

print(f'Average score for {TOTAL_PLAYS} plays: {sum(scores)/TOTAL_PLAYS}')

env.close()
