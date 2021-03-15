[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

This project provides a Reinforment Learning agent to play Tennis game against each other in a simulated [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment by Unity. In this environment, two agents control rackets to bounce a ball over a net. The goal of each agent is to keep the ball in play.

![Trained Agent][image1]

- Reward and **score** of an episode:
    - if an agent hits the ball over the net, it receives a reward of +0.1.  
    - If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
    - The playing task is episodic and ends when ball hits the ground or falls outside the bounds. After each episode, the rewards that each agent received (without discounting) are added to get a score for that agent. This yields 2 (potentially different) scores. The **score** of each episode is the maximum of these 2 scores. 

- Observation space per agent: 3 stacked vector observations where each consists of 8 variables corresponding to the position and velocity of the ball and racket with. The dimension is 24. Each agent receives its own, local observation.  

- Action space per agent: two continuous actions corresponding to movement toward (or away from) the net, and jumping. All actions are between -1 and 1.

The agent solves the environment with the average (over 100 episodes) of those **scores** > +0.5. 

### Getting Started

1. Download the Tennis environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

3. Install Python dependency package by running
```
cd p3_collab-compet-tennis
pip install -r requirements.txt
```

### Training

```
$ cd p3_collab-compet-tennis
$ python trainer.py 
```
After training is done, the model parameters are saved to two files: checkpoint_actor.pth and checkpoint_critic.pth. You can also find a correspoing score plot generated.

### Playing the game

```
$ cd p3_collab-compet-tennis
$ python play.py 
```

