[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

This repo provides a RL agent trained with DDPG (Deterministic Deep Policy Gradient, https://arxiv.org/pdf/1509.02971.pdf) algorithm for Reacher. 

Reacher is an environment which a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 

![Trained Agent][image1]

The RL agent interacts with a version of [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) with 20 identical agents, each with its own copy of the environment.

- The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

- Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Getting Started

1. Download the Reather environment from one of the links below.  You need only select the environment that matches your operating system:
    - Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

3. Python package dependencies can be found in requirements.txt.

### Training

```
$ cd p2_continous_control
$ python trainer.py 
```
After training is done, the model parameters are saved to two files: checkpoint_actor.pth and checkpoint_critic.pth. You can also find a correspoing score plot generated.

### Playing the game

```
$ cd p2_continous_control
$ python play.py 
```