[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This project trains an agent to navigate and collect bananas in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. The agent learns how to best select one of following actions:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic.  

Two training algorithms Deep Q-learning and Double DQN, are implemented. The model contains 3 layers with `relu` as an activation function.

Report.html contains training results for both algorithms.

### Getting Started

1. Download the Unity Banana environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. Install dependencies.
```
$ cd p1_navigation
$ pip install -r requirements.txt
```

### Instructions

1. Train agent
```
$ cd p1_navigation
$ python train.py --algo ddqn  
```
After training is done, the model is saved to {algo}_model.pt file. You can also find a correspoing score plot generated.

2. Play the game

Running following command with the newly training agent to play 100 rounds of game. 

```
$ cd p1_navigation
$ python play.py
```
At the end of the game, average score from the play is printed out to console.