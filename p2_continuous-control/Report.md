# Continous Control Project
To solve Reacher, DDPG algorithm is implemented to train a RL agent. This report summarizes the training scores, performance with the agent and observations. 

### Model Architecture
DDPG algorithm involves both Actor and Critic. It requires four networks and a pair of local and target netwrok for Actor and Critic. In this implementation, local and target network shares the same architecture. 
- Actor
  - input: state, dim = 33, continous.
  - layers: 
```
Actor(
  (fc1): Linear(in_features=33, out_features=256, bias=True)
  (relu):
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (relu):
  (fc3): Linear(in_features=128, out_features=4, bias=True)
  (tanh)
)
```
- Critic
  - input: state, dim = 33, continous; action, dim = 4, continous
  - layers:
```
Critic(
  (fcs1): Linear(in_features=33, out_features=256, bias=True)
  (relu):
  (fc2): Linear(in_features=256 + 4, out_features=128, bias=True)
  (relu):
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

### Training  

- Training parameters:
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of parameters
LR_ACTOR = 1e-4  # learning rate for actor
LR_CRITIC = 1e-4  # learning rate for critic
WEIGHT_DECAY = 0  # L2 weight decay  
```

#### Training results

- Average Score of 30 was reached around 40 episode. 

![Alt text](./ddpg-reacher.png?raw=true "Traing scores")

- Testing Performance 

The average score for DDPG agent playing the game is 38 for 20 consecutive rounds.


### Observations 

Frequent updates to Actor and Critic network seem to slow down training drastically and rewards stopped accumulating. With the current implemenation, even though there are 20 agents to interact with environment and collect experiences, per epsiode, there is only one update made to the network. 

The noise implementation with Ornstein-Uhlenbeck process constains two different options to calcuate the `dx`

```python
if self.wiener_random:
    dx = self.theta * (self.mu - x) + self.sigma * np.array(
                [random.random() for i in range(len(x))])
else:
    dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
```                               
The final training code used the option with `np.random.standard_normal` and the learning was significantly faster than the other option. The difference may be due to too much randomness introduced by `random.random()`.                                  

### Futher Improvement
Additional turning of the hyper parameters for models such as hidden layer size can be done to further spend up agent training. 

State space is a 33 dimension continous, adding layer `nn.BatchNorm1d` to Actor and Critic netowrk could be benefical for normalization. 


