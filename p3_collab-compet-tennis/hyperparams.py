"""
Capture training hypterparameters
"""
import json
import yaml


class HyperParams:
    """Define training parameters.
    """
    def __init__(self):
        self.buffer_size = int(1e6)  # replay buffer capacity
        self.batch_size = 512  # mini batch size

        # DDPG
        self.gamma = 0.95  # discount factor for Q target calculation
        self.tau = 0.02  # for soft update of target network parameters
        self.lr_actor = 1.e-3  # learning rate for actor
        self.lr_critic = 1.e-3  # learning rate for critic
        self.weight_decay = 0  # L2 weight decay
        self.update_every = 1  # how often to learn and update network
        self.clip_grad = True  # whether to clip gradients
        self.clamp_value = 1  # clip value

        # seed
        self.seed = 48  # random seed

        # nn Network
        self.actor_fc1_units = 32  # actor fc1 output dim
        self.actor_fc2_units = 32  # actor fc2 output dim
        self.critic_fcs1_units = 64  # critic fc state processing layer output dim
        self.critic_fc2_units = 64  # critic fc2 output dim
        self.use_batchnorm = False  # apply batchnorm in the network

        # noise
        self.sigma = 0.1  # std of noise

    def __repr__(self):
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=True,
                          indent=4)
