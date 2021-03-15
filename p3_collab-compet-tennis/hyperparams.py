"""
Capture training hypterparameters
"""


class HyperParams:
    """Define training parameters.
    """
    update_every = 1        # how often to learn and update network
    batch_size = 512        # mini batch size
    buffer_size = int(1e6)  # replay buffer capacity
    gamma = 0.95            # discount factor for Q target calculation
    tau = 0.02              # for soft update of target network parameters
    seed = 48               # random seed
    lr_actor = 1.e-3        # learning rate for actor
    lr_critic = 1.e-3       # learning rate for critic
    weight_decay = 0        # L2 weight decay
    
    # nn Network
    actor_fc1_units = 64    # actor fc1 output dim
    actor_fc2_units = 64    # actor fc2 output dim
    critic_fcs1_units = 512 # critic fc state processing layer output dim
    critic_fc2_units = 256  # critic fc2 output dim

    # noise
    sigma = 0.1             # std of noise

    def load(self, file):
        """Load parameters from configuration file
        """
        pass
