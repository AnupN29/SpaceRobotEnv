import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    # converts array of layer shape to neural net
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes) -2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class CNNActor(nn.Module):
    def __init__(self, act_dim, hidden_sizes, activation, act_limit, device=device):
        super().__init__()

        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=5,  stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=5,  stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=5,  stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            # nn.Linear(1024, 4096),
            nn.Linear(1024, 64), #changed
            nn.ReLU(),
            # nn.Linear(4096, 256),
            # nn.ReLU(),
            # nn.Linear(256, 64),
            # nn.ReLU()
        ).to(device)
        
        self.net = mlp([64] + list(hidden_sizes), activation, activation).to(device)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim).to(device)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim).to(device)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        cnn_out = self.cnn_backbone(obs)
        net_out = self.net(cnn_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi
    

from RL_algorithms.Torch.SAC.SAC_ENV import core

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.exp_act_buf = torch.zeros(core.combined_shape(size, act_dim), dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, expert_act):
        self.obs_buf[self.ptr] = obs
        self.exp_act_buf[self.ptr] = expert_act
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, device=device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     expert_act=self.exp_act_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k,v in batch.items()}