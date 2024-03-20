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
            nn.Linear(1024, 64),
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


class CNNQFunction(nn.Module):

    def __init__(self, act_dim, hidden_sizes, activation,device=device):
        super().__init__()
        self.critic  = nn.Sequential(
           nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=5,  stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(28800, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.q = mlp([64 + act_dim] + list(hidden_sizes) + [1], activation).to(device)

    def forward(self, obs, act):
        cnn_out = self.critic(obs)
        q = self.q(torch.cat([cnn_out, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.
    
class CNNActorCritic(nn.Module):

    def __init__(self, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, device=device):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = CNNActor(act_dim, hidden_sizes, activation, act_limit, device)
        self.q1 = CNNQFunction(act_dim, hidden_sizes, activation, device)
        self.q2 = CNNQFunction(act_dim, hidden_sizes, activation, device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()