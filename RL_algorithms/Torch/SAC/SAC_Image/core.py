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

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoderNetwork(nn.Module):
    def __init__(self, output_channels=21, device=device):
        super(ImageEncoderNetwork, self).__init__()
        # Define convolutional layers for processing RGB image
        self.rgb_conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ).to(device)
        # Define convolutional layers for processing depth image
        self.depth_conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ).to(device)
        # Define fully connected layer to combine features
        self.fc_combined = nn.Linear(64 * 2 * 2, 64).to(device)
        self.fc_output = nn.Linear(64, out_features=output_channels).to(device)
        
    def forward(self, rgb_image, depth_image):
        # Process RGB image
        rgb_features = self.rgb_conv_layers(rgb_image)
        # Process depth image
        depth_features = self.depth_conv_layers(depth_image)
        # Concatenate features
        combined_features = torch.cat((rgb_features, depth_features), dim=1)
        combined_features = combined_features.view(combined_features.size(0), -1)  # Flatten
        # Fully connected layer
        combined_features = F.relu(self.fc_combined(combined_features))
        output = self.fc_output(combined_features)
        return output


# # Example usage:
# input_channels = 3  # For RGB image
# output_channels = 64  # Number of output channels


# # Sample input tensors
# rgb_image = torch.randn(1, input_channels, 64, 64)  # Example shape: (batch_size, channels, height, width)
# depth_image = torch.randn(1, 1, 64, 64)  # Example shape: (batch_size, channels, height, width)

# # Forward pass
# output = image_encoder(rgb_image, depth_image)
# print("Output shape:", output.shape)  # Check the shape of the output


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, act_dim, image_encoder, hidden_sizes, act_limit, device=device):
        super().__init__()
        self.image_encoder = image_encoder
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim).to(device)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim).to(device)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.image_encoder(obs[0], obs[1])  # Pass observation through the image encoder
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
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(torch.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, act_dim, image_encoder, hidden_sizes, activation, device=device):
        super().__init__()
        self.image_encoder = image_encoder
        self.q = mlp([hidden_sizes[-1] + act_dim] + list(hidden_sizes) + [1], activation).to(device)

    def forward(self, obs, act):
        net_out = self.image_encoder(obs[0], obs[1])  # Pass observation through the image encoder
        q = self.q(torch.cat([net_out, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, action_space, image_encoder, hidden_sizes=(256,256),
                 activation=nn.ReLU, device=device):
        super().__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(act_dim, image_encoder, hidden_sizes, act_limit, device)
        self.q1 = MLPQFunction(act_dim, image_encoder, hidden_sizes, activation, device)
        self.q2 = MLPQFunction(act_dim, image_encoder, hidden_sizes, activation, device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            try:
                return a.cpu().numpy()
            except:
                return a.detach().numpy()


