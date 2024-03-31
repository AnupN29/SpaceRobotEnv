import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImitationAgent(nn.Module):
    def __init__(self, output_channels=6, device=device):
        super(ImitationAgent, self).__init__()
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
        self.fc_combined = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU()
        ).to(device)
        self.fc_output = nn.Linear(16, out_features=output_channels).to(device)
        
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
        actions = self.fc_output(combined_features)
        return actions