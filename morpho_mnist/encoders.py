# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, latent_dim=512):
        super().__init__()

        self.latent_dim = latent_dim
        self.shape = 32

        #encode
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*(self.shape-1)**2, 2*self.latent_dim)
        self.relu = nn.ReLU()      

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.flatten(x))
        x = self.fc1(x)
        mean, logvar = torch.split(x, self.latent_dim, dim=1)
        return mean, logvar


class Conv6Encoder(nn.Module):

    def __init__(self, latent_dim=256):
        super().__init__()
        hidden_dim = max(512, 2*latent_dim)
        # encode
        # input : 128 x 128
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1) # 64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1) # 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 16
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 8
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 4
        self.conv6 = nn.Conv2d(256, hidden_dim, kernel_size=4, stride=1) # 1
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_dim, 2*latent_dim)
        self.fc_mean = nn.Linear(2*latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(2*latent_dim, latent_dim)
        self.relu = nn.ReLU()      

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.flatten(x)
        x = self.fc1(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

"""
class SeqEncoder(nn.Module):

    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()
        hidden_dim = max(512, 2*latent_dim)
        # encoder
        # input : 256 x 256
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1), # 128
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 64
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 4
            nn.ReLU(True),
            nn.Conv2d(512, hidden_dim, kernel_size=4, stride=1), # 1
            nn.Flatten(),
            nn.Linear(hidden_dim, 2*latent_dim)
        )
        self.fc_mean = nn.Linear(2*latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(2*latent_dim, latent_dim)

    def forward(self,x):
        x = self.encoder(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
"""