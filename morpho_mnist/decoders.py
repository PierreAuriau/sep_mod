# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.shape = 32
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(self.latent_dim,(self.shape**2) *32)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 16,kernel_size=2, stride=2)
        self.conv5 = nn.ConvTranspose2d(16, 1, kernel_size=1, stride=1)

    def forward(self, z):
        x = self.relu(self.fc2(z))
        x = torch.reshape(x,(x.shape[0], 32, self.shape,self.shape))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Conv6Decoder(nn.Module):

    def __init__(self, latent_dim=256):
        super().__init__()

        self.hidden_dim = max(512, 2*latent_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(latent_dim, self.hidden_dim)
        self.conv1 = nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=4, stride=1) # 4
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 8
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 16
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # 32
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # 64
        self.conv6 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1) # 128

    def forward(self, z):
        x = self.relu(self.fc(z))
        x = torch.reshape(x,(x.shape[0], self.hidden_dim, 1, 1))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.sigmoid(x)
        return x

"""
class SeqDecoder(nn.Module):

    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()

        hidden_dim = max(512, 2*latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Unflatten(0, (-1, hidden_dim, 1, 1)),
            nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=4, stride=1), # 4
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1), # 64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 128
            nn.ConvTranspose2d(16, input_channels, kernel_size=4, stride=2, padding=1), # 256
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.decoder(z)
        return x
"""

if __name__ == "__main__":
    z = torch.ones(64, 16)
    decoder = Conv6Decoder(latent_dim=16)
    output = decoder(z)
    print(output.size())