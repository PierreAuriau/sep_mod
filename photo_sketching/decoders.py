# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, input_channels=3, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.shape = 32
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(self.latent_dim,(self.shape**2) *32)
        self.conv3 = nn.ConvTranspose2d(32,64,kernel_size=2,stride=2)
        self.conv4 = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)
        
        self.conv5 = nn.ConvTranspose2d(32,16,kernel_size=2,stride=2)
        self.conv6 = nn.ConvTranspose2d(16, input_channels, kernel_size=1, stride=1)

    def forward(self, z):
        x = self.relu(self.fc2(z))
        x = torch.reshape(x,(x.shape[0],32,self.shape,self.shape))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x
    
class Conv7Decoder(nn.Module):

    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()

        self.hidden_dim = max(512, 2*latent_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(latent_dim, self.hidden_dim)
        self.conv1 = nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=4, stride=1) # 4
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 8
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 16
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # 32
        self.conv5 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1) # 64
        self.conv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # 128
        self.conv7 = nn.ConvTranspose2d(16, input_channels, kernel_size=4, stride=2, padding=1) # 256

    def forward(self, z):
        x = self.relu(self.fc(z))
        x = torch.reshape(x,(x.shape[0], self.hidden_dim, 1, 1))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.sigmoid(x)
        return x


class AntixKDecoder(nn.Module):

    def __init__(self, input_channels=3, latent_dim=256, 
                 hidden_dims=[512, 256, 128, 64, 32, 16, 8]):
        """
        link to Github: <https://github.com/AntixK/PyTorch-VAE>
        """
        super().__init__()
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=input_channels,
                                      kernel_size=3, padding=1),
                            nn.Tanh())
        
    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


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

    z = torch.randn(64, 256)

    decoder = AntixKDecoder()

    outputs = decoder(z)

    print(outputs.size())