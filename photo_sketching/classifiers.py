# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), 
                                nn.ReLU(), 
                                nn.Linear(self.latent_dim, self.latent_dim), 
                                nn.ReLU(), 
                                nn.Linear(self.latent_dim, 1))

    def forward(self, latent):
        latent = latent.view(-1, self.latent_dim)
        h = self.fc(latent)
        pred = torch.sigmoid(h)
        return pred