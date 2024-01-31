# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import torch.nn as nn
from resnet import resnet18

class Encoder(nn.Module):
    def __init__(self, backbone, n_embedding):
        super(Encoder, self).__init__()
        self.latent_dim = n_embedding
        # encoder
        if backbone == "resnet18":
            self.encoder = resnet18(n_embedding=n_embedding, in_channels=1)
        # Add MLP projection.
        self.projector = nn.Sequential(nn.Linear(n_embedding, 128),
                                       nn.BatchNorm1d(128),
                                       nn.ReLU(),
                                       nn.Linear(128, n_embedding))

    def forward(self, x):
        representation = self.encoder(x)
        head = self.projector(representation)
        return representation, head
