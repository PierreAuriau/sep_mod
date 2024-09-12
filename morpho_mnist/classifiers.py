# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, latent_dim, activation="sigmoid"):
        super(Classifier, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), 
                                nn.ReLU(), 
                                nn.Linear(self.latent_dim, self.latent_dim), 
                                nn.ReLU(), 
                                nn.Linear(self.latent_dim, 1))
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation == nn.Softmax()
        else:
            self.activation = activation


    def forward(self, latent, return_logits=False):
        latent = latent.view(-1, self.latent_dim)
        h = self.fc(latent)
        if return_logits:
            return h
        else:
            pred = self.activation(h)
            return pred