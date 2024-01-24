# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import function
import torch.nn as nn
from torchvision.models import resnet18


class WeakEncoder(nn.Module):
    def __init__(self, weak_dim):
        super(WeakEncoder, self).__init__()

        self.weak_dim = weak_dim

        # encoder
        self.weak_enc = resnet18(pretrained=False)
        self.feature_dim = 512

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.weak_enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.weak_enc.maxpool = nn.Identity()
        self.weak_enc.fc = nn.Linear(self.feature_dim, weak_dim)

        # Add MLP projection.
        self.weak_projector = nn.Sequential(nn.Linear(weak_dim, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(),
                                            nn.Linear(128, weak_dim))

    def forward(self, x):
        weak_rep = self.weak_enc(x)

        weak_head = self.weak_projector(weak_rep)

        return weak_rep, weak_head


class StrongEncoder(nn.Module):
    def __init__(self, common_dim, strong_dim):
        super(StrongEncoder, self).__init__()

        self.common_dim = common_dim
        self.strong_dim = strong_dim

        # encoder
        self.common_enc = resnet18(pretrained=False)
        self.strong_enc = resnet18(pretrained=False)
        self.feature_dim = 512

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.common_enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.common_enc.maxpool = nn.Identity()
        self.common_enc.fc = nn.Linear(self.feature_dim, common_dim)

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.strong_enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.strong_enc.maxpool = nn.Identity()
        self.strong_enc.fc = nn.Linear(self.feature_dim, strong_dim)

        # Add MLP projection.
        self.common_projector = nn.Sequential(nn.Linear(common_dim, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(),
                                            nn.Linear(128, common_dim))

        self.strong_projector = nn.Sequential(nn.Linear(strong_dim, 128),
                                              nn.BatchNorm1d(128),
                                              nn.ReLU(),
                                              nn.Linear(128, strong_dim))

    def forward(self, x):
        common_rep = self.common_enc(x)
        strong_rep = self.strong_enc(x)

        common_head = self.common_projector(common_rep)
        strong_head = self.strong_projector(strong_rep)

        return common_rep, common_head, strong_rep, strong_head
