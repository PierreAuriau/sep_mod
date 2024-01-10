# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=2)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 32, 32)
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape, latent_dim=32, channel=512, kernel_size=4,
                 stride=2, padding=1, bias=False):
        super(Decoder, self).__init__()
        self.channel = channel
        output_shape = np.floor((np.array(input_shape) + 2*padding - kernel_size) / stride).astype(int) + 1
        for _ in range(4):
            output_shape = np.floor((output_shape + 2*padding - kernel_size) / stride).astype(int) + 1
        out_features = int(channel*output_shape.prod())
        self.shape = output_shape
        self.fc = nn.Linear(latent_dim, out_features)
        self.bn1 = nn.BatchNorm2d(channel)
        self.tp_conv2 = nn.ConvTranspose2d(channel, channel // 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(channel // 2)
        self.tp_conv3 = nn.ConvTranspose2d(channel // 2, channel // 4, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn3 = nn.BatchNorm2d(channel // 4)
        self.tp_conv4 = nn.ConvTranspose2d(channel // 4, channel // 8, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn4 = nn.BatchNorm2d(channel // 8)
        self.tp_conv5 = nn.ConvTranspose2d(channel // 8, channel // 16, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn5 = nn.BatchNorm2d(channel // 16)
        self.tp_conv6 = nn.ConvTranspose2d(channel // 16, 3, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, latent):
        h = self.fc(latent)
        h = h.view(-1, self.channel, *self.shape)
        h = F.relu(self.bn1(h))
        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))
        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))
        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))
        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv5(h)
        h = F.relu(self.bn5(h))
        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv6(h)
        # h = torch.sigmoid(h)
        return h
