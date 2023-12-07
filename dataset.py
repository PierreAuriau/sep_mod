# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import function
import numpy as np
import torch
import torch.nn as nn
from kornia.augmentation import ColorJitter, RandomGrayscale, RandomResizedCrop, RandomHorizontalFlip
import tensorflow.compat.v2 as tf
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
# Dataset de CIFAR
from keras.datasets import cifar10


class CifarMNISTDataset(Dataset):
    def __init__(self, weak_data, strong_data, weak_targets, strong_targets):
        self.weak_data = torch.from_numpy(np.transpose(weak_data, (0, 3, 1, 2)).astype(float)).float()
        self.strong_data = torch.from_numpy(np.transpose(strong_data, (0, 3, 1, 2)).astype(float)).float()
        self.weak_targets = torch.from_numpy(weak_targets.astype(float)).float()
        self.strong_targets = torch.from_numpy(strong_targets.astype(float)).float()

    def __len__(self):
        return len(self.weak_targets)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        weak_img, strong_img, weak_label, strong_label = (self.weak_data[index], self.strong_data[index],
                                                        int(self.weak_targets[index]), int(self.strong_targets[index]))

        return weak_img, strong_img, weak_label, strong_label


def get_dataloaders(weak_modality, n_train=50000, n_test=1000, batch_size=512):
    # build datasets
    (X_train_cifar, y_train_cif), (X_test_cifar, y_test_cif) = cifar10.load_data()
    X_train_cifar, y_train_cif = X_train_cifar[:n_train], y_train_cif[:n_train]
    X_test_cifar, y_test_cif = X_test_cifar[:n_test], y_test_cif[:n_test]

    mnist = tf.keras.datasets.mnist
    (X_train_mnist, y_train_mni), (X_test_mnist, y_test_mni) = mnist.load_data()
    X_train_mnist, y_train_mni = X_train_mnist[:n_train], y_train_mni[:n_train]
    X_train_mnist = np.array([np.pad(img, (2, 2)) for img in X_train_mnist])
    X_test_mnist, y_test_mni = X_test_mnist[:n_test], y_test_mni[:n_test]
    X_test_mnist = np.array([np.pad(img, (2, 2)) for img in X_test_mnist])

    # build train dataset
    X_train_weak = []
    X_train_strong = []
    y_train_weak = []
    y_train_strong = []
    for i in range(n_train):
        if weak_modality == "cifar":
            X_train_weak.append(0.5 * X_train_cifar[i] / 255.)
        elif weak_modality == "mnist":
            X_train_weak.append(0.5 * np.repeat(X_train_mnist[i][:, :, None], 3, axis=2) / 255.)
        X_train_strong.append((0.5*X_train_cifar[i] + 0.5*np.repeat(X_train_mnist[i][:, :, None], 3, axis=2)) / 255.)
        y_train_weak.append(y_train_cif[i][0])
        y_train_strong.append(y_train_mni[i])
    X_train_weak = np.array(X_train_weak)
    X_train_strong = np.array(X_train_strong)
    y_train_strong = np.array(y_train_strong)
    y_train_weak = np.array(y_train_weak)

    # build test dataset
    X_test_weak = []
    X_test_strong = []
    y_test_weak = []
    y_test_strong = []
    for i in range(n_test):
        if weak_modality == "cifar":
            X_test_weak.append(0.5 * X_test_cifar[i] / 255.)
        elif weak_modality == "mnist":
            X_test_weak.append(0.5 * np.repeat(X_test_mnist[i][:, :, None], 3, axis=2) / 255.)
        X_test_strong.append((0.5 * X_test_cifar[i] + 0.5 * np.repeat(X_test_mnist[i][:, :, None], 3, axis=2)) / 255.)
        y_test_weak.append(y_test_cif[i][0])
        y_test_strong.append(y_test_mni[i])
    X_test_weak = np.array(X_test_weak)
    X_test_strong = np.array(X_test_strong)
    y_test_strong = np.array(y_test_strong)
    y_test_weak = np.array(y_test_weak)

    # Instantiate Dataset and Data Loader
    train_dataset = CifarMNISTDataset(X_train_weak, X_train_strong, y_train_weak, y_train_strong)
    test_dataset = CifarMNISTDataset(X_test_weak, X_test_strong, y_test_weak, y_test_strong)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, test_loader


class SimCLRDataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        # color distortion function
        s = 0.5
        jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        random_jitter = transforms.RandomApply([jitter], p=0.8)
        random_greyscale = RandomGrayscale(p=0.2)
        color_distort = nn.Sequential(random_jitter, random_greyscale)

        self.transforms = nn.Sequential(
            RandomResizedCrop((24, 24), scale=(0.2, 1)),
            RandomHorizontalFlip(p=0.5),
            color_distort
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)
        return x_out

