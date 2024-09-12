# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MorphoMNISTDataset(Dataset):
    def __init__(self, split, targets=None):
        
        data_dir = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/morpho_mnist"
        if split == "test":
            split = "t10k"
        self.skeletons = np.load(os.path.join(data_dir, f"{split}-skeletons.npy"), mmap_mode="r")
        self.images = np.load(os.path.join(data_dir, f"{split}-images.npy"), mmap_mode="r")
        self.labels = pd.read_csv(os.path.join(data_dir, f"{split}-morpho.csv"), sep=",")        
        self.transforms = transforms.Compose([lambda arr: transforms.ToTensor()(arr.astype(np.float32)),
                                              transforms.Normalize(mean=0.5, std=0.5)])
        self.targets = targets
        
    def __getitem__(self, idx):
        skel = self.skeletons[idx]
        img = self.images[idx]
        skel = self.transforms(skel)
        img = self.transforms(img)
        if self.targets is None:
            lab = {}
        else:
            lab = self.labels[self.targets].iloc[idx].to_dict()
        return {"image": img, "skeleton": skel, **lab}
    
    def __len__(self):
        return len(self.labels)
    
if __name__ == "__main__":
    dataset = MorphoMNISTDataset(split="test")
    item = dataset[0]
    print(item)