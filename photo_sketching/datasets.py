# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class PhotoSketchingDataset(Dataset):
    def __init__(self, split):
        
        data_dir = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/photo_sketching"
        self.photos = np.load(os.path.join(data_dir, f"{split}_photos.npy"), mmap_mode="r")
        self.sketches = np.load(os.path.join(data_dir, f"{split}_sketches.npy"), mmap_mode="r")
        #print(self.photos.shape)
        #print(self.sketches.shape)
        self.pht_transforms = transforms.Compose([lambda arr: transforms.ToTensor()(arr.astype(np.uint8)),
                                                  transforms.ConvertImageDtype(torch.float32),
                                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                                                  #transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))])
                                                  #transforms.Normalize(mean=(28.14, 29.84, 24.31), std=(63.43, 65.29, 59.70))])
        
        self.skh_transforms = transforms.Compose([lambda arr: transforms.ToTensor()(arr.astype(np.uint8)),
                                                  transforms.ConvertImageDtype(torch.float32),
                                                  transforms.Normalize(mean=0.5, std=0.5)])
                                                  #transforms.Normalize(mean=46.97, std=98.05)])
        
    def __getitem__(self, idx):
        pht = self.photos[idx]
        skh = self.sketches[idx]
        #print(pht.max(), pht.min())
        pht = self.pht_transforms(pht)
        skh = self.skh_transforms(skh)
        return {"photo": pht, "sketch": skh}
    
    def __len__(self):
        return len(self.photos)
    

class PhotoSketchingDataset3C(PhotoSketchingDataset):

    def __init__(self, split):
        super().__init__(split)
        self.skh_transforms = transforms.Compose([lambda arr: np.repeat(arr[:, :, np.newaxis], repeats=3, axis=2),
                                                  lambda arr: transforms.ToTensor()(arr.astype(np.uint8)),
                                                  transforms.ConvertImageDtype(torch.float32),
                                                  transforms.Normalize(mean=0.5, std=0.5)])


if __name__ == "__main__":
    dataset = PhotoSketchingDataset3C(split="test")    
    sample = dataset[0]
    photo = sample["photo"].detach().cpu().numpy()
    photo = ((photo + 1) / 2)*255
    photo = np.moveaxis(photo, 0, -1)
    print(f"Photo shape: {photo.shape}")

    sketch = sample["sketch"].detach().cpu().numpy()
    sketch = ((sketch + 1) / 2)*255
    sketch = np.moveaxis(sketch, 0, -1)
    print(f"Sketch shape: {sketch.shape}")

    ph = dataset.photos[0]
    sk = dataset.sketches[0]


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1)
    ax[0].imshow(ph)
    ax[1].imshow(sk)
    ax[2].imshow(photo)
    ax[3].imshow(sketch)

    plt.show()
