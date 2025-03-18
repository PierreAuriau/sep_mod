# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from vae import VAE
from datasets import PhotoSketchingDataset
from loggers import TrainLogger, setup_logging


def gs_beta():
    modality = "photo"
    
    path2models = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/photo_sketching"
    if modality == "photo":
        chkpt_dir = os.path.join(path2models, "20240721_vae-photo")
    elif modality == "sketch":
        chkpt_dir = os.path.join(path2models, "20240720_vae-sketch")
    os.makedirs(chkpt_dir, exist_ok=True)
    nb_epochs = 500

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))

    # data loaders
    train_dataset = PhotoSketchingDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=4)
    
    val_dataset = PhotoSketchingDataset(split="val")
    val_loader = DataLoader(val_dataset, batch_size=64,
                            shuffle=False, num_workers=4)
    
    test_dataset = PhotoSketchingDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, num_workers=4)
    
    # model
    input_channels = 3 if modality=="photo" else 1
    latent_dim = "" # FIXME
    nb_layers = "" # FIXME

    for beta in [10, 100, 1000]: # FIXME
        model = VAE(input_channels=input_channels,
                    latent_dim=latent_dim,
                    nb_layers=nb_layers,
                    beta=beta)
        print(f"Beta: {beta}")
        chkpt_dir_gs = os.path.join(chkpt_dir, f"beta_{beta}")
        os.makedirs(chkpt_dir_gs, exist_ok=True)
        model.fit(train_loader, val_loader, chkpt_dir=chkpt_dir_gs, 
                  nb_epochs=nb_epochs, modality=modality, lr=1e-4)
        model.load_chkpt(os.path.join(chkpt_dir_gs, f"vae_mod-{modality}_ep-{nb_epochs-1}.pth"))
        reconstructions = model.get_reconstructions(val_loader, modality)
        filename = f"reconstructions_set-validation_ep-{nb_epochs-1}.npy"
        np.save(os.path.join(chkpt_dir_gs, filename), reconstructions)


if __name__ == "__main__":

    modality = "sketch"
    path2models = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/photo_sketching"
    chkpt_dir = os.path.join(path2models, "20240721_vae-photo")
    os.makedirs(chkpt_dir, exist_ok=True)
    nb_epochs = 500

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))

    # data loaders
    train_dataset = PhotoSketchingDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=4)
    
    val_dataset = PhotoSketchingDataset(split="val")
    val_loader = DataLoader(val_dataset, batch_size=64,
                            shuffle=False, num_workers=4)
    
    test_dataset = PhotoSketchingDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, num_workers=4)
    
    
    # model
    input_channels = 3 if modality=="photo" else 1

    model = VAE(input_channels=input_channels,
                        latent_dim=256,
                        nb_layers=6)
            
    chkpt_dir_gs = os.path.join(path2models, "20250228_vae", modality)
    os.makedirs(chkpt_dir_gs, exist_ok=True)
    model.fit(train_loader, val_loader, chkpt_dir=chkpt_dir_gs, 
              nb_epochs=nb_epochs, modality=modality, lr=1e-4)
    for epoch in [nb_epochs-1, "best"]:
        model.load_chkpt(os.path.join(chkpt_dir_gs, f"vae_mod-{modality}_ep-{epoch}.pth"))
        reconstructions = model.get_reconstructions(val_loader, modality)
        filename = f"reconstructions_set-validation_ep-{epoch}.npy"
        np.save(os.path.join(chkpt_dir_gs, filename), reconstructions)   
    """ GridSearch
    for nb_layers in [7, 4]:
        for latent_dim in [64, 128, 256]:
            model = VAE(input_channels=input_channels,
                        latent_dim=latent_dim,
                        nb_layers=nb_layers)
            print(f"latent dimension : {latent_dim} - nb layers: {nb_layers}")
            chkpt_dir_gs = os.path.join(chkpt_dir, f"lt-{latent_dim}_ly-{nb_layers}")
            os.makedirs(chkpt_dir_gs, exist_ok=True)
            #model.fit(train_loader, val_loader, chkpt_dir=chkpt_dir_gs, 
            #          nb_epochs=nb_epochs, modality=modality, lr=1e-4)
            for epoch in [nb_epochs-1, "best"]:
                model.load_chkpt(os.path.join(chkpt_dir_gs, f"vae_mod-{modality}_ep-{epoch}.pth"))
                reconstructions = model.get_reconstructions(val_loader, modality)
                filename = f"reconstructions_set-validation_ep-{epoch}.npy"
                np.save(os.path.join(chkpt_dir_gs, filename), reconstructions)
            #model.test(test_loader, epoch=nb_epochs-1, 
            #           modality=modality, chkpt_dir=chkpt_dir_gs)
    
    #model.load_chkpt(os.path.join(chkpt_dir, "vae_mod-photo_ep-299.pth"))
    #outputs = model.get_reconstructions(test_loader)
    #print("outputs", outputs.shape, outputs.dtype, outputs.min(), outputs.max())
    #import re
    #epochs = [re.search("ep-([0-9]+).pth", f) for f in os.listdir(chkpt_dir)]
    #epochs = list(filter(None, epochs))
    #epochs = [m.group(1) for m in epochs]
    #for epoch in epochs:
    #    model.test(test_loader, epoch=epoch, 
    #            modality=modality, chkpt_dir=chkpt_dir)
    """