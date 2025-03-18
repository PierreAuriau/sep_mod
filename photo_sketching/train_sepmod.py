# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from datasets import PhotoSketchingDataset3C

from loggers import setup_logging
from sepmod import SepMod



def test_alignement():
    
    path2models = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/photo_sketching"
    chkpt_dir = os.path.join(path2models, "20250306_sepmod_test_alignement_mmd")
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    nb_epochs = 300 
    latent_dim = 256

    for beta in [10e6, 10e8, 10e10]:
        chkpt_dir_beta = os.path.join(chkpt_dir, f"beta-{beta:g}")
        os.makedirs(chkpt_dir_beta, exist_ok=True)
        # train(chkpt_dir_beta, latent_dim, nb_epochs, tc_loss_between_shared_and_specific=False,
        #      loss_between_shared="mmd", betas={"alignement": beta})
        # predict_modality_from_z_shared(chkpt_dir_beta, epoch=(nb_epochs-1))
        test(chkpt_dir_beta, epoch=nb_epochs-1)

def test_tc_loss():
    path2models = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/photo_sketching"
    chkpt_dir = os.path.join(path2models, "20240725_sepmod_test_tc_loss")
    os.makedirs(chkpt_dir, exist_ok=True)
    nb_epochs = 300 
    latent_dim = 128 # FIXME : 256

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    print(f"Checkpoint directory: {chkpt_dir}")
    
    # data loaders
    train_dataset = PhotoSketchingDataset3C(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=4)
    
    val_dataset = PhotoSketchingDataset3C(split="val")
    val_loader = DataLoader(val_dataset, batch_size=128,
                            shuffle=False, num_workers=4)
                    
    test_dataset = PhotoSketchingDataset3C(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4)
    
    # Train model
    model = SepMod(input_channels=3,
                   latent_dim=latent_dim,
                   tc_loss_between_shared_and_specific=True,
                   kl_loss_between_shared=False,
                   tc_loss_between_specifics=False)
    for beta in [10, 100, 1000]:
        print(f"Beta: {beta}")
        model.betas["total_correlation"] = beta
        chkpt_dir_beta = os.path.join(chkpt_dir, f"beta_{beta}")
        os.makedirs(chkpt_dir_beta, exist_ok=True)
        model.fit(train_loader, val_loader, chkpt_dir=chkpt_dir_beta, 
                    nb_epochs=nb_epochs)
        # Reconstructions of validation set
        print("Reconstructions of validation dataset")
        model.load_chkpt(os.path.join(chkpt_dir_beta, f"sepmod_ep-{nb_epochs-1}.pth"))
        model = model.to(model.device)
        val_reconstructions = model.get_reconstructions(val_loader)
        for mod in ("photo", "sketch"):
            np.save(os.path.join(chkpt_dir_beta, 
                                 f"reconstructions_mod-{mod}_set-validation_ep-299.npy"), val_reconstructions[mod])
    
    #model.test(test_loader, epoch=nb_epochs-1, chkpt_dir=chkpt_dir)

def train(chkpt_dir, latent_dim, nb_epochs, tc_loss_between_shared_and_specific=True,
          loss_between_shared="mmd", betas={}):
    
    os.makedirs(chkpt_dir, exist_ok=True)
    print(f"Checkpoint directory: {chkpt_dir}")
    
    # data loaders
    train_dataset = PhotoSketchingDataset3C(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=8)
    
    val_dataset = PhotoSketchingDataset3C(split="val")
    val_loader = DataLoader(val_dataset, batch_size=128,
                            shuffle=False, num_workers=8)
        
    # Train model
    print("Training model")
    model = SepMod(input_channels=3,
                   latent_dim=latent_dim,
                   tc_loss_between_shared_and_specific=tc_loss_between_shared_and_specific,
                   loss_between_shared=loss_between_shared,
                   tc_loss_between_specifics=False)
    for k, v in betas.items():
        model.betas[k] = v
    model.fit(train_loader, val_loader, chkpt_dir=chkpt_dir, 
                nb_epochs=nb_epochs)
    
def test(chkpt_dir, epoch):
    # data loading   
    test_dataset = PhotoSketchingDataset3C(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=8)
    
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as f:
        hyperparameters = json.load(f)
    latent_dim = hyperparameters["latent_dim"]
    model = SepMod(input_channels=3,
                latent_dim=latent_dim,
                tc_loss_between_shared_and_specific=False,
                loss_between_shared=None,
                tc_loss_between_specifics=False) 
    # Reconstructions
    model.load_chkpt(os.path.join(chkpt_dir, f"sepmod_ep-{epoch}.pth"))
    print("Reconstructions of test set")
    test_reconstructions = model.get_reconstructions(test_loader)
    for mod in ("photo", "sketch"):
        np.save(os.path.join(chkpt_dir, 
                                f"reconstructions_mod-{mod}_set-test_ep-{epoch}.npy"), test_reconstructions[mod])
    #model.test(test_loader, epoch=epoch, chkpt_dir=chkpt_dir)

def predict_modality_from_z_shared(chkpt_dir, epoch):
        
    print("Prediction of modality from z_shared")
    print(f"Chkpt directory: {chkpt_dir}")
    
    # data loaders
    print("Data Loading")
    train_dataset = PhotoSketchingDataset3C(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=8)
    
    val_dataset = PhotoSketchingDataset3C(split="val")
    val_loader = DataLoader(val_dataset, batch_size=128,
                            shuffle=False, num_workers=8)
                    
    test_dataset = PhotoSketchingDataset3C(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=8)

    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as f:
        hyperparameters = json.load(f)
    latent_dim = hyperparameters["latent_dim"]
    model = SepMod(input_channels=3,
                   latent_dim=latent_dim)
    
    model.load_chkpt(os.path.join(chkpt_dir, f"sepmod_ep-{epoch}.pth"))
    model = model.to(model.device)
    # Get input data
    z_shared = {s: [] for s in ["train", "validation", "test"]}
    y =  {s: [] for s in ["train", "validation", "test"]}
    model.eval()
    with torch.no_grad():
        for split, loader in zip(["train", "validation", "test"], 
                                [train_loader, val_loader, test_loader]):
            for inputs in tqdm(loader, desc=split):
                    for mod in ("photo", "sketch"):
                        inputs[mod] = inputs[mod].to(model.device)
                    z = model.get_embeddings(inputs)
                    for mod, repr in z.items():
                        z_shared[split].extend(repr[:, model.shared_slice].cpu().numpy())
                        y[split].extend([mod for _ in range(len(repr))])
            z_shared[split] = np.asarray(z_shared[split])
            y[split] = np.asarray(y[split])

    # Train Logistic Regression
    print("Train logistic regression")
    logs = defaultdict(list)
    log_reg = LogisticRegression(penalty="l2", C=1.0, fit_intercept=True, max_iter=1000)
    log_reg.fit(z_shared["train"], y["train"])
    for split in ("train", "validation", "test"):
        print(f"Prediction score on {split} set:", log_reg.score(z_shared[split], y[split]))
        y_score = log_reg.predict_proba(z_shared[split])
        roc_auc = roc_auc_score(y_score=y_score[:, 1], y_true=y[split])
        logs["split"].append(split)
        logs["roc_auc"].append(roc_auc)
    logs = pd.DataFrame(logs)
    logs.to_csv(os.path.join(chkpt_dir, f"predictions_of_modality_from_z_shared_ep-{epoch}.csv"), index=False)

def get_mixed_reconstructions(chkpt_dir, epoch):
    
    print("Get mixed reconstructions")
    print(f"Checkpoint directory: {chkpt_dir}")

    # Data Loading
    test_dataset = PhotoSketchingDataset3C(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=8)

    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as f:
        hyperparameters = json.load(f)
    latent_dim = hyperparameters["latent_dim"]
    model = SepMod(input_channels=3,
                   latent_dim=latent_dim)
    
    model.load_chkpt(os.path.join(chkpt_dir, f"sepmod_ep-{epoch}.pth"))
    model = model.to(model.device)

    outputs_shared_only = {mod: [] for mod in ("photo", "sketch")}
    outputs_spe_only = {mod: [] for mod in ("photo", "sketch")}
    outputs_crossed = {mod: [] for mod in ("photo", "sketch")}
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="test"):
                for mod in ("photo", "sketch"):
                    inputs[mod] = inputs[mod].to(model.device)
                z = model.get_embeddings(inputs)
                z_crossed = {}
                for mod, repr in z.items():
                    z_shared_only = torch.zeros_like(repr)
                    z_shared_only[model.shared_slice] = repr[model.shared_slice]
                    z_spe_only = torch.zeros_like(repr)
                    z_spe_only[model.specific_slice] = repr[model.specific_slice]
                    z_crossed = torch.zeros_like(repr)
                    z_crossed[model.specific_slice] = repr[model.specific_slice]
                    cross_mod = {"photo": "sketch", "sketch": "photo"}[mod]
                    z_crossed[model.shared_slice] = z[cross_mod][model.shared_slice]
                    outputs_shared_only[mod].extend(model.decoders[mod](z_shared_only).cpu().numpy())
                    outputs_spe_only[mod].extend(model.decoders[mod](z_spe_only).cpu().numpy())
                    outputs_crossed[mod].extend(model.decoders[mod](z_crossed).cpu().numpy())
    for outputs in [outputs_shared_only, outputs_spe_only, outputs_crossed]:
        for mod in ("photo", "sketch"):
            outputs[mod] = np.asarray(outputs[mod])
            # normalize outputs
            # outputs[mod] = (outputs[mod] - outputs[mod].min(axis=0)) / (outputs[mod].max(axis=0) - outputs[mod].min(axis=0)) * 255
            outputs[mod] = (outputs[mod] + 1) * (255 / 2)
            assert np.all(outputs[mod] >= 0) & np.all(outputs[mod] <= 255), \
                f"Wrong values for reconstructions ({outputs[mod].min(), outputs[mod].max()})"
            outputs[mod] = outputs[mod].astype(int)
            # put channels in last
            outputs[mod] = np.moveaxis(outputs[mod], 1, -1)
    for mod in ("photo", "sketch"):
        np.save(os.path.join(chkpt_dir, 
                                f"reconstructions_shared_only_mod-{mod}_set-test_ep-{epoch}.npy"), outputs_shared_only[mod])
        np.save(os.path.join(chkpt_dir, 
                                f"reconstructions_spe_only_mod-{mod}_set-test_ep-{epoch}.npy"), outputs_spe_only[mod])
        np.save(os.path.join(chkpt_dir, 
                                f"reconstructions_crossed_mod-{mod}_set-test_ep-{epoch}.npy"), outputs_crossed[mod])


if __name__ == "__main__":
    chkpt_dir = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/photo_sketching/20250312_sepmod_mmd"
    os.makedirs(chkpt_dir, exist_ok=True)
    #train(chkpt_dir, latent_dim=256, nb_epochs=1000, tc_loss_between_shared_and_specific=True,
    #      loss_between_shared="mmd", betas={"alignement": 10e8, "total_correlation": 10e4})
    for epoch in (400, 500, 999, 200, 600, 800):
        test(chkpt_dir, epoch=epoch)
        get_mixed_reconstructions(chkpt_dir, epoch=epoch)
        predict_modality_from_z_shared(chkpt_dir, epoch=epoch)