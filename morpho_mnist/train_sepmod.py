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
import itertools
from collections import defaultdict

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score, \
                            mean_absolute_error, mean_squared_error

import torch
from torch.utils.data import DataLoader

from datasets import MorphoMNISTDataset
from loggers import setup_logging
from sepmod import SepMod
from config import Config

config = Config()


def test_alignement():
    chkpt_dir = os.path.join(config.path2models, "20240723_sepmod_test_alignement")
    os.makedirs(chkpt_dir, exist_ok=True)
    nb_epochs = 300 
    latent_dim = 128 # FIXME : 256

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    print(f"Checkpoint directory: {chkpt_dir}")
    # data loaders
    train_dataset = MorphoMNISTDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=4)
    
    val_dataset = MorphoMNISTDataset(split="val")
    val_loader = DataLoader(val_dataset, batch_size=128,
                            shuffle=False, num_workers=4)
                    
    test_dataset = MorphoMNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4)
    
    # model
    model = SepMod(input_channels=3,
                    latent_dim=latent_dim,
                    tc_loss_between_shared_and_specific=False,
                    kl_loss_between_shared=False,
                    tc_loss_between_specifics=False)
    
    model.fit(train_loader, val_loader, chkpt_dir=chkpt_dir, 
                nb_epochs=nb_epochs)
    
    model.test(test_loader, epoch=nb_epochs-1, chkpt_dir=chkpt_dir)
    
    # Reconstructions of validation set
    print("Reconstructions of validation dataset")
    model.load_chkpt(os.path.join(chkpt_dir, "sepmod_ep-299.pth"))
    model = model.to(model.device)
    val_reconstructions = model.get_reconstructions(val_loader)
    for mod in ("image", "skeleton"):
        np.save(os.path.join(chkpt_dir, 
                                f"reconstructions_mod-{mod}_set-validation_ep-299.npy"), val_reconstructions[mod])

           
    # Test linear probe
    print("Prediction of z_shared modality")
    model.load_chkpt(os.path.join(chkpt_dir, "sepmod_ep-299.pth"))
    model = model.to(model.device)
    # Get input data
    z_shared = {s: [] for s in ["train", "validation", "test"]}
    y =  {s: [] for s in ["train", "validation", "test"]}
    model.eval()
    with torch.no_grad():
        for split, loader in zip(["train", "validation", "test"], 
                                [train_loader, val_loader, test_loader]):
            for inputs in tqdm(loader, desc=split):
                    for mod in ("image", "skeleton"):
                        inputs[mod] = inputs[mod].to(model.device)
                    z = model.get_embeddings(inputs)
                    for mod, repr in z.items():
                        z_shared[split].extend(repr[:, (latent_dim//2):].cpu().numpy())
                        y[split].extend([mod for _ in range(len(repr))])
            z_shared[split] = np.asarray(z_shared[split])
            y[split] = np.asarray(y[split])

    # Train Logistic Regression
    logs = defaultdict(list)
    log_reg = LogisticRegression(penalty="l2", C=1.0, fit_intercept=True, max_iter=1000)
    # log_reg_gs = GridSearchCV(LogisticRegression(),
    #                         {"C": 10. ** np.arange(-3, 3),
    #                         "penalty": ["l2", "l1", "elasticnet"]},
    #                         cv={"train": [i for i in range(len(y["train"]))],
    #                             "test": [i for i in range(len(y["train"]), len(y["train"])+len(y["validation"]))]} )
    
    log_reg.fit(z_shared["train"], y["train"])
    for split in ("train", "validation", "test"):
        print(f"Prediction score on {split} set:", log_reg.score(z_shared[split], y[split]))
        y_score = log_reg.predict_proba(z_shared[split])
        roc_auc = roc_auc_score(y_score=y_score[:, 1], y_true=y[split])
        logs["split"].append(split)
        logs["roc_auc"].append(roc_auc)
    logs = pd.DataFrame(logs)
    logs.to_csv(os.path.join(chkpt_dir, "predictions_z_shared_modality.csv"), index=False)
    
    for beta in [0, 1.0, 10.0, 100.0]:
        print("Beta alignement:", beta)
        chkpt_dir = os.path.join(config.path2models, "20240723_sepmod_test_alignement", f"beta_{int(beta)}")
        os.makedirs(chkpt_dir, exist_ok=True)
        # model
        model = SepMod(input_channels=3,
                       latent_dim=latent_dim,
                       tc_loss_between_shared_and_specific=False,
                       kl_loss_between_shared=True,
                       tc_loss_between_specifics=False)
        
        model.betas["alignement"] = beta
        model.fit(train_loader, val_loader, chkpt_dir=chkpt_dir, 
                  nb_epochs=nb_epochs)
        
        model.test(test_loader, epoch=nb_epochs-1, chkpt_dir=chkpt_dir)

        # Reconstructions of validation set
        print("Reconstructions of validation dataset")
        model.load_chkpt(os.path.join(chkpt_dir, "sepmod_ep-299.pth"))
        model = model.to(model.device)
        val_reconstructions = model.get_reconstructions(val_loader)
        for mod in ("image", "skeleton"):
            np.save(os.path.join(chkpt_dir, 
                                 f"reconstructions_mod-{mod}_set-validation_ep-299.npy"), val_reconstructions[mod])
            
        # Test linear probe
        print("Prediction of z_shared modality")
        model.load_chkpt(os.path.join(chkpt_dir, "sepmod_ep-299.pth"))
        model = model.to(model.device)
        # Get input data
        z_shared = {s: [] for s in ["train", "validation", "test"]}
        y =  {s: [] for s in ["train", "validation", "test"]}
        model.eval()
        with torch.no_grad():
            for split, loader in zip(["train", "validation", "test"], 
                                    [train_loader, val_loader, test_loader]):
                for inputs in tqdm(loader, desc=split):
                        for mod in ("image", "skeleton"):
                            inputs[mod] = inputs[mod].to(model.device)
                        z = model.get_embeddings(inputs)
                        for mod, repr in z.items():
                            z_shared[split].extend(repr[:, (latent_dim//2):].cpu().numpy())
                            y[split].extend([mod for _ in range(len(repr))])
                z_shared[split] = np.asarray(z_shared[split])
                y[split] = np.asarray(y[split])

        # Train Logistic Regression
        logs = defaultdict(list)
        log_reg = LogisticRegression(penalty="l2", C=1, fit_intercept=True, max_iter=1000)
        log_reg.fit(z_shared["train"], y["train"])
        for split in ("train", "validation", "test"):
            print(f"Prediction score on {split} set:", log_reg.score(z_shared[split], y[split]))
            y_score = log_reg.predict_proba(z_shared[split])
            roc_auc = roc_auc_score(y_score=y_score[:, 1], y_true=y[split])
            logs["split"].append(split)
            logs["roc_auc"].append(roc_auc)
        logs = pd.DataFrame(logs)
        logs.to_csv(os.path.join(chkpt_dir, "predictions_z_shared_modality.csv"), index=False)
        

def test_tc_loss():
    chkpt_dir = os.path.join(config.path2models, "20240904_sepmod_test_tc_loss")
    os.makedirs(chkpt_dir, exist_ok=True)
    nb_epochs = 500 
    latent_dim = 16

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    print(f"Checkpoint directory: {chkpt_dir}")
    
    # data loaders
    train_dataset = MorphoMNISTDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=4)
                    
    test_dataset = MorphoMNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4)
    
    # Train model
    model = SepMod(latent_dim=latent_dim,
                   tc_loss_between_shared_and_specific=True,
                   kl_loss_between_shared=False,
                   tc_loss_between_specifics=False)
    for beta in [0, 1, 10, 100]:
        print(f"Beta: {beta}")
        if beta == 0:
            model.betas["total_correlation"] = beta
            chkpt_dir_beta = os.path.join(chkpt_dir, f"beta_{beta}")
            os.makedirs(chkpt_dir_beta, exist_ok=True)
            model.fit(train_loader, chkpt_dir=chkpt_dir_beta, 
                        nb_epochs=nb_epochs)
        
        # Reconstructions of test set
        print("Reconstructions of test dataset")
        model.load_chkpt(os.path.join(chkpt_dir_beta, f"sepmod_ep-{nb_epochs-1}.pth"))
        model = model.to(model.device)
        model.eval()
        test_reconstructions = model.get_reconstructions(test_loader)
        for mod in ("skeleton", "image"):
            np.save(os.path.join(chkpt_dir_beta, 
                                 f"reconstructions_mod-{mod}_set-test_ep-{nb_epochs-1}.npy"), test_reconstructions[mod])
                
    #model.test(test_loader, epoch=nb_epochs-1, chkpt_dir=chkpt_dir)

def test_linear_probe():
    chkpt_dir = os.path.join(config.path2models, "20240904_sepmod_test_tc_loss")
    nb_epochs = 500 
    latent_dim = 16

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    print(f"Checkpoint directory: {chkpt_dir}")

    targets = ["fracture_x", "fracture_y", "swelling_amount", "thickness", "area", \
               "length", "slant", "height", "width", "label"]
    
    # data loaders
    train_dataset = MorphoMNISTDataset(split="train", targets=targets)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=4)
                    
    test_dataset = MorphoMNISTDataset(split="test", targets=targets)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4)
    
    # Train model
    model = SepMod(latent_dim=latent_dim,
                   tc_loss_between_shared_and_specific=True,
                   kl_loss_between_shared=False,
                   tc_loss_between_specifics=False)
    for beta in [0, 1, 10, 100]:
        print(f"Beta: {beta}")
        model.betas["total_correlation"] = beta
        chkpt_dir_beta = os.path.join(chkpt_dir, f"beta_{beta}")
        
        model.load_chkpt(os.path.join(chkpt_dir_beta, f"sepmod_ep-{nb_epochs-1}.pth"))
        model = model.to(model.device)
        model.eval()
        
        # Test linear probe
        z = {s: defaultdict(list) for s in ["train", "test"]}
        y = {s: defaultdict(list) for s in ["train", "test"]}
        with torch.no_grad():
            for split, loader in zip(["train", "test"], 
                                    [train_loader, test_loader]):
                for inputs in tqdm(loader, desc=split):
                    for mod in ("skeleton", "image"):
                        inputs[mod] = inputs[mod].to(model.device)
                    emb = model.get_embeddings(inputs)
                    for mod, repr in emb.items():
                        z[split][mod].extend(repr.cpu().numpy())
                    for target in targets:    
                        y[split][target].extend(inputs[target].cpu().numpy())
                for mod in ("skeleton", "image"):
                    z[split][mod] = np.asarray(z[split][mod])
                for target in targets:
                    y[split][target] = np.asarray(y[split][target])
        
        logs = defaultdict(list)
        for mod in ("image", "skeleton"):
            for latent_space, cut in zip(("specific", "shared", "concat"), (model.specific_slice, model.shared_slice, slice(0, latent_dim))): 
                for target in targets:
                    if target == "label":
                        log_reg = LogisticRegression(penalty="l2", C=1.0, fit_intercept=True, max_iter=1000)
                        log_reg.fit(z["train"][mod][:, cut], y["train"][target])
                        for split in ("train", "test"):
                            print(f"Prediction of {target} on {split} set: {log_reg.score(z[split][mod][:, cut], y[split][target]):.2f}")
                            y_pred = log_reg.predict(z[split][mod][:, cut])
                            bacc = balanced_accuracy_score(y_pred=y_pred, y_true=y[split][target])
                            logs["modality"].append(mod)
                            logs["latent_space"].append(latent_space)
                            logs["target"].append(target)
                            logs["split"].append(split)
                            logs["metric"].append("balanced_accuracy")
                            logs["value"].append(bacc)
                    else:
                        ridge = Ridge(alpha=0.1, fit_intercept=True, solver="auto")
                        ridge.fit(z["train"][mod][:, cut], y["train"][target])
                        for split in ("train", "test"):
                            print(f"Prediction of {target} on {split} set: {ridge.score(z[split][mod][:, cut], y[split][target]):.2f}")
                            y_pred = ridge.predict(z[split][mod][:, cut])
                            r2 = ridge.score(z[split][mod][:, cut], y[split][target])
                            rmse = mean_squared_error(y_pred=y_pred, y_true=y[split][target], squared=False) 
                            mae = mean_absolute_error(y_pred=y_pred, y_true=y[split][target])

                            for metric, value in zip(["r2", "root_mean_squarred_error", "mean_absolute_error"], 
                                                    [r2, rmse, mae]):
                                logs["modality"].append(mod)
                                logs["latent_space"].append(latent_space)
                                logs["target"].append(target)
                                logs["split"].append(split)
                                logs["metric"].append(metric)
                                logs["value"].append(value)
            
        logs = pd.DataFrame(logs)
        logs.to_csv(os.path.join(chkpt_dir_beta, "linear_predictions_from_embeddings.csv"), index=False)


def get_mixed_reconstructions():
    chkpt_dir = os.path.join(config.path2models, "20240723_sepmod_test_alignement")
    nb_epochs = 300 
    latent_dim = 128

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    print(f"Checkpoint directory: {chkpt_dir}")
    
    # data loaders
    train_dataset = MorphoMNISTDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=4)
    
    val_dataset = MorphoMNISTDataset(split="val")
    val_loader = DataLoader(val_dataset, batch_size=128,
                            shuffle=False, num_workers=4)
                    
    test_dataset = MorphoMNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4)
    
    # Train model
    model = SepMod(input_channels=3,
                   latent_dim=latent_dim,
                   tc_loss_between_shared_and_specific=False,
                   kl_loss_between_shared=True,
                   tc_loss_between_specifics=False)
    
    for beta in [0, 1]:
        print("Beta alignement:", beta)
        if beta == 1:
            chkpt_dir_beta = os.path.join(chkpt_dir)
        else:
            chkpt_dir_beta = os.path.join(chkpt_dir, f"beta_{beta}")
        # Reconstructions of validation set
        print("Reconstructions of validation dataset")
        model.load_chkpt(os.path.join(chkpt_dir_beta, f"sepmod_ep-{nb_epochs-1}.pth"))
        model = model.to(model.device)

        outputs_shared_only = {mod: [] for mod in ("image", "skeleton")}
        outputs_spe_only = {mod: [] for mod in ("image", "skeleton")}
        outputs_crossed = {mod: [] for mod in ("image", "skeleton")}
        model.eval()
        with torch.no_grad():
            for inputs in tqdm(val_loader, desc="validation"):
                    for mod in ("image", "skeleton"):
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
                        cross_mod = {"image": "skeleton", "skeleton": "image"}[mod]
                        z_crossed[model.shared_slice] = z[cross_mod][model.shared_slice]
                        outputs_shared_only[mod].extend(model.decoders[mod](z_shared_only).cpu().numpy())
                        outputs_spe_only[mod].extend(model.decoders[mod](z_spe_only).cpu().numpy())
                        outputs_crossed[mod].extend(model.decoders[mod](z_crossed).cpu().numpy())
        for outputs in [outputs_shared_only, outputs_spe_only, outputs_crossed]:
            for mod in ("image", "skeleton"):
                outputs[mod] = np.asarray(outputs[mod])
                # normalize outputs
                outputs[mod] = (outputs[mod] - outputs[mod].min(axis=0)) / (outputs[mod].max(axis=0) - outputs[mod].min(axis=0)) * 255
                outputs[mod] = outputs[mod].astype(int)
                # put channels in last
                outputs[mod] = np.moveaxis(outputs[mod], 1, -1)
        for mod in ("image", "skeleton"):
            np.save(os.path.join(chkpt_dir_beta, 
                                 f"reconstructions_shared_only_mod-{mod}_set-validation_ep-299.npy"), outputs_shared_only[mod])
            np.save(os.path.join(chkpt_dir_beta, 
                                 f"reconstructions_spe_only_mod-{mod}_set-validation_ep-299.npy"), outputs_spe_only[mod])
            np.save(os.path.join(chkpt_dir_beta, 
                                 f"reconstructions_crossed_mod-{mod}_set-validation_ep-299.npy"), outputs_crossed[mod])

def train_one_model(chkpt_dir, latent_dim=16, nb_epochs=500, 
                    tc_loss_between_shared_and_specific=True,
                    loss_between_shared="mmd",
                    betas={"total_correlation": 10e4,
                           "alignement": 10e8}):    
    # data loaders
    train_dataset = MorphoMNISTDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=8)
                    
    test_dataset = MorphoMNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=8)
    
    # Train model
    model = SepMod(latent_dim=latent_dim,
                   tc_loss_between_shared_and_specific=tc_loss_between_shared_and_specific,
                   loss_between_shared=loss_between_shared,
                   tc_loss_between_specifics=False)
    
    for k, v in betas.items():
        model.betas[k] = v
    model.fit(train_loader, chkpt_dir=chkpt_dir, 
              nb_epochs=nb_epochs)

def test_one_model(chkpt_dir, epoch, targets=["label"]):    
    # data loaders
    train_dataset = MorphoMNISTDataset(split="train", targets=targets)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=8)
                    
    test_dataset = MorphoMNISTDataset(split="test", targets=targets)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=8)
    
    # Load model
    with open(os.path.join(chkpt_dir, "hyperparameters.json"), "r") as f:
        hyperparameters = json.load(f)
    model = SepMod(latent_dim=hyperparameters["latent_dim"],
                   tc_loss_between_shared_and_specific=False,
                   loss_between_shared="mmd",
                   tc_loss_between_specifics=False)
    model.load_chkpt(os.path.join(chkpt_dir, f"sepmod_ep-{epoch}.pth"))
    model = model.to(model.device)
    model.eval()
    
    # Reconstrucions
    test_reconstructions, reconstruction_error = model.get_reconstructions(test_loader, raw=True, return_loss=True)
    with open(os.path.join(chkpt_dir, f"reconstruction_error_set-test_ep-{epoch}.json"), "w") as file:
        json.dump(reconstruction_error, file)
    for mod in ("skeleton", "image"):
        np.save(os.path.join(chkpt_dir, 
                            f"reconstructions_mod-{mod}_set-test_ep-{epoch}.npy"), test_reconstructions[mod])
    
    # Test linear probe
    z = {s: defaultdict(list) for s in ["train", "test"]}
    y = {s: defaultdict(list) for s in ["train", "test"]}
    with torch.no_grad():
        for split, loader in zip(["train", "test"], 
                                [train_loader, test_loader]):
            for inputs in tqdm(loader, desc=split):
                for mod in ("skeleton", "image"):
                    inputs[mod] = inputs[mod].to(model.device)
                emb = model.get_embeddings(inputs)
                for mod, repr in emb.items():
                    z[split][mod].extend(repr.cpu().numpy())
                for mod in ("skeleton", "image"):
                    inputs.pop(mod, None)
                for target, value in inputs.items():
                    y[split][target].extend(inputs[target].cpu().numpy())

            for mod in ("skeleton", "image"):
                z[split][mod] = np.asarray(z[split][mod])
            for target in targets:
                y[split][target] = np.asarray(y[split][target])
    
    logs = defaultdict(list)
    for mod in ("image", "skeleton"):
        for latent_space, cut in zip(("specific", "shared", "concat"), (model.specific_slice, model.shared_slice, slice(0, hyperparameters["latent_dim"]))): 
            for target in targets:
                if target in ["label", "fracture"]:
                    log_reg = LogisticRegression(penalty="l2", C=1.0, fit_intercept=True, max_iter=1000)
                    log_reg.fit(z["train"][mod][:, cut], y["train"][target])
                    for split in ("train", "test"):
                        print(f"Prediction of {target} on {split} set: {log_reg.score(z[split][mod][:, cut], y[split][target]):.2f}")
                        logs["modality"].append(mod)
                        logs["latent_space"].append(latent_space)
                        logs["target"].append(target)
                        logs["split"].append(split)
                        if target == "fracture":
                            y_score = log_reg.predict_proba(z[split][mod][:, cut])
                            roc_auc = roc_auc_score(y_score=y_score[:, 1], y_true=y[split][target])
                            logs["metric"].append("roc_auc")
                            logs["value"].append(roc_auc)
                        else:
                            y_pred = log_reg.predict(z[split][mod][:, cut])
                            bacc = balanced_accuracy_score(y_pred=y_pred, y_true=y[split][target])
                            logs["metric"].append("balanced_accuracy")
                            logs["value"].append(bacc)
                else:
                    ridge = Ridge(alpha=0.1, fit_intercept=True, solver="auto")
                    ridge.fit(z["train"][mod][:, cut], y["train"][target])
                    for split in ("train", "test"):
                        print(f"Prediction of {target} on {split} set: {ridge.score(z[split][mod][:, cut], y[split][target]):.2f}")
                        y_pred = ridge.predict(z[split][mod][:, cut])
                        r2 = ridge.score(z[split][mod][:, cut], y[split][target])
                        rmse = mean_squared_error(y_pred=y_pred, y_true=y[split][target], squared=False) 
                        mae = mean_absolute_error(y_pred=y_pred, y_true=y[split][target])

                        for metric, value in zip(["r2", "root_mean_squarred_error", "mean_absolute_error"], 
                                                [r2, rmse, mae]):
                            logs["modality"].append(mod)
                            logs["latent_space"].append(latent_space)
                            logs["target"].append(target) # FIXME : how to store the right target ?
                            logs["split"].append(split)
                            logs["metric"].append(metric)
                            logs["value"].append(value)
        
    logs = pd.DataFrame(logs)
    logs.to_csv(os.path.join(chkpt_dir, "linear_predictions_from_embeddings_modality_morphometrics.csv"), index=False)
    

def gs_betas():
    chkpt_dir = os.path.join(config.path2models, "20250317_beta_gridsearch")
    os.makedirs(chkpt_dir, exist_ok=True)
    nb_epochs = 500 
    latent_dim = 32

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    print(f"Checkpoint directory: {chkpt_dir}")
    
    # data loaders
    targets = ["label", "thickening_amount", "fracture", "area", "length", 
            "thickness", "slant", "width", "height"]
    train_dataset = MorphoMNISTDataset(split="train", targets=targets)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=8)
                    
    test_dataset = MorphoMNISTDataset(split="test", targets=targets)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=8)
        
    # grid search
    for beta_alg, beta_tc in itertools.product([10e4, 10e6], [10e4, 10e6]):

        # saving dir
        print(f"\nBetas: alignement: {beta_alg:g} - independance: {beta_tc:g}")
        print("-"*45 + "\n")
        chkpt_dir_beta = os.path.join(chkpt_dir, f"balg-{beta_alg:g}_btc-{beta_tc:g}")
        os.makedirs(chkpt_dir_beta, exist_ok=True)
        
        train_one_model(chkpt_dir_beta, latent_dim=latent_dim,
                        nb_epochs=nb_epochs, tc_loss_between_shared_and_specific=True,
                        loss_between_shared="mmd",
                        betas={"total_correlation": beta_tc,
                               "alignement": beta_alg})
        
        test_one_model(chkpt_dir_beta, epoch=nb_epochs-1, targets=targets)


if __name__ == "__main__":
    
    # TODO: remplacer area / thickness par area_img/ thickness_img ?
    gs_betas()
    """
    chkpt_dir = os.path.join(config.path2models, "20250312_sepmod_mmd")
    os.makedirs(chkpt_dir, exist_ok=True)
    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    print(f"Checkpoint directory: {chkpt_dir}")
    
    train_one_model(chkpt_dir, latent_dim=32, nb_epochs=800)
    targets = ["label", "thickening_amount", "fracture", "area", "length", 
                "thickness", "slant", "width", "height"]
    test_one_model(chkpt_dir, epoch=799, targets=targets)
    """