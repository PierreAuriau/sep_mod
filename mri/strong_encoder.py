# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import pickle
import json

from sklearn.metrics import roc_auc_score, mean_squared_error, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

# project import
from loss import align_loss, uniform_loss, norm, joint_entropy_loss
from datamanager import TwoModalityDataManager
from logs import History, setup_logging, save_hyperparameters
from encoder import Encoder

logger = logging.getLogger("StrongEncoder") 

# UDPATES:
# - improve test with loops OK
# - add backbone and latent size to torch.save OK
# - define optimizers OK
# - history -> improve it ! OK
# - logging OK
# - checckpoint dir, exp name, ponderation --> parameter for train OK
# - test on continuous labels OK
# -------------------------------------------------
# * outputs of models as namedtuple
# - keep weak encoder ?
# - save hyperparameters --> in main ?
# - load weak encoder in  init or train/test ?
# - improve training_step
# - remove get target by target
# - iterate over encoder in test (set dico representations instead of z.. and change returns in get embeddings)

class StrongEncoder(object):
    
    def __init__(self, backbone, latent_dim, weak_encoder_chkpt):
        # set models
        self.specific_encoder = Encoder(backbone=backbone, n_embedding=latent_dim)
        self.common_encoder = Encoder(backbone=backbone, n_embedding=latent_dim)
        self.weak_encoder = Encoder(backbone=backbone, n_embedding=latent_dim)
        checkpoint = torch.load(weak_encoder_chkpt)
        try:
            status = self.weak_encoder.load_state_dict(checkpoint["model"], strict=False)
        except KeyError:
            status =  self.weak_encoder.load_state_dict(checkpoint["weak_encoder"], strict=False)
        logger.info(f"Loading weak encoder : {status}")

        # set attributes
        self.latent_dim = latent_dim
        self.backbone = backbone
        self.weak_encoder_chkpt = weak_encoder_chkpt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device : {self.device}")
    
    def train(self, chkpt_dir, exp_name, dataset, ponderation, nb_epochs, nb_epochs_per_saving=10):
        # loader
        manager = TwoModalityDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                         db=dataset, weak_modality="skeleton", strong_modality="vbm",
                                         labels=None, batch_size=8, two_views=True,
                                         num_workers=8, pin_memory=True)
        loader = manager.get_dataloader(train=True, validation=False)
        nb_batch = len(loader.train)
        # define optimizer and scaler
        optimizer = self.configure_optimizers()
        scaler = GradScaler()
        # prepare attributes
        self.checkpointdir = chkpt_dir
        self.exp_name = exp_name
        self.ponderation = ponderation
        self.history = History(name="Train_StrongEncoder", chkpt_dir=self.checkpointdir)
        
        # train model
        self.weak_encoder = self.weak_encoder.to(self.device)
        self.specific_encoder = self.specific_encoder.to(self.device)
        self.common_encoder = self.common_encoder.to(self.device)
        
        for epoch in range(nb_epochs):
            pbar = tqdm(total=nb_batch, desc=f"Epoch {epoch}")
            self.history.step()
            # train
            self.weak_encoder.eval()
            self.common_encoder.train()
            self.specific_encoder.train()
            train_loss = 0
            common_loss, specific_loss, jem_loss = 0, 0, 0
            for batch in loader.train:
                pbar.update()

                co_loss, spe_loss, j_loss = self.training_step(batch)
                loss = co_loss + spe_loss + self.ponderation*j_loss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                common_loss += co_loss.item()
                specific_loss += spe_loss.item()
                jem_loss += j_loss.item()
            pbar.close()
            # saving
            self.history.log(epoch=epoch, train_loss=train_loss, common_loss=common_loss,
                             specific_loss=specific_loss, jem_loss=jem_loss)
            self.history.reduce(reduce_fx="sum")
            if (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) and epoch > 0:
                self.history.summary()
                self.save_checkpoint(
                    epoch=epoch,
                    n_embedding=self.latent_dim,
                    backbone=self.backbone,
                    weak_encoder_checkpoint=self.weak_encoder_chkpt,
                    optimizer=optimizer.state_dict())
                self.history.save()
        logger.info(f"Training duration: {self.history.get_duration()}")


    def training_step(self, batch):
        strong_inputs = batch.strong        
        weak_inputs = batch.weak
        strong_view_1 = strong_inputs.view_1.to(self.device)
        strong_view_2 = strong_inputs.view_2.to(self.device)
        weak_view_1 = weak_inputs.view_1.to(self.device)
        weak_view_2 = weak_inputs.view_2.to(self.device)
        
        with autocast():
            with torch.no_grad():
                _, weak_head_1 = self.weak_encoder(weak_view_1)
                _, weak_head_2 = self.weak_encoder(weak_view_2)
            _, specific_head_1 = self.specific_encoder(strong_view_1)
            _, specific_head_2 = self.specific_encoder(strong_view_2)
            _, common_head_1 = self.common_encoder(strong_view_1)
            _, common_head_2 = self.common_encoder(strong_view_2)

            # loss
            co_loss, spe_loss, j_loss = self.loss_fn(weak_head_1, weak_head_2, 
                                                     specific_head_1, specific_head_2,
                                                     common_head_1, common_head_2)
        return co_loss, spe_loss, j_loss

    def configure_optimizers(self):
        return Adam(list(self.specific_encoder.parameters()) + list(self.common_encoder.parameters()), lr=1e-4)
        
    def save_checkpoint(self, epoch, **kwargs):
        outfile = os.path.join(self.checkpointdir, self.get_chkpt_name(epoch))
        torch.save({
            "epoch": epoch,
            "weak_encoder": self.weak_encoder.state_dict(),
            "specific_encoder": self.specific_encoder.state_dict(),
            "common_encoder": self.common_encoder.state_dict(),
            **kwargs}, outfile)
        return outfile
    
    def load_from_checkpoint(self, epoch):
        filename = os.path.join(self.checkpointdir, self.get_chkpt_name(epoch))
        checkpoint = torch.load(filename)
        status = self.weak_encoder.load_state_dict(checkpoint["weak_encoder"], strict=False)
        logger.info(f"Loading weak encoder : {status}")
        status = self.specific_encoder.load_state_dict(checkpoint["specific_encoder"], strict=False)
        logger.info(f"Loading specific encoder : {status}")
        status = self.common_encoder.load_state_dict(checkpoint["common_encoder"], strict=False)
        logger.info(f"Loading common encoder : {status}")

    def get_chkpt_name(self, epoch):
        return f"StrongEncoder_exp-{self.exp_name}_ep-{epoch}.pth"
    
    def get_representation_name(self, epoch):
        return f"Representations_StrongEncoder_exp-{self.exp_name}_ep-{epoch}.pkl"
    
    def test(self, chkpt_dir, exp_name, dataset, labels, list_epochs):
        
        # datamanagers
        manager = TwoModalityDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                           db=dataset, strong_modality="vbm", weak_modality="skeleton", 
                                           labels=labels, batch_size=8, two_views=False,
                                           num_workers=8, pin_memory=True)
        self.exp_name = exp_name
        self.checkpointdir = chkpt_dir
        self.history = History(name="Test_StrongEncoder", chkpt_dir=self.checkpointdir)
        
        for epoch in list_epochs:
            logger.info(f"Epoch {epoch}")
            self.load_from_checkpoint(epoch=epoch)
            self.weak_encoder = self.weak_encoder.to(self.device)
            self.specific_encoder = self.specific_encoder.to(self.device)
            self.common_encoder = self.common_encoder.to(self.device)

            self.weak_encoder.eval()
            self.common_encoder.eval()
            self.specific_encoder.eval()

            loader = manager.get_dataloader(train=True,
                                            validation=True,
                                            test=True)
            # get embeddings
            zw = {} # weak representations
            zc =  {} # common representations
            zs = {} # specific representations
            y = {} # labels
            filename = os.path.join(self.checkpointdir, self.get_representation_name(epoch=epoch))
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    representations = pickle.load(f)
                logger.info(f"Loading representations : {filename}")
                for split in ("train", "validation", "test", "test_intra"):
                    zw[split] = representations["weak"][split]
                    zc[split] = representations["common"][split]
                    zs[split] = representations["specific"][split]
                    y[split] = manager.dataset[split].get_target()
            else:
                for split in ("train", "validation", "test", "test_intra"):
                    logger.info(f"{split} set")
                    if split == "test_intra":
                        loader = manager.get_dataloader(test_intra=True)
                        zw[split], zc[split], zs[split] = self.get_embeddings(dataloader=loader.test)
                    else:
                        zw[split], zc[split], zs[split]= self.get_embeddings(dataloader=getattr(loader, split))
                    y[split] = manager.dataset[split].get_target()
                self.save_representations(weak=zw, common=zc, specific=zs, epoch=epoch)

            # train predictors
            for i, label in enumerate(labels):
                splits = ["train", "validation", "test", "test_intra"]
                if label in ("diagnosis", "sex"):
                    clf = LogisticRegression(max_iter=1000)
                    clf.get_predictions = clf.predict_proba
                    metrics = {"roc_auc": lambda y_pred, y_true: roc_auc_score(y_score=y_pred[:, 0], y_true=y_true),
                               "balanced_accuracy": lambda y_pred, y_true : balanced_accuracy_score(y_pred=y_pred.argmax(axis=1),
                                                                                                     y_true=y_true)}
                elif label in ("age", "tiv"):
                    clf = Ridge()
                    clf.get_predictions = clf.predict
                    metrics = {"rmse": lambda y_pred, y_true: mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)}
                elif label in ("site"):
                    splits.remove("test") # new sites on external test set
                    lbl = sorted(manager.dataset[split].all_labels[label].unique())
                    map_lbl = np.vectorize(lambda pred: {j: l for j,l in enumerate(lbl)}[pred])
                    clf = LogisticRegression(max_iter=1000)
                    clf.get_predictions = clf.predict_proba
                    metrics = {"roc_auc": lambda y_pred, y_true: roc_auc_score(y_score=y_pred, y_true=y_true,
                                                                               multi_class="ovr", average="macro",
                                                                               labels=lbl),
                               "balanced_accuracy": lambda y_pred, y_true: balanced_accuracy_score(y_pred=map_lbl(y_pred.argmax(axis=1)),
                                                                                                    y_true=y_true)}
                logger.info(f"Test weak encoder on {label}")
                clf = clf.fit(zw["train"], y["train"][:, i])
                print("y_train[i]", np.unique(y["train"][:, i]))
                for split in splits:
                    y_pred = clf.get_predictions(zw[split])
                    print("y_pred shape", y_pred.shape)
                    print("y[i]", y[split][:, i].shape, np.unique(y[split][:, i]))
                    values = {}
                    for name, metric in metrics.items():
                        values[name] = metric(y_pred=y_pred, y_true=y[split][:, i])
                    self.history.step()
                    self.history.log(epoch=epoch, encoder="weak", label=label, set=split, **values)
                
                logger.info(f"Test common encoder on {label}")
                clf = clf.fit(zc["train"], y["train"][:, i])
                for split in splits:
                    y_pred = clf.get_predictions(zc[split])
                    values = {}
                    for name, metric in metrics.items():
                        values[name] = metric(y_pred=y_pred, y_true=y[split][:, i])
                    self.history.step()
                    self.history.log(epoch=epoch, encoder="common", label=label, set=split, **values)

                logger.info(f"Test specific encoder on {label}")
                clf = clf.fit(zs["train"], y["train"][:, i])
                for split in splits:
                    y_pred = clf.get_predictions(zs[split])
                    values = {}
                    for name, metric in metrics.items():
                        values[name] = metric(y_pred=y_pred, y_true=y[split][:, i])
                    self.history.step()
                    self.history.log(epoch=epoch, encoder="specific", label=label, set=split, **values)
            self.history.save()
    
    def get_embeddings(self, dataloader):
        weak_representations, common_representations, specific_representations = [], [], [] 
        pbar = tqdm(total=len(dataloader), desc=f"Get embeddings")
        for dataitem in dataloader:
            pbar.update()
            with torch.no_grad():
                weak_inputs = dataitem.weak
                strong_inputs = dataitem.strong
                weak_repr, _ = self.weak_encoder(weak_inputs.to(self.device))
                common_repr, _ = self.common_encoder(strong_inputs.to(self.device))
                specific_repr, _ = self.specific_encoder(strong_inputs.to(self.device))
            weak_representations.extend(weak_repr.detach().cpu().numpy())
            common_representations.extend(common_repr.detach().cpu().numpy())
            specific_representations.extend(specific_repr.detach().cpu().numpy())
        pbar.close()
        common_representations = np.asarray(common_representations)
        specific_representations = np.asarray(specific_representations)
        weak_representations = np.asarray(weak_representations)
        labels = np.asarray(labels)
        return weak_representations, common_representations, specific_representations
    
    def save_representations(self, weak, common, specific, epoch):
        representations = {
            "weak": weak,
            "common": common,
            "specific": specific}
        with open(os.path.join(self.checkpointdir, self.get_representation_name(epoch=epoch)), "wb") as f:
            pickle.dump(representations, f)

    def loss_fn(self, weak_head_1, weak_head_2, common_head_1, 
                common_head_2, specific_head_1, specific_head_2):
        # common loss (uniformity and alignement on weak representations)
        common_align_loss = align_loss(norm(weak_head_1.detach()), norm(common_head_1))
        common_align_loss +=  align_loss(norm(weak_head_2.detach()), norm(common_head_2))
        common_align_loss /= 2.0
        common_uniform_loss = (uniform_loss(norm(common_head_2)) + uniform_loss(norm(common_head_1))) / 2.0
        common_loss = common_align_loss + common_uniform_loss

        # specific loss (uniformity and alignement)
        specific_align_loss = align_loss(norm(specific_head_1), norm(specific_head_2))
        specific_uniform_loss = (uniform_loss(norm(specific_head_2)) + uniform_loss(norm(specific_head_1))) / 2.0
        specific_loss = specific_align_loss + specific_uniform_loss

        # mi minimization loss between weak and specific representations
        jem_loss = joint_entropy_loss(norm(specific_head_1), norm(weak_head_1.detach()))
        jem_loss = jem_loss + joint_entropy_loss(norm(specific_head_2), norm(weak_head_2.detach()))
        jem_loss = jem_loss / 2.0

        return common_loss, specific_loss, jem_loss

    
if __name__ == "__main__":
    setup_logging(level="debug")
    chkpt = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/mri/20240122_weak_encoder/exp-weakresnet18scz_ep-49.pth"
    model = StrongEncoder(backbone="resnet18", latent_dim=64, weak_encoder_chkpt=chkpt)
    #model.train(chkpt_dir="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/mri/20240124_strong_encoder",
    #           exp_name="resnet18scz", dataset="scz", ponderation=10, nb_epochs=50)
    model.test(chkpt_dir="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/mri/20240124_strong_encoder",
               exp_name="resnet18scz", dataset="scz", labels=["site"], list_epochs=[i for i in range(10, 50, 10)])