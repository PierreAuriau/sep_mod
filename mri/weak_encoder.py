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
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge

# project import
from loss import align_loss, uniform_loss, norm
from datamanager import ClinicalDataManager
from logs import setup_logging, save_hyperparameters, History
from encoder import Encoder

logger = logging.getLogger("WeakEncoder")

class WeakEncoder:
    def __init__(self, backbone, latent_dim):
        
        self.weak_encoder = Encoder(backbone=backbone, n_embedding=latent_dim)
        self.backbone = backbone
        self.n_embedding = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device : {self.device}")

    def train(self, chkpt_dir, exp_name, dataset, nb_epochs, 
              data_augmentation, nb_epochs_per_saving=10):
        # loader
        manager = ClinicalDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                      db=dataset, preproc="skeleton", labels=None, 
                                      batch_size=8, two_views=True,
                                      num_workers=8, pin_memory=True)
        loader = manager.get_dataloader(train=True, validation=False)
        nb_batch = len(loader.train)
        # define optimizers and scaler
        optimizer = self.configure_optimizers()
        scaler = GradScaler()
        # prepare attributes
        self.checkpointdir = chkpt_dir
        self.exp_name = exp_name
        self.history = History(name="Train_WeakEncoder", chkpt_dir=chkpt_dir)

        # train model
        self.weak_encoder.to(self.device)
        
        for epoch in range(nb_epochs):
            pbar = tqdm(total=nb_batch, desc=f"Epoch {epoch}")
            self.history.step()
            # train
            self.weak_encoder.train()
            train_loss = 0
            for dataitem in loader.train:
                pbar.update()
                inputs = dataitem.inputs
                loss = self.training_step(inputs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            pbar.close()
            self.history.log(epoch=epoch, train_loss=train_loss)
            self.history.reduce(reduce_fx="sum")
        if (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) and epoch > 0:
            self.history.summary()
            self.save_checkpoint(epoch=epoch,
                                 n_embedding=self.n_embedding,
                                 backbone=self.backbone, 
                                 optimizer=optimizer.state_dict())
            self.history.save()
        logger.info(f"Training duration: {self.history.get_duration()}")

    def training_step(self, batch):
        weak_view_1 = batch.view_1.to(self.device)
        weak_view_2 = batch.view_2.to(self.device)
        with autocast():
            _, weak_head_1 = self.weak_encoder(weak_view_1)
            _, weak_head_2 = self.weak_encoder(weak_view_2)
            loss = self.loss_fn(weak_head_1, weak_head_2)
        return loss 

    def test(self, chkpt_dir, exp_name, dataset, labels, list_epochs):
        manager = ClinicalDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                      db=dataset, preproc="skeleton", labels=labels, 
                                      batch_size=32, two_views=False,
                                      num_workers=8, pin_memory=True)
        self.exp_name = exp_name
        self.checkpointdir = chkpt_dir
        self.history = History(name="Test_StrongEncoder", chkpt_dir=chkpt_dir)

        for epoch in list_epochs:
            self.looad_checkpoint(epoch=epoch)
            self.weak_encoder = self.weak_encoder.to(self.device)        
            self.weak_encoder.eval()
            
            loader = manager.get_dataloader(train=True,
                                            validation=True,
                                            test=True)
            # get embeddings
            representations = {}
            label = {}
            filename = os.path.join(self.checkpointdir, self.get_representation_name(epoch=epoch))
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    saved_repr = pickle.load(f)
                logger.info(f"Loading representations : {filename}")
                for split in ("train", "validation", "test", "test_intra"):
                    representations[split] = saved_repr[split]
                    label[split] = manager.dataset[split].get_target()
            else:
                for split in ("train", "validation", "test", "test_intra"):
                    logger.info(f"{split} set")
                    if split == "test_intra":
                        loader = manager.get_dataloader(test_intra=True)
                        representations[split], _ = self.get_embeddings(loader.test)
                    else:
                        representations[split], _ = self.get_embeddings(getattr(loader, split))
                    label[split] = manager.dataset[split].get_target()
                self.save_representations(representations=representations, epoch=epoch)
            # train predictors
            for i, label  in enumerate(labels):
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
                clf = clf.fit(representations["train"], labels["train"][:, i])
                for split in splits:
                    y_pred = clf.get_predictions(representations[split])
                    values = {}
                    for name, metric in metrics.items():
                        values[name] = metric(y_pred=y_pred, y_true=labels[split][:, i])
                    self.history.step()
                    self.history.log(epoch=epoch, label=label, set=split, **values)
            self.history.save()
            
    def get_embeddings(self, dataloader):
        representations = []
        heads = []
        pbar = tqdm(total=len(dataloader), desc=f"Get embeddings")
        for dataitem in dataloader:
            pbar.update()
            with torch.no_grad():
                inputs = dataitem.inputs
                repr, head = self.weak_encoder(inputs.to(self.device))
            representations.extend(repr.detach().cpu().numpy())
            heads.extend(head.detach().cpu().numpy())
        pbar.close()
        return np.asarray(representations), np.asarray(heads)
    
    def save_representations(self, representations, epoch):
        filename = os.path.join(self.checkpointdir, self.get_representation_name(epoch=epoch))
        with open(filename, "wb") as f:
            pickle.dump(representations, f)
    
    def get_representatinon_name(self, epoch):
        return f"Representations_WeakEncoder_exp-{self.exp_name}_ep-{epoch}.pkl"
        
    @staticmethod
    def loss_fn(z_1, z_2):
        weak_align_loss = align_loss(norm(z_1), norm(z_2))
        weak_uniform_loss = (uniform_loss(norm(z_1)) + uniform_loss(norm(z_2))) / 2.0
        return weak_align_loss + weak_uniform_loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.weak_encoder.parameters(), lr=1e-4)
        return optimizer
    
    def save_checkpoint(self, epoch, **kwargs):
        outfile = os.path.join(self.checkpointdir, self.get_chkpt_name(epoch))
        torch.save({
            "epoch": epoch,
            "weak_encoder": self.weak_encoder.state_dict(),
            **kwargs}, outfile)
    
    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.checkpointdir, 
                                             self.get_chkpt_name(epoch=epoch)))
        status = self.weak_encoder.load_state_dict(checkpoint["weak_encoder"], strict=False)
        logger.info(f"Loading weak encoder: {status}")
    
    def get_chkpt_name(self, epoch):
        return f"WeakEncoder_exp-{self.exp_name}_ep-{epoch}.pth"


def parse_args(argv):
    parser = argparse.Argparse()
    parser.add_argument("--backbone", requires=True, )
    parser.add_argument("--latent_dim", requires=True)
    parser.add_argument("--chkpt_dir", requires=True) 
    parser.add_argument("--exp_name", requires=True)
    parser.add_argument("--dataset", requires=True)
    parser.add_argument("--nb_epochs", default=50)
    parser.add_argument("--data_augmentation")
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--verbose")

if __name__ == "__main__":
    pass