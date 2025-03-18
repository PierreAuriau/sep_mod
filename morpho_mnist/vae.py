# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error

from encoders import Encoder, Conv6Encoder, PytorchEncoder
from decoders import Decoder, Conv6Decoder, PytorchDecoder
from loggers import TrainLogger

logging.setLoggerClass(TrainLogger)


class VAE(nn.Module):

    def __init__(self, latent_dim=256, beta=1.0, nb_layers=None):
        super().__init__()
        if nb_layers == 6:
            self.encoder = Conv6Encoder(latent_dim=latent_dim)
            self.decoder = Conv6Decoder(latent_dim=latent_dim)
        elif nb_layers == 5:
            self.encoder = PytorchEncoder(latent_dim=latent_dim)
            self.decoder = PytorchDecoder(latent_dim=latent_dim)
        else:
            self.encoder = Encoder(latent_dim=latent_dim)
            self.decoder = Decoder(latent_dim=latent_dim)
        self.nb_layers = nb_layers if nb_layers in [6, 5] else 3
        self.latent_dim = latent_dim
        self.beta = beta # FIXME: in init or in fit method ?
        self.reconstruction_loss = "mse"
        self.logger = logging.getLogger("vae")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device used : {self.device}")
        
    def reparamatrize(self, mean, logvar):
        std = torch.exp(logvar/2)
        q = torch.distributions.Normal(mean, std)
        return q.rsample()

    def loss_fn(self, inputs, outputs, mean, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        if self.reconstruction_loss == "bce":
            rec_loss = F.binary_cross_entropy(outputs, inputs, reduction="sum")
        elif self.reconstruction_loss == "l1":
            rec_loss = F.l1_loss(outputs, inputs, reduction="sum")
        else:
            rec_loss = F.mse_loss(outputs, inputs, reduction="sum")
        return rec_loss, kl_loss

    def forward(self, inputs):
        mean, logvar = self.encoder(inputs)
        z = self.reparamatrize(mean, logvar)
        outputs = self.decoder(z)
        return outputs
    
    def get_embeddings(self, inputs):
        mean, logvar = self.encoder(inputs)
        z = self.reparamatrize(mean, logvar)
        return z
    
    def configure_optimizers(self, **kwargs):
        return optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                          **kwargs)
    
    def fit(self, train_loader, nb_epochs, 
            modality, chkpt_dir, **kwargs_optimizer):
        
        self.optimizer = self.configure_optimizers(**kwargs_optimizer)
        self.lr_scheduler = None
        self.save_hyperparameters(chkpt_dir)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.logger.reset_history()
        for epoch in range(nb_epochs):
            self.logger.info(f"Epoch: {epoch}")
            train_loss, val_loss = 0, 0

            self.encoder = self.encoder.train()
            self.decoder = self.decoder.train()
            self.logger.step()
            for batch in tqdm(train_loader, desc="train"):
                inputs = batch[modality].to(self.device)
                loss = self.training_step(inputs)
                train_loss += loss
            self.logger.reduce(reduce_fx="sum")
            self.logger.store(epoch=epoch, set="train", loss=train_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Train loss: {train_loss:.2g}")
                self.logger.info(f"Training duration: {self.logger.get_duration()}")
                self.save_chkpt(os.path.join(chkpt_dir,
                                             f'vae_mod-{modality}_ep-{epoch}.pth'))
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(os.path.join(chkpt_dir,
                                     f'vae_mod-{modality}_ep-{epoch}.pth'))
        self.logger.info(f"End of training: {self.logger.get_duration()}")

    def training_step(self, inputs):
        self.optimizer.zero_grad()
        mean, logvar = self.encoder(inputs)
        z = self.reparamatrize(mean, logvar)
        outputs = self.decoder(z)
        rec_loss, kl_loss = self.loss_fn(inputs, outputs, mean, logvar)
        self.logger.store(reconstruction_loss=rec_loss.item(), kl_divergence=kl_loss.item())
        loss = rec_loss + self.beta*kl_loss
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()

    def test(self, loader, epoch, modality, chkpt_dir):
        self.load_chkpt(os.path.join(chkpt_dir,
                                f'vae_mod-{modality}_ep-{epoch}.pth'))
        reconstructions = self.get_reconstructions(loader, modality)
        test_loss = 0
        self.logger.step()
        for batch in tqdm(loader, desc="test"):
            inputs = batch[modality].to(self.device)
            with torch.no_grad():
                mean, logvar = self.encoder(inputs)
                z = self.reparamatrize(mean, logvar)
                outputs = self.decoder(z)
                rec_loss, kl_loss = self.loss_fn(inputs, outputs, mean, logvar)
                self.logger.store(reconstruction_loss=rec_loss.item(), kl_divergence=kl_loss.item())
                loss = rec_loss + self.beta*kl_loss
            test_loss += loss.item()
        self.logger.info(f"Test loss : {test_loss:.2g}")
        self.logger.reduce(reduce_fx="sum")
        self.logger.store(epoch=epoch, set="test", loss=test_loss)
        self.logger.save(chkpt_dir, filename="_test")
        filename = f"reconstructions_set-test_ep-{epoch}.npy"
        np.save(os.path.join(chkpt_dir, filename), reconstructions)

    def test_linear_probe(self, train_loader, test_loader, epoch, modality, targets, chkpt_dir):
        self.load_chkpt(os.path.join(chkpt_dir,
                                f'vae_mod-{modality}_ep-{epoch}.pth'))
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        # Get input data
        self.logger.info("Get embeddings")
        z = {s: [] for s in ["train", "test"]}
        y =  {s: defaultdict(list) for s in ["train", "test"]}
        with torch.no_grad():
            for split, loader in zip(["train", "test"], 
                                    [train_loader, test_loader]):
                for batch in tqdm(loader, desc=split):
                    inputs = batch[modality].to(self.device)
                    z[split].extend(self.get_embeddings(inputs).cpu().numpy())
                    for target in targets:
                        y[split][target].extend(batch[target].cpu().numpy())
                z[split] = np.asarray(z[split])
                for target in targets:
                    y[split][target] = np.asarray(y[split][target])

        # Train and test linear models
        logs = defaultdict(list)       
        for target in targets:
            if target in ["label", "fracture"]:
                log_reg = LogisticRegression(penalty="l2", C=1.0, fit_intercept=True, max_iter=1000)
                log_reg.fit(z["train"], y["train"][target])
                for split in ("train", "test"):
                    self.logger.info(f"Prediction of {target} on {split} set: {log_reg.score(z[split], y[split][target])}")
                    y_pred = log_reg.predict(z[split])
                    bacc = balanced_accuracy_score(y_pred=y_pred, y_true=y[split][target])
                    logs["target"].append(target)
                    logs["split"].append(split)
                    logs["metric"].append("balanced_accuracy")
                    logs["value"].append(bacc)
            else:
                ridge = Ridge(alpha=0.1, fit_intercept=True, solver="auto")
                ridge.fit(z["train"], y["train"][target])
                for split in ("train", "test"):
                    self.logger.info(f"Prediction of {target} on {split} set: {ridge.score(z[split], y[split][target])}")
                    y_pred = ridge.predict(z[split])
                    r2 = ridge.score(z[split], y[split][target])
                    rmse = mean_squared_error(y_pred=y_pred, y_true=y[split][target], squared=False) 
                    mae = mean_absolute_error(y_pred=y_pred, y_true=y[split][target])

                    for metric, value in zip(["r2", "root_mean_squarred_error", "mean_absolute_error"], 
                                            [r2, rmse, mae]):
                        logs["target"].append(target)
                        logs["split"].append(split)
                        logs["metric"].append(metric)
                        logs["value"].append(value)
            
        logs = pd.DataFrame(logs)
        logs.to_csv(os.path.join(chkpt_dir, "linear_predictions_from_embeddings.csv"), index=False)

    def get_reconstructions(self, loader, modality, raw=False):
        self.encoder.eval()
        self.decoder.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        outputs = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[modality].to(self.device)
                outputs.extend(self(inputs).cpu().numpy())
        outputs = np.asarray(outputs)
        if raw:
            return outputs
        # normalize outputs
        if self.nb_layers == 6: # with sigmoid
            outputs = outputs * 255
        else: # no sigmoid
            outputs = (outputs - outputs.min(axis=0)) / (outputs.max(axis=0) - outputs.min(axis=0)) * 255
        outputs = outputs.astype(int)
        # put channels in last
        outputs = np.moveaxis(outputs, 1, -1)
        return outputs
    
    def save_hyperparameters(self, chkpt_dir):
        hp = {"latent_dim": self.latent_dim,
              "nb_layers": self.nb_layers,
              "beta": self.beta}
        with open(os.path.join(chkpt_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f)

    def save_chkpt(self, filename):
        torch.save({"encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict()},
                   filename)

    def load_chkpt(self, filename):
        chkpt = torch.load(filename)
        status = self.encoder.load_state_dict(chkpt["encoder"], strict=False)
        self.logger.info(f"Loading encoder : {status}")
        status =self.decoder.load_state_dict(chkpt["decoder"], strict=False)
        self.logger.info(f"Loading decoder : {status}")
    