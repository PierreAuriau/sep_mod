# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from encoders import Encoder, Conv7Encoder, AntixKEncoder
from decoders import Decoder, Conv7Decoder, AntixKDecoder
from loggers import TrainLogger

logging.setLoggerClass(TrainLogger)


class VAE(nn.Module):

    def __init__(self, input_channels, latent_dim=256, beta=1.0, nb_layers=None):
        super().__init__()
        if nb_layers == 7:
            self.encoder = Conv7Encoder(input_channels=input_channels,
                                        latent_dim=latent_dim)
            self.decoder = Conv7Decoder(input_channels=input_channels,
                                        latent_dim=latent_dim)
        elif nb_layers == 6:
            self.encoder = AntixKEncoder(input_channels=input_channels,
                                         latent_dim=latent_dim)
            self.decoder = AntixKDecoder(input_channels=input_channels,
                                         latent_dim=latent_dim)
        else:
            self.encoder = Encoder(input_channels=input_channels,
                                   latent_dim=latent_dim)
            self.decoder = Decoder(input_channels=input_channels,
                                   latent_dim=latent_dim)
        """self.scale = nn.Parameter(torch.tensor([0.0]))"""
        self.nb_layers = nb_layers if nb_layers in [6, 7] else 4
        self.beta = beta # FIXME: in init or in fit method ?
        self.latent_dim = latent_dim
        self.logger = logging.getLogger("vae")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device used : {self.device}")
        
    def reparamatrize(self, mean, logvar):
        std = torch.exp(logvar/2)
        q = torch.distributions.Normal(mean, std)
        return q.rsample()
    """
    def kl_loss(self, z, mean, std):
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        q = torch.distributions.Normal(mean, std)
        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)
        kl_loss = (log_qzx - log_pz)
        kl_loss = kl_loss.sum(-1)
        return kl_loss.mean()   
    
    def gaussian_likelihood(self, inputs, outputs, scale):
        dist = torch.distributions.Normal(outputs,torch.exp(self.scale))
        log_pxz = dist.log_prob(inputs)
        return log_pxz.sum(dim=(1,2,3))
    """
    def loss_fn(self, inputs, outputs, mean, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
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
    
    def fit(self, train_loader, val_loader, nb_epochs, 
            modality, chkpt_dir, **kwargs_optimizer):
        
        self.optimizer = self.configure_optimizers(**kwargs_optimizer)
        self.lr_scheduler = None
        self.save_hyperparameters(chkpt_dir)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.logger.reset_history()
        best_epoch, best_loss = 0, np.inf
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
            
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()
            self.logger.step()
            for batch in tqdm(val_loader, desc="val"):
                inputs = batch[modality].to(self.device)
                val_loss += self.valid_step(inputs)
            self.logger.reduce(reduce_fx="sum")
            self.logger.store(epoch=epoch, set="validation", loss=val_loss)

            if val_loss <= best_loss:
                self.logger.info(f"Validation loss improvement : {best_loss:.2g} --> {val_loss:.2g}")
                best_loss = val_loss
                best_epoch = epoch
                self.save_chkpt(os.path.join(chkpt_dir,
                                             f'vae_mod-{modality}_ep-best.pth'))
            if epoch % 10 == 0:
                self.logger.info(f"Train loss: {train_loss:.2g} - Val loss: {val_loss:.2g}")
                self.logger.info(f"Training duration: {self.logger.get_duration()}")
                self.save_chkpt(os.path.join(chkpt_dir,
                                             f'vae_mod-{modality}_ep-{epoch}.pth'))
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(os.path.join(chkpt_dir,
                                     f'vae_mod-{modality}_ep-{epoch}.pth'))
        self.logger.info(f"End of training: {self.logger.get_duration()}")
        self.logger.info(f"Best epoch: {best_epoch} - best val loss: {val_loss}")

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
    
    def valid_step(self, inputs):
        with torch.no_grad():
            mean, logvar = self.encoder(inputs)
            z = self.reparamatrize(mean, logvar)
            outputs = self.decoder(z)
            rec_loss, kl_loss = self.loss_fn(inputs, outputs, mean, logvar)
            self.logger.store(reconstruction_loss=rec_loss.item(), kl_divergence=kl_loss.item())
            loss = rec_loss + self.beta*kl_loss
        return loss.item()

    def test(self, loader, epoch, modality, chkpt_dir):
        self.load_chkpt(os.path.join(chkpt_dir,
                                f'vae_mod-{modality}_ep-{epoch}.pth'))
        reconstructions = self.get_reconstructions(loader, modality)
        test_loss = 0
        self.logger.step()
        for batch in tqdm(loader, desc="test"):
            inputs = batch[modality].to(self.device)
            test_loss += self.valid_step(inputs)
        self.logger.info(f"Test loss : {test_loss:.2g}")
        self.logger.reduce(reduce_fx="sum")
        self.logger.store(epoch=epoch, set="test", loss=test_loss)
        self.logger.save(chkpt_dir, filename="_test")
        filename = f"reconstructions_set-test_ep-{epoch}.npy"
        np.save(os.path.join(chkpt_dir, filename), reconstructions)

    def get_reconstructions(self, loader, modality):
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
        # normalize outputs
        if self.nb_layers == 7: # with sigmoid
            outputs = outputs * 255
        elif self.nb_layers == 6: # with tanh
            outputs = (outputs + 1) * (255 / 2)
        else: # no sigmoid
            outputs = (outputs - outputs.min(axis=0)) / (outputs.max(axis=0) - outputs.min(axis=0)) * 255
        assert np.all(outputs >= 0) & np.all(outputs <= 256), \
            f"outputs of decoder have wrong data range ({outputs.min(), outputs.max()})"
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
