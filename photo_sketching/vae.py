# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from encoders import Encoder, Conv7Encoder
from decoders import Decoder, Conv7Decoder
from loggers import TrainLogger


class VAE(nn.Module):

    def __init__(self, input_channels, latent_dim=256):
        super().__init__()
        self.encoder = Conv7Encoder(input_channels=input_channels,
                                    latent_dim=latent_dim)
        self.decoder = Conv7Decoder(input_channels=input_channels,
                                    latent_dim=latent_dim)
        """self.scale = nn.Parameter(torch.tensor([0.0]))"""
        self.logger = logging.getLogger("vae")
        
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
        return kl_loss + rec_loss

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
        
        self.train_logger = TrainLogger("trainlogger")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device used : {self.device}")
        self.optimizer = self.configure_optimizers(**kwargs_optimizer)
        self.lr_scheduler = None
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        best_epoch, best_loss = 0, np.inf
        for epoch in range(nb_epochs):
            self.train_logger.step()
            self.train_logger.store(epoch=epoch)
            self.logger.info(f"Epoch: {epoch}")
            train_loss, val_loss = 0, 0
    
            self.encoder = self.encoder.train()
            self.decoder = self.decoder.train()
            for batch in tqdm(train_loader, desc="train"):
                inputs = batch[modality].to(self.device)
                loss = self.training_step(inputs)
                train_loss += loss
            
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()
            for batch in tqdm(val_loader, desc="val"):
                inputs = batch[modality].to(self.device)
                val_loss += self.valid_step(inputs)

            self.train_logger.reduce(reduce_fx="sum")
            self.train_logger.store(train_loss=train_loss, val_loss=val_loss)
            if val_loss <= best_loss:
                best_loss = val_loss
                best_epoch = epoch
                self.logger.info(self.train_logger.summary())
                self.save_chkpt(os.path.join(chkpt_dir,
                                             f'vae_mod-{modality}_ep-best.pth'))
            if epoch % 10 == 0:
                self.logger.info(self.train_logger.summary())
                self.save_chkpt(os.path.join(chkpt_dir,
                                             f'vae_mod-{modality}_ep-{epoch}.pth'))
        
        self.train_logger.save(chkpt_dir)
        self.save_chkpt(os.path.join(chkpt_dir,
                                     f'vae_mod-{modality}_ep-{epoch}.pth'))
        self.logger.info(f"End of training: {self.train_logger.get_duration()}")
        self.logger.info(f"Best epoch: {best_epoch} --> best loss : {val_loss}")

    def training_step(self, inputs):
        self.optimizer.zero_grad()
        mean, logvar = self.encoder(inputs)
        z = self.reparamatrize(mean, logvar)
        outputs = self.decoder(z)
        loss = self.loss_fn(inputs, outputs, mean, logvar)
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
            loss = self.loss_fn(inputs, outputs, mean, logvar)
        return loss.item()

    def test(self, loader, epoch, modality, chkpt_dir):
        self.load_chkpt(os.path.join(chkpt_dir,
                                f'vae_mod-{modality}_ep-{epoch}.pth'))
        reconstructions = self.get_reconstructions(loader)
        test_loss = 0
        for batch in tqdm(loader, desc="test"):
            inputs = batch[modality].to(self.device)
            test_loss += self.valid_step(inputs)
        self.logger.info(f"Test loss : {test_loss:.2g}")
        np.save(os.path.join(chkpt_dir, f"reconstructions_ep-{epoch}.npy"), reconstructions)

    def get_reconstructions(self, loader):
        self.encoder.eval()
        self.decoder.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        outputs = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[modality].to(model.device)
                outputs.extend(self(inputs).cpu().numpy())
        outputs = np.asarray(outputs)
        # normalize outputs
        outputs = (outputs - outputs.min(axis=0)) / (outputs.max(axis=0) - outputs.min(axis=0)) * 255
        outputs = outputs.astype(int)
        # put channels in last
        outputs = np.moveaxis(outputs, 1, -1)
        return outputs

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
