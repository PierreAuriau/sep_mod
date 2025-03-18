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
from torch.cuda.amp import GradScaler, autocast

from encoders import Encoder, Conv6Encoder, PytorchEncoder
from decoders import Decoder, Conv6Decoder, PytorchDecoder
from classifiers import Classifier
from loggers import TrainLogger

logging.setLoggerClass(TrainLogger)


# FIXME : add attributs for specific/shared indices of latent space DONE
# FIXME : add attributes for tc loss DONE

# FIXME : dataset return tensordict ?
# FIXME : forward : cat before or after reparametrize ?
# FIXME : optimizers for tc loss : one or several ?
# FIXME : handle hyperparameters : save and load
# FIXME : linear probe method or function outside class ?
# FIXME : check loss (regarding SepVAE) : remove ReLu from TC LOSS ? Change Sigmoid into softmax ?


class SepMod(nn.Module):

    def __init__(self, latent_dim=16,
                 tc_loss_between_specifics=False,
                 tc_loss_between_shared_and_specific=True, 
                 loss_between_shared="kl"):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "skeleton": PytorchEncoder(latent_dim=latent_dim//2),
            "image": PytorchEncoder(latent_dim=latent_dim//2),
            "shared": PytorchEncoder(latent_dim=latent_dim//2)
        })
        self.decoders = nn.ModuleDict({
            "skeleton": PytorchDecoder(latent_dim=latent_dim),
            "image": PytorchDecoder(latent_dim=latent_dim)
        })        

        self.discriminators = nn.ModuleDict()
        if tc_loss_between_shared_and_specific:
            self.discriminators["skeleton"] = Classifier(latent_dim=latent_dim)
            self.discriminators["image"] = Classifier(latent_dim=latent_dim)
        if tc_loss_between_specifics:
             self.discriminators["specific"] = Classifier(latent_dim=latent_dim)
        self.logger = logging.getLogger("sepmod")
        
        # Hyperparameters
        self.latent_dim = latent_dim
        self.specific_slice = slice(0, (latent_dim//2)) # see get_embeddings, train/valid step
        self.shared_slice = slice((latent_dim//2), latent_dim) # see get_embeddings, train/valid step
        self.tc_loss_between_specifics = tc_loss_between_specifics
        self.loss_between_shared = loss_between_shared
        self.tc_loss_between_shared_and_specific = tc_loss_between_shared_and_specific
        self.betas = defaultdict(lambda: 1.0)
        self.gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device used : {self.device}")
        if not (tc_loss_between_shared_and_specific or tc_loss_between_shared_and_specific):
            self.logger.warning("NO TC LOSS")
    
    @staticmethod
    def reparamatrize(mean, logvar):
        std = torch.exp(logvar/2)
        q = torch.distributions.Normal(mean, std)
        return q.rsample()
    
    @staticmethod
    def kl_divergence(mean, logvar, mean_=None, logvar_=None):
        if (mean_ is None) or (logvar_ is None):
            assert (mean_ is None) & (logvar_ is None), print('mean_ and logvar_ have to be None together.')
            kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        else:
            kl_div = -0.5 * torch.sum(1 + logvar - logvar_  - (logvar.exp() + (mean - mean_).pow(2)) / (2*logvar_.exp()))
        return kl_div

    def mmd(self, x, y):
        gammas = self.gammas.to(self.device)
        cost = torch.mean(self.gram_matrix(x, x, gammas=gammas)).to(self.device)
        cost += torch.mean(self.gram_matrix(y, y, gammas=gammas)).to(self.device)
        cost -= 2 * torch.mean(self.gram_matrix(x, y, gammas=gammas)).to(self.device)

        if cost < 0:
            return torch.tensor(0).to(self.device)
        return cost

    @staticmethod
    def gram_matrix(x, y, gammas):
        gammas = gammas.unsqueeze(1)
        pairwise_distances = torch.cdist(x, y, p=2.0)

        pairwise_distances_sq = torch.square(pairwise_distances)
        tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
        tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
        return tmp

    
    def loss_fn(self, inputs, outputs, mean_specific, logvar_specific, mean_shared, logvar_shared, joint_predictions, z):
        kl_loss = 0
        for mod in ("image", "skeleton"):
            kl_loss += self.kl_divergence(mean_specific[mod], logvar_specific[mod])
            kl_loss += self.kl_divergence(mean_shared[mod], logvar_shared[mod])
        rec_loss = 0
        for mod in ("image", "skeleton"):
            rec_loss += F.mse_loss(outputs[mod], inputs[mod], reduction="sum")
        tc_loss = 0
        if self.tc_loss_between_shared_and_specific:
            for mod in ("image", "skeleton"):
                tc_loss += torch.log(joint_predictions[mod] / (1 - joint_predictions[mod])).sum()
        if self.tc_loss_between_specifics:
            tc_loss += F.relu(torch.log(joint_predictions["specific"] / (1 - joint_predictions["specific"]))).sum()
        alg_loss = 0
        if self.loss_between_shared == "kl":
            alg_loss = self.kl_divergence(mean_shared["image"], logvar_shared["image"], mean_shared["skeleton"], logvar_shared["skeleton"])
        elif self.loss_between_shared == "mmd":
            alg_loss = self.mmd(z["image"][self.shared_slice], z["skeleton"][self.shared_slice])
        return kl_loss, rec_loss, tc_loss, alg_loss

    def forward(self, inputs):
        outputs = {}
        z = self.get_embeddings(inputs)
        for mod in ("image", "skeleton"):
            outputs[mod] = self.decoders[mod](z[mod])
        return outputs
    
    def get_embeddings(self, inputs):
        z = {}
        for mod in ("image", "skeleton"):
            mean_specific, logvar_specific = self.encoders[mod](inputs[mod])
            mean_shared, logvar_shared = self.encoders["shared"](inputs[mod])
            # FIXME
            # z[mod] = self.reparamatrize(torch.cat((mean_specific, mean_shared), dim=1), 
            #                            torch.cat((logvar_specific, logvar_shared), dim=1))
            z[mod] = torch.cat((self.reparamatrize(mean_specific, logvar_specific),
                                self.reparamatrize(mean_shared, logvar_shared)), 
                                dim=1)
        return z
    
    def configure_optimizers(self, **kwargs):
        vae_parameters = []
        for enc in self.encoders.values():
            vae_parameters += list(enc.parameters())
        for dec in self.decoders.values():
            vae_parameters += list(dec.parameters())
        optimizer_vae = optim.Adam(vae_parameters, **kwargs)
        if self.tc_loss_between_shared_and_specific or self.tc_loss_between_shared_and_specific:
            discriminator_parameters = []
            for disc in self.discriminators.values():
                discriminator_parameters += list(disc.parameters())
            optimizer_discriminators = optim.Adam(discriminator_parameters, lr=1e-5, 
                                                  weight_decay=1e-4)
            return optimizer_vae, optimizer_discriminators
        else:
            return optimizer_vae, None
    
    def fit(self, train_loader, nb_epochs, chkpt_dir):
        self.optimizer_vae, self.optimizer_discriminators = self.configure_optimizers(lr=1e-5)
        self.lr_scheduler = None
        self.scaler = GradScaler()
        self = self.to(self.device)
        self.logger.reset_history()
        self.save_hyperparameters(chkpt_dir)
        for epoch in range(nb_epochs):
            self.logger.info(f"Epoch: {epoch}")
            train_loss = 0

            self.train()
            self.logger.step()
            for inputs in tqdm(train_loader, desc="train"):
                for mod in ("image", "skeleton"):
                    inputs[mod] = inputs[mod].to(self.device)
                train_loss += self.training_step(inputs)
            self.logger.reduce(reduce_fx="sum")
            self.logger.store(epoch=epoch, set="train", loss=train_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Train loss: {train_loss:.2g}")
                self.logger.info(f"Training duration : {self.logger.get_duration()}")
                self.save_chkpt(os.path.join(chkpt_dir,
                                             f'sepmod_ep-{epoch}.pth'))
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(os.path.join(chkpt_dir,
                                     f'sepmod_ep-{epoch}.pth'))
        self.logger.info(f"End of training: {self.logger.get_duration()}")

    def training_step(self, inputs):

        # Discriminator loss
        if self.tc_loss_between_shared_and_specific or self.tc_loss_between_specifics:
            self.optimizer_discriminators.zero_grad()
            z = self.get_embeddings(inputs)
            discriminator_loss = 0
            if self.tc_loss_between_shared_and_specific:
                for mod in ("image", "skeleton"):
                    with autocast(dtype=torch.float32):
                        joint_predictions = self.discriminators[mod](z[mod], return_logits=True)
                        product_of_marginals_predictions = self.discriminators[mod](torch.cat((z[mod][:, self.specific_slice],
                                                                                               z[mod][torch.randperm(z[mod].size(0)), self.shared_slice]), 
                                                                                               dim=1),
                                                                                    return_logits=True)
                        discriminator_predictions = torch.cat((joint_predictions[:, 0], 
                                                               product_of_marginals_predictions[:, 0]), dim=0)
                        discriminator_targets = torch.cat((torch.ones_like(joint_predictions[:, 0]), 
                                                           torch.zeros_like(product_of_marginals_predictions[:, 0])), dim=0)
                        discriminator_loss += F.binary_cross_entropy_with_logits(discriminator_predictions, discriminator_targets, reduction="sum") # FIXME: remove logits ?
            
            if self.tc_loss_between_specifics:
                z_specific = torch.cat((z["image"][:, self.specific_slice] ,
                                        z["skeleton"][:, self.specific_slice]), 
                                        dim=1)
                with autocast(dtype=torch.float16):
                    joint_predictions = self.discriminators["specific"](z_specific)
                    product_of_marginals_predictions = self.discriminators["specific"]((torch.cat((z_specific[:, (self.latent_dim//2):],
                                                                                                   z_specific[torch.randperm(z_specific.size(0)), :(self.latent_dim//2)]), 
                                                                                                   dim=1)))
                    discriminator_predictions = torch.cat((joint_predictions[:, 0], 
                                                            product_of_marginals_predictions[:, 0]), dim=0)
                    discriminator_targets = torch.cat((torch.ones_like(joint_predictions[:, 0]), 
                                                       torch.zeros_like(product_of_marginals_predictions[:, 0])), dim=0)
                    discriminator_loss += F.binary_cross_entropy_with_logits(discriminator_predictions, discriminator_targets, reduction="sum")                                                                          
            self.scaler.scale(discriminator_loss).backward()
            self.scaler.step(self.optimizer_discriminators)
            self.logger.store(discriminator_loss=discriminator_loss.item())
        
        # VAE Loss
        self.optimizer_vae.zero_grad()
        mean_specific, logvar_specific = {}, {}
        mean_shared, logvar_shared = {}, {}
        z, outputs = {}, {}
        joint_predictions = {}
        with autocast(dtype=torch.float32):
            for mod in ("image", "skeleton"):
                mean_specific[mod], logvar_specific[mod] = self.encoders[mod](inputs[mod])
                mean_shared[mod], logvar_shared[mod] = self.encoders["shared"](inputs[mod])
                z[mod] = torch.cat((self.reparamatrize(mean_specific[mod], logvar_specific[mod]),
                                    self.reparamatrize(mean_shared[mod], logvar_shared[mod])), 
                                   dim=1)
                outputs[mod] = self.decoders[mod](z[mod])
                if self.tc_loss_between_shared_and_specific:
                    joint_predictions[mod] = self.discriminators[mod](z[mod])
            if self.tc_loss_between_specifics:
                z_specific = torch.cat((z["image"][:, self.specific_slice] ,
                                        z["skeleton"][:, self.specific_slice]), 
                                        dim=1)
                joint_predictions["specific"] = self.discriminators["specific"](z_specific)
            kl_loss, rec_loss, tc_loss, alg_loss = self.loss_fn(inputs, outputs, mean_specific, logvar_specific, mean_shared, logvar_shared, joint_predictions, z)
            # if torch.isnan(tc_loss):
            #     import pdb
            #     pdb.set_trace(header="Nan")
            # elif tc_loss <= 0:
            #     import pdb
            #     pdb.set_trace(header="<=0")
            loss = rec_loss + self.betas["kl_divergence"]*kl_loss
            self.logger.store(kl_loss=kl_loss.item(), reconstruction_loss=rec_loss.item())
            if self.tc_loss_between_shared_and_specific or self.tc_loss_between_specifics:
                loss += self.betas["total_correlation"]*tc_loss
                self.logger.store(total_correlation_loss=tc_loss.item())
                if np.isnan(tc_loss.item()):
                    raise ValueError("Nan in the total correlatoin loss !")
            if self.loss_between_shared is not None:
                loss += self.betas["alignement"]*alg_loss
                self.logger.store(alignement_loss=alg_loss.item())

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer_vae)
        self.scaler.update()            
        return loss.item()
    
    def valid_step(self, inputs):
        mean_specific, logvar_specific = {}, {}
        mean_shared, logvar_shared = {}, {}
        z, outputs = {}, {}
        joint_predictions = {}
        with torch.no_grad():
            for mod in ("image", "skeleton"):
                mean_specific[mod], logvar_specific[mod] = self.encoders[mod](inputs[mod])
                mean_shared[mod], logvar_shared[mod] = self.encoders["shared"](inputs[mod])
                z[mod] = torch.cat((self.reparamatrize(mean_specific[mod], logvar_specific[mod]),
                                    self.reparamatrize(mean_shared[mod], logvar_shared[mod])), 
                                   dim=1)
                outputs[mod] = self.decoders[mod](z[mod])
                if self.tc_loss_between_shared_and_specific:
                    joint_predictions[mod] = self.discriminators[mod](z[mod])
            if self.tc_loss_between_specifics:
                z_specific = torch.cat((z["image"][:, self.specific_slice] ,
                                        z["skeleton"][:, self.specific_slice]), 
                                    dim=1)
                joint_predictions["specific"] = self.discriminators["specific"](z_specific)
            kl_loss, rec_loss, tc_loss, alg_loss = self.loss_fn(inputs, outputs, mean_specific, logvar_specific, mean_shared, logvar_shared, joint_predictions, z)
            self.logger.store(kl_loss=kl_loss.item(), reconstruction_loss=rec_loss.item())
            loss = kl_loss + rec_loss
            if self.tc_loss_between_shared_and_specific or self.tc_loss_between_specifics:
                loss += tc_loss
                self.logger.store( total_correlation_loss=tc_loss.item())
            if self.kl_loss_between_shared:
                loss += self.betas["alignement"]*alg_loss
                self.logger.store(alignement_loss=alg_loss.item())
        return loss.item()
    
    def test(self, loader, epoch, chkpt_dir):
        self.load_chkpt(os.path.join(chkpt_dir,
                                f'sepmod_ep-{epoch}.pth'))
        self = self.to(self.device)
        self.eval()
        reconstructions = self.get_reconstructions(loader)
        test_loss = 0
        self.logger.step()
        for inputs in tqdm(loader, desc="test"):
            for mod in ("image", "skeleton"):
                inputs[mod] = inputs[mod].to(self.device)
            test_loss += self.valid_step(inputs)
        self.logger.reduce(reduce_fx="sum")
        self.logger.store(epoch=epoch, set="test", loss=test_loss)
        self.logger.info(f"Test loss : {test_loss:.2g}")
        self.logger.save(chkpt_dir, filename="_test")
        for mod in ("image", "skeleton"):
            np.save(os.path.join(chkpt_dir, f"reconstructions_mod-{mod}_ep-{epoch}.npy"), 
                    reconstructions[mod])
    
    def test_linear_probe(self, loader, epoch, chkpt_dir):
        self.eval()
        linear_probe = nn.Sequential(nn.Linear(self.latent_dim, 1), nn.Sigmoid())
        linear_probe = linear_probe.to(self.device)
        # optimizer
        for inputs in loader:
            for mod in ("image", "skeleton"):
                inputs[mod] = inputs[mod].to(self.device)
                outputs = linear_probe
        return 0

    def get_reconstructions(self, loader, raw=False, return_loss=False):
        self.eval()
        reconstructions = {"image": [], "skeleton": []}
        test_loss = {"image": [], "skeleton": []}
        with torch.no_grad():
            for inputs in loader:
                for mod in ("image", "skeleton"):
                    inputs[mod] = inputs[mod].to(self.device)
                outputs = self(inputs)
                for mod in ("image", "skeleton"):
                    test_loss[mod] += F.mse_loss(inputs[mod], outputs[mod], reduction="sum").item()
                    reconstructions[mod].extend(outputs[mod].cpu().numpy())
        self.logger.info(f"Reconstruction error: {test_loss}")
        for mod in ("image", "skeleton"):
            outputs = np.array(reconstructions[mod])
            if not raw:
                # normalize outputs
                # outputs = (outputs - outputs.min(axis=0)) / (outputs.max(axis=0) - outputs.min(axis=0)) * 255
                outputs = (np.clip(outputs, -1, 1)*0.5 + 0.5)
                outputs = outputs.astype(int)
            # put channels in last
            outputs = np.moveaxis(outputs, 1, -1)
            reconstructions[mod] = outputs
        if return_loss:
            return reconstructions, test_loss
        return reconstructions
    
    def save_hyperparameters(self, chkpt_dir):
        hp = {"latent_dim": self.latent_dim,
              "tc_loss_between_shared_and_specific": self.tc_loss_between_shared_and_specific,
              "tc_loss_between_specifics": self.tc_loss_between_specifics,
              "loss_between_shared": self.loss_between_shared}
        for key, value in self.betas.items():
            hp[f"beta_{key}"] = value
        with open(os.path.join(chkpt_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f)
        
    def save_chkpt(self, filename):
        torch.save({"encoders": self.encoders.state_dict(),
                    "decoders": self.decoders.state_dict(),
                    "discriminators": self.discriminators.state_dict()},
                   filename)

    def load_chkpt(self, filename):
        chkpt = torch.load(filename)
        status = self.encoders.load_state_dict(chkpt["encoders"], strict=False)
        self.logger.info(f"Loading encoders : {status}")
        status = self.decoders.load_state_dict(chkpt["decoders"], strict=False)
        self.logger.info(f"Loading decoder : {status}")
        status = self.discriminators.load_state_dict(chkpt["discriminators"], strict=False)
        self.logger.info(f"Loading discriminators : {status}")