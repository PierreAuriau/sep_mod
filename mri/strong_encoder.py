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
import json

from sklearn.metrics import roc_auc_score, mean_squared_error, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

# project import
from loss import align_loss, uniform_loss, norm, joint_entropy_loss
from datamanager import TwoModalityDataManager, TwoViewItem
from logs import History, setup_logging
from encoder import Encoder

logger = logging.getLogger("StrongEncoder") 

# UDPATES:
# - data augmentation : change DAModule OK
# - save hyperparameters --> in main ? OK
# - improve test step : iterate over encoder in test (set dico representations instead of z.. and change returns in get embeddings) OK
# - remove get target by target OK
# - rename checkpoint_dir into chkpt_dir OK 
# ----------------------------------------
# * outputs of models as namedtuple or dict, or parameter in forward or init
# - improve training_step : add epoch attributes to log
# - keep weak encoder ? load weak encoder in init or train/test ? (save or not weak encoder)


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
    
    def train(self, chkpt_dir, exp_name, dataset, ponderation, nb_epochs, 
              data_augmentation, jem_loss_config, nb_epochs_per_saving=10):
        
        logger.info(f"Train model: {exp_name} for {nb_epochs} epochs")
        # loader
        manager = TwoModalityDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                         db=dataset, weak_modality="skeleton", strong_modality="vbm",
                                         labels=None, batch_size=24, two_views=True,
                                         data_augmentation=data_augmentation,
                                         num_workers=8, pin_memory=True)
        loader = manager.get_dataloader(train=True, validation=False)
        nb_batch = len(loader.train)
        # define optimizer and scaler
        optimizer = self.configure_optimizers()
        scaler = GradScaler()
        # prepare variables
        self.chkpdt_dir = chkpt_dir
        self.exp_name = exp_name
        self.ponderation = ponderation
        self.jem_loss_config = jem_loss_config
        logger.debug(f"jem loss config {jem_loss_config}")
        self.history = History(name=f"Train_StrongEncoder_exp-{exp_name}", chkpt_dir=self.chkpdt_dir)
        self.save_hyperparameters(dataset=dataset, ponderation=ponderation, nb_epochs=nb_epochs,
                                  data_augmentation=data_augmentation, jem_loss_config=jem_loss_config)

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
            if (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1):
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
                weak_head = TwoViewItem(
                    view_1=self.weak_encoder(weak_view_1)[1],
                    view_2=self.weak_encoder(weak_view_2)[1])
            specific_head = TwoViewItem(
                view_1=self.specific_encoder(strong_view_1)[1],
                view_2=self.specific_encoder(strong_view_2)[1])
            common_head = TwoViewItem(
                view_1=self.common_encoder(strong_view_1)[1],
                view_2=self.common_encoder(strong_view_2)[1])

            # loss
            co_loss, spe_loss, j_loss = self.loss_fn(weak_head=weak_head, 
                                                     common_head=common_head, 
                                                     specific_head=specific_head)
        return co_loss, spe_loss, j_loss
    
    def loss_fn(self, weak_head, common_head, specific_head):
        # common loss (uniformity and alignement on weak representations)
        common_align_loss = align_loss(norm(weak_head.view_1.detach()), norm(common_head.view_1))
        common_align_loss +=  align_loss(norm(weak_head.view_2.detach()), norm(common_head.view_2))
        common_align_loss /= 2.0
        common_uniform_loss = (uniform_loss(norm(common_head.view_2)) + uniform_loss(norm(common_head.view_1))) / 2.0
        common_loss = common_align_loss + common_uniform_loss

        # specific loss (uniformity and alignement)
        specific_align_loss = align_loss(norm(specific_head.view_1), norm(specific_head.view_2))
        specific_uniform_loss = (uniform_loss(norm(specific_head.view_2)) + uniform_loss(norm(specific_head.view_1))) / 2.0
        specific_loss = specific_align_loss + specific_uniform_loss

        # mi minimization loss between weak and specific representations
        if self.jem_loss_config == "specific-weak":
            #logger.warning("JEM LOSS BETWEEN SPECIFIC AND WEAK!")
            jem_loss = joint_entropy_loss(norm(specific_head.view_1), norm(weak_head.view_1.detach()))
            jem_loss = jem_loss + joint_entropy_loss(norm(specific_head.view_2), norm(weak_head.view_2.detach()))
        elif self.jem_loss_config == "specific-common":
            #logger.warning("JEM LOSS BETWEEN SPECIFIC AND COMMON!")
            jem_loss = joint_entropy_loss(norm(specific_head.view_1), norm(common_head.view_1))
            jem_loss = jem_loss + joint_entropy_loss(norm(specific_head.view_2), norm(common_head.view_2))
            jem_loss = jem_loss / 2.0
                
        return common_loss, specific_loss, jem_loss
        
    def test(self, chkpt_dir, exp_name, dataset, labels, list_epochs):
        logger.info(f"Test model: {exp_name} on {labels}")
        # datamanagers
        manager = TwoModalityDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                           db=dataset, strong_modality="vbm", weak_modality="skeleton", 
                                           labels=labels, batch_size=8, two_views=False,
                                           num_workers=8, pin_memory=True)
        self.exp_name = exp_name
        self.chkpdt_dir = chkpt_dir
        self.history = History(name=f"Test_StrongEncoder_exp-{exp_name}", chkpt_dir=self.chkpdt_dir)
        
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
            representations = {"specific": {}, "common": {}, "weak": {}}
            y_true = {} # ground truth labels
            filename = os.path.join(self.chkpdt_dir, self.get_representation_name(epoch=epoch))
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    saved_repr = pickle.load(f)
                logger.info(f"Loading representations : {filename}")
                representations = saved_repr
                for split in ("train", "validation", "test", "test_intra"):
                    y_true[split] = manager.dataset[split].target
            else:
                for split in ("train", "validation", "test", "test_intra"):
                    logger.info(f"{split} set")
                    if split == "test_intra":
                        loader = manager.get_dataloader(test_intra=True)
                        embeddings = self.get_embeddings(dataloader=loader.test)
                        for enc, repr in embeddings.items():
                            representations[enc][split] = repr 
                    else:
                        embeddings = self.get_embeddings(dataloader=getattr(loader, split))
                        for enc, repr in embeddings.items():
                            representations[enc][split] = repr
                    y_true[split] = manager.dataset[split].target
                self.save_representations(representatinos=representations, epoch=epoch)

            # train predictors
            for i, label in enumerate(labels):
                splits = ["train", "validation", "test", "test_intra"]
                if label in ("diagnosis", "sex"):
                    clf = LogisticRegression(max_iter=1000)
                    clf.get_predictions = clf.predict_proba
                    metrics = {"roc_auc": lambda y_pred, y_true: roc_auc_score(y_score=y_pred[:, 1], y_true=y_true),
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
                for encoder in ("weak", "common", "specific"):
                    logger.info(f"Test {encoder} encoder on {label}")
                    clf = clf.fit(representations[encoder]["train"], y_true["train"][:, i])
                    for split in splits:
                        y_pred = clf.get_predictions(representations[encoder][split])
                        values = {}
                        for name, metric in metrics.items():
                            values[name] = metric(y_pred=y_pred, y_true=y_true[split][:, i])
                        self.history.step()
                        self.history.log(epoch=epoch, encoder=encoder, label=label, set=split, **values)
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
        representations = {"common": np.asarray(common_representations),
                           "specific": np.asarray(specific_representations),
                           "weak": np.asarray(weak_representations)}
        return representations
    
    def configure_optimizers(self):
        return Adam(list(self.specific_encoder.parameters()) + list(self.common_encoder.parameters()), 
                    lr=1e-4, weight_decay=5e-5)
        
    def save_checkpoint(self, epoch, **kwargs):
        outfile = os.path.join(self.chkpdt_dir, self.get_chkpt_name(epoch))
        torch.save({
            "epoch": epoch,
            "weak_encoder": self.weak_encoder.state_dict(),
            "specific_encoder": self.specific_encoder.state_dict(),
            "common_encoder": self.common_encoder.state_dict(),
            **kwargs}, outfile)
        return outfile
    
    def load_from_checkpoint(self, epoch):
        filename = os.path.join(self.chkpdt_dir, self.get_chkpt_name(epoch))
        checkpoint = torch.load(filename)
        status = self.weak_encoder.load_state_dict(checkpoint["weak_encoder"], strict=False)
        logger.info(f"Loading weak encoder : {status}")
        status = self.specific_encoder.load_state_dict(checkpoint["specific_encoder"], strict=False)
        logger.info(f"Loading specific encoder : {status}")
        status = self.common_encoder.load_state_dict(checkpoint["common_encoder"], strict=False)
        logger.info(f"Loading common encoder : {status}")

    def get_chkpt_name(self, epoch):
        return f"StrongEncoder_exp-{self.exp_name}_epoch-{epoch}.pth"
    
    def get_representation_name(self, epoch):
        return f"Representations_StrongEncoder_exp-{self.exp_name}_epoch-{epoch}.pkl"
    
    def save_representations(self, representations, epoch):
        with open(os.path.join(self.chkpdt_dir, self.get_representation_name(epoch=epoch)), "wb") as f:
            pickle.dump(representations, f)
    
    def save_hyperparameters(self, **kwargs):
        filename = f"StrongEncoder_exp-{self.exp_name}_hyperparameters.json"
        hyperparameters = {"backbone": self.backbone, "n_embeddings": self.latent_dim,
                           "weak_encoder_chkpt": self.weak_encoder_chkpt,
                           "weak_modality": "skeleton", "strong_modality": "vbm",
                           "bactch_size": 8, "lr": 1e-4, 
                           "weight_decay": 5e-5, **kwargs}
        with open(os.path.join(self.chkpdt_dir, filename), "w") as f:
            json.dump(hyperparameters, f)


def parse_args(argv):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description='Train/test strong encoder.')
    parser.add_argument("--backbone", required=True, choices=["resnet18", "densenet121", "alexnet"],
                        help="Architecture of the neural network.")
    parser.add_argument("--latent_dim", required=True, type=int,
                        help="Dimenseion of the latent space.")
    parser.add_argument("--weak_encoder_chkpt", required=True, type=str,
                        help="Path toward weak encoder pretrained model.")
    parser.add_argument("--chkpt_dir", required=True, type=str,
                        help="Path to the folder where results and models are saved.") 
    parser.add_argument("--exp_name", required=True, type=str,
                        help="Experience name.")
    parser.add_argument("--dataset", required=True, type=str, choices=["asd", "bd", "scz"],
                        help="Dataset on which the model is trained.")
    parser.add_argument("--ponderation", type=float, default=10,
                        help="Ponderation of the joint entropy minimisation. Default is: 10.")
    parser.add_argument("--jem_loss_config", type=str, default="specific-common", choices=["specific-common", "specific-weak"],
                        help="Representations between which the jem loss is computed. Default is: specific-common.")
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="Number of training epochs. Default is 50.")
    parser.add_argument("--data_augmentation", type=str, choices=["cutout", "all_tf"],
                        help="Apply data augmentations during training.")
    parser.add_argument("--labels", type=str, nargs="+", choices=["diagnosis", "age", "site", "sex", "tiv"],
                        help="Labels on which the model will be tested.")
    parser.add_argument("--epochs_to_test", type=int, nargs="+", default=49,
                        help="Epoch at which the model will be tested.")
    parser.add_argument("--train", action="store_true",
                        help="Train model")
    parser.add_argument("--test", action="store_true",
                        help="Test the model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate verbosity mode")

    args = parser.parse_args(argv)    
    return args

def main(argv):
    args = parse_args(argv)

    # Create saving directory
    os.makedirs(args.chkpt_dir, exist_ok=True)

    # Setup Logging
    setup_logging(level="debug" if args.verbose else "info",
                  logfile=os.path.join(args.chkpt_dir, f"exp-{args.exp_name}.log"))
    
    logger.info(f"Checkpoint directory : {args.chkpt_dir}")

    model = StrongEncoder(backbone=args.backbone, latent_dim=args.latent_dim,
                          weak_encoder_chkpt=args.weak_encoder_chkpt)

    if args.train:
        model.train(chkpt_dir=args.chkpt_dir, exp_name=args.exp_name, dataset=args.dataset,
                    ponderation=args.ponderation, jem_loss_config=args.jem_loss_config,
                    nb_epochs=args.nb_epochs, data_augmentation=args.data_augmentation)
    if args.test:
        model.test(chkpt_dir=args.chkpt_dir, exp_name=args.exp_name, dataset=args.dataset,
                   labels=args.labels, list_epochs=args.epochs_to_test)

    
if __name__ == "__main__":
    main(sys.argv[1:])
