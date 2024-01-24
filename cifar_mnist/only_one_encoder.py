# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import random
# project import
from dataset import get_dataloaders, SimCLRDataAugmentation
from loss import align_loss, uniform_loss, joint_entropy_loss, norm

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        # encoder
        self.encoder = resnet18(pretrained=False)
        self.feature_dim = 512

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Linear(self.feature_dim, latent_dim)

        # Add MLP projection.
        self.projector = nn.Sequential(nn.Linear(latent_dim, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(),
                                            nn.Linear(128, latent_dim))

    def forward(self, x):
        representation = self.encoder(x)

        head = self.projector(representation)

        return representation, head

def train(weak_encoder, strong_encoder, train_loader, test_loader, n_epochs, pond):
    # define optimizer
    optimizer = Adam(list(weak_encoder.parameters()) + list(strong_encoder.parameters()), lr=3e-4)
    # data augmentation
    data_aug_pipeline = SimCLRDataAugmentation()

    weak_encoder = weak_encoder.cuda()
    strong_encoder = strong_encoder.cuda()

    dico_pred = {"epoch": [], "encoder": [], "modality": [], "label": [], "accuracy": []}
    dico_loss = {"epoch": [], "loss": [], "value":[]}

    # Train model
    print("epoch : 0")
    scores = test_linear_probe(weak_encoder, strong_encoder, train_loader, test_loader)
    dico_pred["epoch"] += [0 for _ in range(len(scores["accuracy"]))]
    for k, v in scores.items():
        dico_pred[k] += v
    for epoch in range(1, n_epochs):
        weak_encoder.train()
        strong_encoder.train()

        train_loss = 0
        for batch_idx, (weak_view, strong_view, _, _) in enumerate(train_loader):
            with torch.no_grad():
                weak_view1 = data_aug_pipeline(weak_view)
                weak_view2 = data_aug_pipeline(weak_view)
                strong_view1 = data_aug_pipeline(strong_view)
                strong_view2 = data_aug_pipeline(strong_view)

            # weak_head
            _, weak_head_1 = weak_encoder(weak_view1.cuda())
            _, weak_head_2 = weak_encoder(weak_view2.cuda())

            # strong head
            _, strong_head_1 = strong_encoder(strong_view1.cuda())
            _, strong_head_2 = strong_encoder(strong_view2.cuda())

            # common head
            _, common_strong_head_1 = weak_encoder(strong_view1.cuda())
            _, common_strong_head_2 = weak_encoder(strong_view2.cuda())           

            # weak loss
            weak_align_loss = align_loss(norm(weak_head_1), norm(weak_head_2))
            weak_uniform_loss = (uniform_loss(norm(weak_head_2)) + uniform_loss(norm(weak_head_1))) / 2.0
            weak_loss = weak_align_loss + weak_uniform_loss

            # common strong to weak
            common_strong_align_loss = align_loss(norm(weak_head_1.detach()), norm(common_strong_head_1))
            common_strong_align_loss +=  align_loss(norm(weak_head_2.detach()), norm(common_strong_head_2))
            common_strong_align_loss /= 2.0
            common_strong_uniform_loss = (uniform_loss(norm(common_strong_head_2)) + uniform_loss(norm(common_strong_head_1))) / 2.0
            common_strong_loss = common_strong_align_loss + common_strong_uniform_loss

            # strong loss
            strong_align_loss = align_loss(norm(strong_head_1), norm(strong_head_2))
            strong_uniform_loss = (uniform_loss(norm(strong_head_2)) + uniform_loss(norm(strong_head_1))) / 2.0
            strong_loss = strong_align_loss + strong_uniform_loss

            # mi minimization loss
            jem_loss = joint_entropy_loss(norm(strong_head_1), norm(common_strong_head_1.detach()))
            jem_loss += joint_entropy_loss(norm(strong_head_2), norm(common_strong_head_2.detach()))
            jem_loss = jem_loss / 2.0

            loss = strong_loss + common_strong_loss + weak_loss + pond*jem_loss

            dico_loss["epoch"].append(epoch)
            dico_loss["loss"].append("weak_align")
            dico_loss["value"].append(weak_align_loss.item())

            dico_loss["epoch"].append(epoch)
            dico_loss["loss"].append("weak_uniform")
            dico_loss["value"].append(weak_uniform_loss.item())

            dico_loss["epoch"].append(epoch)
            dico_loss["loss"].append("strong_align")
            dico_loss["value"].append(strong_align_loss.item())

            dico_loss["epoch"].append(epoch)
            dico_loss["loss"].append("strong_uniform")
            dico_loss["value"].append(strong_uniform_loss.item())

            dico_loss["epoch"].append(epoch)
            dico_loss["loss"].append("common_align")
            dico_loss["value"].append(common_strong_align_loss.item())

            dico_loss["epoch"].append(epoch)
            dico_loss["loss"].append("common_uniform")
            dico_loss["value"].append(common_strong_uniform_loss.item())

            dico_loss["epoch"].append(epoch)
            dico_loss["loss"].append("jem")
            dico_loss["value"].append(jem_loss.item())

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            print("epoch : ", epoch)
            print("loss :", train_loss)
            scores = test_linear_probe(weak_encoder, strong_encoder, train_loader, test_loader)
            dico_pred["epoch"] += [epoch for _ in range(len(scores["accuracy"]))]
            for k, v in scores.items():
                dico_pred[k] += v

    return dico_loss, dico_pred


# test linear probes
def test_linear_probe(weak_encoder, strong_encoder, train_loader, test_loader):
    weak_encoder.eval()
    strong_encoder.eval()
    with torch.no_grad():
        X_train_strong = []
        X_test_strong = []
        X_train_strong_common = []
        X_test_strong_common = []
        X_train_weak = []
        X_test_weak = []
        y_weak_train = []
        y_weak_test = []
        y_strong_train = []
        y_strong_test = []
        for weak_view, strong_view, weak_label, strong_label in train_loader:
            weak_rep, _ = weak_encoder(weak_view.cuda())
            common_strong_rep, _ = weak_encoder(strong_view.cuda())
            strong_rep, _ = strong_encoder(strong_view.cuda())
            X_train_strong.extend(strong_rep.cpu().numpy())
            X_train_strong_common.extend(common_strong_rep.cpu().numpy())
            X_train_weak.extend(weak_rep.cpu().numpy())
            y_weak_train.extend(weak_label.cpu().numpy())
            y_strong_train.extend(strong_label.cpu().numpy())
        for weak_view, strong_view, weak_label, strong_label in test_loader:
            weak_rep, _ = weak_encoder(weak_view.cuda())
            common_strong_rep, _ = weak_encoder(strong_view.cuda()) 
            strong_rep, _ = strong_encoder(strong_view.cuda())
            X_test_strong.extend(strong_rep.cpu().numpy())
            X_test_strong_common.extend(common_strong_rep.cpu().numpy())
            X_test_weak.extend(weak_rep.cpu().numpy())
            y_weak_test.extend(weak_label.cpu().numpy())
            y_strong_test.extend(strong_label.cpu().numpy())
        X_train_strong = np.array(X_train_strong)
        X_test_strong = np.array(X_test_strong)
        X_train_strong_common = np.array(X_train_strong_common)
        X_test_strong_common = np.array(X_test_strong_common)
        X_train_weak = np.array(X_train_weak)
        X_test_weak = np.array(X_test_weak)
        y_weak_train = np.array(y_weak_train)
        y_weak_test = np.array(y_weak_test)
        y_strong_train = np.array(y_strong_train)
        y_strong_test = np.array(y_strong_test)

    scores = {"encoder": [], "modality":[], "label": [], "accuracy": []}
    log_reg = LogisticRegression().fit(X_train_strong, y_strong_train)
    log_reg_score = log_reg.score(X_test_strong, y_strong_test)
    print("strong evaluated on strong labels (high) : ", log_reg_score)
    scores["encoder"].append("strong")
    scores["modality"].append("strong")
    scores["label"].append("strong")
    scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_strong, y_weak_train)
    log_reg_score = log_reg.score(X_test_strong, y_weak_test)
    print("strong evaluated on weak labels (low) : ", log_reg_score)
    scores["encoder"].append("strong")
    scores["modality"].append("strong")
    scores["label"].append("weak")
    scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_strong_common, y_strong_train)
    log_reg_score = log_reg.score(X_test_strong_common, y_strong_test)
    print("weak evaluated on strong labels (low) : ", log_reg_score)
    scores["encoder"].append("weak")
    scores["modality"].append("strong")
    scores["label"].append("strong")
    scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_strong_common, y_weak_train)
    log_reg_score = log_reg.score(X_test_strong_common, y_weak_test)
    print("weak evaluated on weak labels (high) : ", log_reg_score)
    scores["encoder"].append("weak")
    scores["modality"].append("strong")
    scores["label"].append("weak")
    scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_weak, y_weak_train)
    log_reg_score = log_reg.score(X_test_weak, y_weak_test)
    print("weak evaluated on weak labels (high) - weak modality: ", log_reg_score)
    scores["encoder"].append("weak")
    scores["modality"].append("weak")
    scores["label"].append("weak")
    scores["accuracy"].append(log_reg_score)
    print('-----')

    return scores


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Train strong encoder')
    parser.add_argument(
        "-m", "--weak_modality", type=str, required=True, choices=["mnist", "cifar"],
        help="Weak modality, can be either 'cifar' or 'mnist'")
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, required=True,
        help="Directory where models are saved")
    parser.add_argument(
        "-p", "--ponderation", type=float, default=100.0,
        help="Ponderation of the jem loss.")
    parser.add_argument(
        "-s", "--latent_size", type=int, default=32,
        help="Latent space size of the encoders. Default is 32.")
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=251,
        help="Number of epochs for training. Default is 251.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.checkpoint_dir):
        raise NotADirectoryError("Checkpoint directory is not found.")
    return args


def main(argv):
    simplefilter("ignore", category=ConvergenceWarning)
    random.seed(0)

    args = parse_args(argv)
    # hyper-parameter
    weak_modality = args.weak_modality
    latent_size = args.latent_size
    n_epochs = args.n_epochs
    pond = args.ponderation

    # Instantiate Dataset and Data Loader
    train_loader, test_loader = get_dataloaders(weak_modality=weak_modality)
    
    # build model
    weak_encoder = Encoder(latent_dim=latent_size).float()
    strong_encoder = Encoder(latent_dim=latent_size).float()

    # train model
    dico_loss, dico_pred = train(weak_encoder, strong_encoder, train_loader, test_loader, n_epochs, pond)

    # save model
    print("Saving model...")
    torch.save(weak_encoder, os.path.join(args.checkpoint_dir,
                                        f"sep_mod_weak_for_{weak_modality}_recognition.pth"))
    recognition_modality = {"cifar": "mnist", "mnist": "cifar"}[weak_modality]
    torch.save(strong_encoder, os.path.join(args.checkpoint_dir,
                                            f"sep_mod_strong_for_{recognition_modality}_recognition.pth"))

    # save results
    print("Saving results...")
    df = pd.DataFrame(dico_loss)
    df.to_csv(os.path.join(args.checkpoint_dir, "loss.csv"), index=False)
    df = pd.DataFrame(dico_pred)
    df.to_csv(os.path.join(args.checkpoint_dir, "accuracies.csv"), index=False)

    print("--------\nREMINDER\n--------")
    print("WEAK MODALITY : ", weak_modality)
    print("STRONG MODALITY : ", recognition_modality)

if __name__ == "__main__":
    main(argv=sys.argv[1:])
