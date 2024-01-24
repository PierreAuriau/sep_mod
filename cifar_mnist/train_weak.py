# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import argparse
import numpy as np
import torch
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import random
# project import
from dataset import get_dataloaders, SimCLRDataAugmentation
from loss import align_loss, uniform_loss, norm
from encoder import WeakEncoder


# train encoder
def train(weak_encoder, train_loader, test_loader, n_epochs):
    # define optimizer
    optimizer = Adam(list(weak_encoder.parameters()), lr=3e-4)
    # data augmentation
    data_aug_pipeline = SimCLRDataAugmentation()

    # train model
    print("epoch : 0")
    test_linear_probe(weak_encoder, train_loader, test_loader)
    for epoch in range(1, n_epochs):
        weak_encoder.train()
        train_loss = 0
        for batch_idx, (weak_view, _, _, _) in enumerate(train_loader):
            with torch.no_grad():
                weak_view1 = data_aug_pipeline(weak_view)
                weak_view2 = data_aug_pipeline(weak_view)

            # weak_head
            _, weak_head_1 = weak_encoder(weak_view1.cuda())
            _, weak_head_2 = weak_encoder(weak_view2.cuda())

            # weak loss
            weak_align_loss = align_loss(norm(weak_head_1), norm(weak_head_2))
            weak_uniform_loss = (uniform_loss(norm(weak_head_2)) + uniform_loss(norm(weak_head_1))) / 2.0
            weak_loss = weak_align_loss + weak_uniform_loss

            loss = weak_loss

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            print("epoch : ", epoch)
            test_linear_probe(weak_encoder, train_loader, test_loader)


# test linear probes
def test_linear_probe(weak_encoder, train_loader, test_loader):
    weak_encoder.eval()
    with torch.no_grad():
        X_train_weak = []
        X_test_weak = []
        y_weak_train = []
        y_weak_test = []
        for weak_view, _, weak_label, _ in train_loader:
            weak_rep, _ = weak_encoder(weak_view.cuda())
            X_train_weak.extend(weak_rep.cpu().numpy())
            y_weak_train.extend(weak_label.cpu().numpy())
        for weak_view, _, weak_label, _ in test_loader:
            weak_rep, _ = weak_encoder(weak_view.cuda())
            X_test_weak.extend(weak_rep.cpu().numpy())
            y_weak_test.extend(weak_label.cpu().numpy())
        X_train_weak = np.array(X_train_weak)
        X_test_weak = np.array(X_test_weak)
        y_weak_train = np.array(y_weak_train)
        y_weak_test = np.array(y_weak_test)

    log_reg = LogisticRegression().fit(X_train_weak, y_weak_train)
    log_reg_score = log_reg.score(X_test_weak, y_weak_test)
    print("weak trained on weak labels (high) : ", log_reg_score)
    print('-----')


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Train weak encoder')
    parser.add_argument(
        "-m", "--weak_modality", type=str, required=True, choices=["mnist", "cifar"],
        help="Weak modality, can be either 'cifar' or 'mnist'")
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, required=True,
        help="Directory where models are saved")
    parser.add_argument(
        "-w", "--weak_size", type=int, default=32,
        help="Latent space size of the weak encoder. Default is 32.")
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=251,
        help="Number of epochs (+1) for training. Default is 251.")
    args = parser.parse_args(argv)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print("Checkpoint directory created.")
    return args


def main(argv):
    simplefilter("ignore", category=ConvergenceWarning)
    random.seed(0)

    args = parse_args(argv)

    # hyper-parameter
    weak_modality = args.weak_modality
    weak_size = args.weak_size
    n_epochs = args.n_epochs

    # Instantiate dataset and dataloader
    train_loader, test_loader = get_dataloaders(weak_modality=weak_modality)

    # build model
    weak_encoder = WeakEncoder(weak_dim=weak_size).float().cuda()

    # train model
    train(weak_encoder, train_loader, test_loader, n_epochs)

    # save model
    print("Saving model...")
    torch.save(weak_encoder, os.path.join(args.checkpoint_dir,
                                          f"sep_mod_weak_{weak_modality}.pth"))


if __name__ == "__main__":
    main(argv=sys.argv[1:])
