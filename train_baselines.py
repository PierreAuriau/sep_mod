# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import argparse
import numpy as np
import pandas as pd
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
    predictions = {"epoch": [], "label": [], "accuracy": []}
    print("epoch : 0")
    scores = test_linear_probe(weak_encoder, train_loader, test_loader)
    predictions["epoch"] += [0 for _ in range(len(scores["accuracy"]))]
    for k, v in scores.items():
        predictions[k] += v
    for epoch in range(1, n_epochs):
        weak_encoder.train()
        train_loss = 0
        for _, (_, strong_view, _, _) in enumerate(train_loader):
            with torch.no_grad():
                strong_view1 = data_aug_pipeline(strong_view)
                strong_view2 = data_aug_pipeline(strong_view)

            # weak_head
            _, strong_head_1 = weak_encoder(strong_view1.cuda())
            _, strong_head_2 = weak_encoder(strong_view2.cuda())

            # weak loss
            weak_align_loss = align_loss(norm(strong_head_1), norm(strong_head_2))
            weak_uniform_loss = (uniform_loss(norm(strong_head_2)) + uniform_loss(norm(strong_head_1))) / 2.0
            weak_loss = weak_align_loss + weak_uniform_loss

            loss = weak_loss

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            print("epoch : ", epoch)
            scores = test_linear_probe(weak_encoder, train_loader, test_loader)
            predictions["epoch"] += [epoch for _ in range(len(scores["accuracy"]))]
            for k, v in scores.items():
                predictions[k] += v
    return predictions

# test linear probes
def test_linear_probe(weak_encoder, train_loader, test_loader):
    weak_encoder.eval()
    with torch.no_grad():
        X_train_strong = []
        X_test_strong = []
        y_weak_train = []
        y_weak_test = []
        y_strong_train = []
        y_strong_test = []
        for _, strong_view, weak_label, strong_label in train_loader:
            strong_rep, _ = weak_encoder(strong_view.cuda())
            X_train_strong.extend(strong_rep.cpu().numpy())
            y_weak_train.extend(weak_label.cpu().numpy())
            y_strong_train.extend(strong_label.cpu().numpy())
        for _, strong_view, weak_label, strong_label in test_loader:
            strong_rep, _ = weak_encoder(strong_view.cuda())
            X_test_strong.extend(strong_rep.cpu().numpy())
            y_weak_test.extend(weak_label.cpu().numpy())
            y_strong_test.extend(strong_label.cpu().numpy())
        X_train_strong = np.array(X_train_strong)
        X_test_strong = np.array(X_test_strong)
        y_weak_train = np.array(y_weak_train)
        y_weak_test = np.array(y_weak_test)
        y_strong_train = np.array(y_strong_train)
        y_strong_test = np.array(y_strong_test)

    
    scores = {"label": [], "accuracy": []}

    log_reg = LogisticRegression().fit(X_train_strong, y_weak_train)
    log_reg_score = log_reg.score(X_test_strong, y_weak_test)
    print("encoder trained on strong modality evaluated on weak labels : ", log_reg_score)
    scores["label"].append("weak")
    scores["accuracy"].append(log_reg_score)
    log_reg = LogisticRegression().fit(X_train_strong, y_strong_train)
    log_reg_score = log_reg.score(X_test_strong, y_strong_test)
    print("encoder trained on strong modality evaluated on strong labels : ", log_reg_score)
    scores["label"].append("strong")
    scores["accuracy"].append(log_reg_score)
    print('-----')

    return scores


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
    predictions = train(weak_encoder, train_loader, test_loader, n_epochs)

    # save model
    print("Saving model...")
    torch.save(weak_encoder, os.path.join(args.checkpoint_dir,
                                          f"simclr_big_space_trained_on_strong_modality.pth"))
    # save predictions
    print("Saving results...")
    strong_labels = {"cifar": "mnist", "mnist": "cifar"}[weak_modality]
    df = pd.DataFrame(predictions)
    df = df.replace({"weak": weak_modality, "strong": strong_labels})
    df.to_csv(os.path.join(args.checkpoint_dir, "simclr_big_space_trained_on_strong_modality_accuracies.csv"), index=False)
    print("\n--------\nREMINDER\n--------\n")
    print("WEAK MODALITY : ", weak_modality)

if __name__ == "__main__":
    main(argv=sys.argv[1:])
