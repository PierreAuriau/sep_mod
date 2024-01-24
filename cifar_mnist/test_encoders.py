# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import random
# project import
from dataset import get_dataloaders, SimCLRDataAugmentation
from encoder import StrongEncoder, WeakEncoder


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
            common_strong_rep, _, strong_rep, _ = strong_encoder(strong_view.cuda())
            X_train_strong.extend(strong_rep.cpu().numpy())
            X_train_strong_common.extend(common_strong_rep.cpu().numpy())
            X_train_weak.extend(weak_rep.cpu().numpy())
            y_weak_train.extend(weak_label.cpu().numpy())
            y_strong_train.extend(strong_label.cpu().numpy())
        for weak_view, strong_view, weak_label, strong_label in test_loader:
            weak_rep, _ = weak_encoder(weak_view.cuda())
            common_strong_rep, _, strong_rep, _ = strong_encoder(strong_view.cuda())
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

    scores = {"encoder": [], "label": [], "accuracy": []}
    log_reg = LogisticRegression().fit(X_train_strong, y_strong_train)
    log_reg_score = log_reg.score(X_test_strong, y_strong_test)
    print("strong evaluated on strong labels (high) : ", log_reg_score)
    scores["encoder"].append("strong")
    scores["label"].append("strong")
    scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_strong, y_weak_train)
    log_reg_score = log_reg.score(X_test_strong, y_weak_test)
    print("strong evaluated on weak labels (low) : ", log_reg_score)
    scores["encoder"].append("strong")
    scores["label"].append("weak")
    scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_strong_common, y_strong_train)
    log_reg_score = log_reg.score(X_test_strong_common, y_strong_test)
    print("strong common evaluated on strong labels (low) : ", log_reg_score)
    scores["encoder"].append("common")
    scores["label"].append("strong")
    scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_strong_common, y_weak_train)
    log_reg_score = log_reg.score(X_test_strong_common, y_weak_test)
    print("strong common evaluated on weak labels (high) : ", log_reg_score)
    scores["encoder"].append("common")
    scores["label"].append("weak")
    scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_weak, y_weak_train)
    log_reg_score = log_reg.score(X_test_weak, y_weak_test)
    print("weak evaluated on weak labels (high) : ", log_reg_score)
    scores["encoder"].append("weak")
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
    recognition_modality = {"cifar": "mnist", "mnist": "cifar"}[weak_modality]

    # Instantiate Dataset and Data Loader
    train_loader, test_loader = get_dataloaders(weak_modality=weak_modality)
    
    # build model
    weak_encoder = torch.load(os.path.join(args.checkpoint_dir,
                                           f"sep_mod_weak_{weak_modality}.pth"))  # WeakEncoder(weak_dim=weak_size).float().cuda()
    strong_encoder = torch.load(os.path.join(args.checkpoint_dir,
                                           f"sep_mod_strong_for_{recognition_modality}_recognition.pth"))
    
    scores = test_linear_probe(weak_encoder, strong_encoder, train_loader, test_loader)
    # save results
    print("Saving results...")
    df = pd.DataFrame(scores)
    df.to_csv(os.path.join(args.checkpoint_dir, "accuracies.csv"), index=False)
    print("\n--------\nREMINDER\n--------\n")
    print("WEAK MODALITY : ", weak_modality)
    print("STRONG MODALITY : ", recognition_modality)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
