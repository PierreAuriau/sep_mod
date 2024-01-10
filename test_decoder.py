# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import random
from sklearn.linear_model import LogisticRegression

# project import
from dataset import get_dataloaders, SimCLRDataAugmentation
from encoder import WeakEncoder
from decoder import Decoder, ResNet18Dec
from only_one_encoder import Encoder


# test
def test(decoder, encoder, test_loader, train_loader, n_samples, type_encoder):
    encoder.eval()
    decoder.eval()
    mse = nn.MSELoss()
    randn = np.random.randint(0, len(test_loader))
    with torch.no_grad():
        log_reg, score = test_linear_probe(encoder=encoder, train_loader=train_loader, 
                                           test_loader=test_loader, type_encoder=type_encoder)
        loss = 0
        for n, (weak_img, strong_img, _, _) in enumerate(test_loader):
            if type_encoder == "double":
                common_rep, _, _, _ = encoder(strong_img.cuda())
            elif type_encoder == "simple":
                common_rep, _ = encoder(strong_img.cuda())
            output = decoder(common_rep.cuda())
            loss += mse(weak_img.cuda(), output).item()
            if n == randn:
                rep = common_rep.detach().cpu().numpy()
                inpt = strong_img.detach().cpu().numpy()
                ground_truth = weak_img.detach().cpu().numpy()
                outpt = output.detach().cpu().numpy()
                randi = np.random.randint(0, (inpt.shape[0]-n_samples))
                img_inpt = np.transpose(inpt[randi:(randi+n_samples), :, :, :], (0, 2, 3, 1))
                img_ground_truth = np.transpose(ground_truth[randi:(randi+n_samples), :, :, :], (0, 2, 3, 1))
                img_outpt = np.transpose(outpt[randi:(randi+n_samples), :, :, :], (0, 2, 3, 1))
                pred = log_reg.predict(rep[randi:(randi+n_samples), :]) #predict_proba

    print("MSE on test set : ", loss)
    print('-----')
    dico2save = {
        "score": score,
        "loss": loss,
        "inputs": img_inpt,
        "ground_truth": img_ground_truth,
        "outputs": img_outpt,
        "predictions": pred
    }

    return dico2save


# test linear probes
def test_linear_probe(encoder, train_loader, test_loader, type_encoder):
    encoder.eval()
    with torch.no_grad():
        X_train_strong_common = []
        X_test_strong_common = []
        y_weak_train = []
        y_weak_test = []
        y_strong_train = []
        y_strong_test = []
        for _, strong_view, weak_label, strong_label in train_loader:
            if type_encoder == "double":
                common_strong_rep, _, _, _ = encoder(strong_view.cuda())
            elif type_encoder == "simple":
                common_strong_rep, _ = encoder(strong_view.cuda())
            X_train_strong_common.extend(common_strong_rep.cpu().numpy())
            y_weak_train.extend(weak_label.cpu().numpy())
            y_strong_train.extend(strong_label.cpu().numpy())
        for _, strong_view, weak_label, strong_label in test_loader:
            if type_encoder == "double":
                common_strong_rep, _, _, _ = encoder(strong_view.cuda())
            elif type_encoder == "simple":
                common_strong_rep, _ = encoder(strong_view.cuda())
            X_test_strong_common.extend(common_strong_rep.cpu().numpy())
            y_weak_test.extend(weak_label.cpu().numpy())
            y_strong_test.extend(strong_label.cpu().numpy())
        X_train_strong_common = np.array(X_train_strong_common)
        X_test_strong_common = np.array(X_test_strong_common)
        y_weak_train = np.array(y_weak_train)
        y_weak_test = np.array(y_weak_test)
        y_strong_train = np.array(y_strong_train)
        y_strong_test = np.array(y_strong_test)

    #scores = {"label": [], "accuracy": []}
    #log_reg = LogisticRegression().fit(X_train_strong_common, y_strong_train)
    #log_reg_score = log_reg.score(X_test_strong_common, y_strong_test)
    #print("strong common evaluated on strong labels (low) : ", log_reg_score)
    #scores["label"].append("strong")
    #scores["accuracy"].append(log_reg_score)

    log_reg = LogisticRegression().fit(X_train_strong_common, y_weak_train)
    log_reg_score = log_reg.score(X_test_strong_common, y_weak_test)
    print("strong common evaluated on weak labels (high) : ", log_reg_score)
    #scores["label"].append("weak")
    #scores["accuracy"].append(log_reg_score)

    return log_reg, log_reg_score


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Train weak encoder')
    parser.add_argument(
        "-m", "--weak_modality", type=str, required=True, choices=["mnist", "cifar"],
        help="Weak modality, can be either 'cifar' or 'mnist'")
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, required=True,
        help="Directory where the encoder and the decoder are saved")
    parser.add_argument(
        "-t", "--type_encoder", type=str, default="double", choices=["simple", "double"],
        help="One or two encoder(s). Default is : 'double'.")
    parser.add_argument(
        "-n", "--n_samples", type=int, default=8,
        help="Number of samples to save.")
    parser.add_argument(
        "-d", "--decoder", type=str, default="simple", choices=["simple", "resnet18"],
        help="Architecture of the decoder. Default is: 'simple'.")
    args = parser.parse_args(argv)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print("Checkpoint directory created.")
    return args


def main(argv):
    simplefilter("ignore", category=ConvergenceWarning)
    #random.seed(0)

    args = parse_args(argv)

    # hyper-parameter
    weak_modality = args.weak_modality
    strong_modality = {"mnist": "cifar", "cifar": "mnist"}[weak_modality]

    # Instantiate dataset and dataloader
    train_loader, test_loader = get_dataloaders(weak_modality=weak_modality)
    
    # build model
    decoder = torch.load(os.path.join(args.checkpoint_dir, f"decoder_{args.decoder}_{weak_modality}.pth"))
    if args.type_encoder == "double":
        encoder = torch.load(os.path.join(args.checkpoint_dir, f"sep_mod_strong_for_{strong_modality}_recognition.pth"))
    elif args.type_encoder == "simple":
        encoder = torch.load(os.path.join(args.checkpoint_dir, f"sep_mod_weak_for_{weak_modality}_recognition.pth"))
    
    dico2save = test(decoder=decoder, encoder=encoder, 
                     train_loader=train_loader, test_loader=test_loader, 
                     n_samples=args.n_samples, type_encoder=args.type_encoder)
    
    # save results
    print("Saving results...")

    with open(os.path.join(args.checkpoint_dir, f"decoder_{args.decoder}_{weak_modality}_test.pkl"), "wb") as f:
        pickle.dump(dico2save, f)

if __name__ == "__main__":
    main(argv=sys.argv[1:])