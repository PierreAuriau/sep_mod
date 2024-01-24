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
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
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

def train_supervised(classifier, train_loader, test_loader, n_epochs, data_aug=False):
    # define optimizer
    optimizer = Adam(list(classifier.parameters()), lr=3e-4)
    # loss
    loss_fn = nn.CrossEntropyLoss()
    # data augmentation
    if data_aug:
        data_aug_pipeline = SimCLRDataAugmentation()
    
    to_save = {"epoch": [], "train_loss": [], "test_loss": [], "accuracy": []}
    for epoch in range(1, n_epochs):
        classifier.train()
        train_loss = 0
        for _, (_, strong_view, weak_label, _) in enumerate(train_loader):
            if data_aug:
                with torch.no_grad():
                    strong_view = data_aug_pipeline(strong_view)

            # weak_head
            output = classifier(strong_view.cuda())

            # weak loss
            loss = loss_fn(output, weak_label.cuda())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

            
        to_save["epoch"].append(epoch)
        to_save["train_loss"].append(train_loss)
        if epoch % 10 == 0:
            test_loss, accuracy = test_supervised(classifier, test_loader)
            to_save["test_loss"].append(test_loss)
            to_save["accuracy"].append(accuracy)
        else:
            to_save["test_loss"].append(None)
            to_save["accuracy"].append(None)
        
        """
        if epoch % 50 == 0:
            pck = {}
            classifier.eval()
            with torch.no_grad():
                for _, (_, strong_view, weak_label, _) in enumerate(train_loader):
                    output = classifier(strong_view.cuda())
                    img = strong_view.detach().cpu().numpy()
                    y_true = weak_label.detach().cpu().numpy()
                    y_pred = output.detach().cpu().numpy()
                    break
                pck["train"] = {"inputs": img, "y_true": y_true, "y_pred": y_pred}
                for _, (_, strong_view, weak_label, _) in enumerate(test_loader):
                    output = classifier(strong_view.cuda())
                    img = strong_view.detach().cpu().numpy()
                    y_true = weak_label.detach().cpu().numpy()
                    y_pred = output.detach().cpu().numpy()
                    break
                pck["test"] = {"inputs": img, "y_true": y_true, "y_pred": y_pred}
                with open(f"/neurospin/dico/pauriau/data/exp_sepmod/baselines/inputs_outputs_{epoch}.pkl", "wb") as f:
                    pickle.dump(pck, f)
        """
    return to_save

def test_supervised(classifier, test_loader):
    loss_fn = nn.CrossEntropyLoss()
    classifier.eval()
    with torch.no_grad():
        test_loss = 0
        y_pred, y_true = [], []
        for _, (_, strong_view, weak_label, _) in enumerate(test_loader):
            output = classifier(strong_view.cuda())
            test_loss += loss_fn(output, weak_label.cuda()).item()
            y_pred.extend(torch.sigmoid(output).detach().cpu().numpy())
            y_true.extend(weak_label.detach().cpu().numpy())
        y_pred = np.asarray(y_pred).argmax(axis=1)
        y_true = np.asarray(y_true)
        bacc = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
    return test_loss, bacc


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
        "-s", "--supervised", action="store_true",
        help="If set, train a supervised classifer.")
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

    save_train = False
    save_test = False

    if args.supervised:
        # build model
        classifier = resnet18(num_classes=10, weights=None).float().cuda()

        # train_model
        dico = train_supervised(classifier, train_loader, test_loader, n_epochs)

        # save model
        print("Saving model...")
        torch.save(classifier, os.path.join(args.checkpoint_dir,
                                            f"supervised_for_{weak_modality}_recognition.pth"))
        # save predictions
        print("Saving results...")
        df = pd.DataFrame(dico)
        df.to_csv(os.path.join(args.checkpoint_dir, f"supervised_{weak_modality}_loss_accuracies.csv"), index=False)
        
    else:
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
