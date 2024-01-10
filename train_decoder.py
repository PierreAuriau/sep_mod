# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import argparse
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
# project import
from dataset import get_dataloaders, SimCLRDataAugmentation
from encoder import WeakEncoder
from decoder import Decoder, ResNet18Dec
from only_one_encoder import Encoder

# train decoder
def train(decoder, weak_encoder, train_loader, test_loader, n_epochs):
    decoder = decoder.cuda()
    weak_encoder = weak_encoder.cuda()
    # define optimizer
    optimizer = Adam(list(decoder.parameters()), lr=3e-4)
    # define loss
    mse = nn.MSELoss()
    
    # data augmentation
    #data_aug_pipeline = SimCLRDataAugmentation()

    # train model
    loss_to_save = {"epoch": [], "set": [], "value": []}
    fig_to_save = {}
    print("epoch : 0")
    test_loss, fig = test(decoder, weak_encoder, test_loader)
    loss_to_save["epoch"].append(0)
    loss_to_save["set"].append("test")
    loss_to_save["value"].append(test_loss)
    fig_to_save[0] = fig
    for epoch in range(1, n_epochs):
        decoder.train()
        weak_encoder.eval()
        train_loss = 0
        for batch_idx, (weak_view, _, _, _) in enumerate(train_loader):
            #with torch.no_grad():
            #    weak_view1 = data_aug_pipeline(weak_view)
            #    weak_view2 = data_aug_pipeline(weak_view)

            # weak_head
            weak_head, _ = weak_encoder(weak_view.cuda())
            # decoder
            output = decoder(weak_head)
            # weak loss
            loss = mse(weak_view.cuda(), output)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        loss_to_save["epoch"].append(epoch)
        loss_to_save["set"].append("train") 
        loss_to_save["value"].append(train_loss)
        if epoch % 10 == 0:
            print("epoch : ", epoch)
            test_loss, fig = test(decoder, weak_encoder, test_loader)
            loss_to_save["epoch"].append(epoch)
            loss_to_save["set"].append("test")
            loss_to_save["value"].append(test_loss)
            if epoch % 50 == 0:
                fig_to_save[epoch] = fig
    return loss_to_save, fig_to_save


# test
def test(decoder, weak_encoder, test_loader):
    n_samples = 8
    weak_encoder.eval()
    decoder.eval()
    mse = nn.MSELoss()
    randn = np.random.randint(0, len(test_loader))
    with torch.no_grad():
        loss = 0
        for n, (weak_view, _, _, _) in enumerate(test_loader):
            weak_rep, _ = weak_encoder(weak_view.cuda())
            output = decoder(weak_rep.cuda())
            loss += mse(weak_view.cuda(), output).item()
            if n == randn:
                fig= np.zeros((n_samples, 2, 32, 32, 3))
                inpt = weak_view.detach().cpu().numpy()
                outpt = output.detach().cpu().numpy()
                randi = np.random.randint(0, (inpt.shape[0]-n_samples))
                fig[:, 0, :, :, :] = np.transpose(inpt[randi:(randi+n_samples), :, :, :], (0, 2, 3, 1))
                fig[:, 1, :, :, :] = np.transpose(outpt[randi:(randi+n_samples), :, :, :], (0, 2, 3, 1))

    print("MSE on test set : ", loss)
    print('-----')

    return loss, fig

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
    parser.add_argument(
        "-d", "--decoder", type=str, default="simple", choices=["simple", "resnet18"],
        help="Archiecture of the decoder. Default is: 'simple'.")
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
    if args.decoder == "resnet18":
        decoder = ResNet18Dec(z_dim=weak_size)
    else:
        decoder = Decoder(input_shape=[32, 32], latent_dim=weak_size)
    weak_encoder = torch.load(os.path.join(args.checkpoint_dir,
                                           f"sep_mod_weak_{weak_modality}.pth"))
    # train model
    loss, fig = train(decoder, weak_encoder, train_loader, test_loader, n_epochs)

    # save model
    print("Saving model...")
    torch.save(decoder, os.path.join(args.checkpoint_dir,
                                          f"decoder_{args.decoder}_{weak_modality}.pth"))
    # saving results
    print("Saving results...")
    df = pd.DataFrame(loss)
    df.to_csv(os.path.join(args.checkpoint_dir, f"decoder_{args.decoder}_{weak_modality}_mse_loss.csv"), index=False)

    for epoch, f in fig.items():
        np.save(os.path.join(args.checkpoint_dir, f"decoder_{args.decoder}_{weak_modality}_epoch_{epoch}_inputs_outputs.npy"),
                f)

if __name__ == "__main__":
    main(argv=sys.argv[1:])