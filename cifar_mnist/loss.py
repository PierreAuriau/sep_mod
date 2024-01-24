# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import function
import torch


def norm(x):
    return torch.nn.functional.normalize(x, dim=1, p=2)


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=1):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def joint_entropy_loss(w, s, tau=0.5):
    s_similarity_matrix = torch.cdist(s, s, p=2.0).pow(2).mul(-1/(2*tau)).exp()
    c_similarity_matrix = torch.cdist(w, w, p=2.0).pow(2).mul(-1/(2*tau)).exp()
    jem_loss = (s_similarity_matrix * c_similarity_matrix).mean(-1).log().mean()
    return jem_loss

