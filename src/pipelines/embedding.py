"""This file contains the embedding pipeline using a variational autoencoder (VAE) for dimensionality reduction of single-cell RNA-seq data."""

from scipy.io import mmread
from scipy import sparse
import anndata as ad
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import yaml

with open("config/embedding.yml", "r") as f:
    embedding_config = yaml.safe_load(f)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):

        # a continuar
