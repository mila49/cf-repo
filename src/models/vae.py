import torch
from torch import nn


class VAE(nn.Module):
    """
    Variational autoencoder for dimensionality reduction of single-cell data.

    Dropout is configurable so it can be included in the hyperparameter sweep.
    The latent mean (mu) is used as the deterministic embedding for clustering.
    """

    def __init__(
        self,
        input_dim,
        latent_dim=16,
        dropout_rate=0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z


def vae_loss(reconstruction, x, mu, logvar, kl_weight=0.001):
    reconstruction_loss = nn.functional.mse_loss(
        reconstruction,
        x,
        reduction="mean",
    )

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return reconstruction_loss + kl_weight * kl_loss