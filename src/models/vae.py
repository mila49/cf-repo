import torch
from torch import nn


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) implementation for dimensionality reduction of single-cell data.
    The architecture consists of an encoder that maps input data to a latent space, and a decoder that
    reconstructs the input from the latent representation.
    The loss function combines a reconstruction loss (MSE) and a KL divergence term to regularize the latent space.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )


    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


def vae_loss(x_hat, x, mu, logvar, kl_weight: float = 0.001):
    recon_loss = nn.functional.mse_loss(
        x_hat,
        x,
        reduction="mean",
    )

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_weight * kl_loss
