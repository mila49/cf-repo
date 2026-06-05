import torch
from torch import nn


class DAE(nn.Module):
    """
    Denoising Autoencoder architecture for dimensionality reduction of single-cell data.
    During training, a random fraction of input genes is masked to zero before encoding.
    The decoder must reconstruct the original clean input, forcing the encoder to learn
    robust latent representations rather than simply copying the input.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 30,
        dropout_rate: float = 0.2,
        mask_rate: float = 0.3,
    ):
        super(DAE, self).__init__()
        self.mask_rate = mask_rate

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, input_dim)
        )


    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1.0 - self.mask_rate))
            x = x * mask

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


def dae_loss(x_hat, x):
    recon_loss = nn.functional.mse_loss(
        x_hat,
        x,
        reduction="mean",
    )

    return recon_loss
