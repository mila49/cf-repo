from torch import nn

class AE(nn.Module):
    """
    Simple Autoencoder architecture for dimensionality reduction of single-cell data.
    The encoder compresses the input to a latent representation, while the decoder reconstructs it.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 30,
        dropout_rate: float = 0.3,
    ):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, input_dim)
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


def ae_loss(x_hat, x):
    recon_loss = nn.functional.mse_loss(
        x_hat,
        x,
        reduction="mean",
    )

    return recon_loss