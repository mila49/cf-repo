import torch
from typing import override

from src.models import VAE, vae_loss
from src.pipelines.embedding import EmbeddingPipeline


class VAEPipeline(EmbeddingPipeline):
    """
    Pipeline for training a variational autoencoder on single-cell data.
    """

    def __init__(self, config_path):
        super().__init__(config_path)

    @override
    def setup_model(self):
        input_dim = self.adata.shape[1]

        self.model = VAE(
            input_dim=input_dim,
            latent_dim=self.config["latent_dim"],
            dropout_rate=self.config.get("dropout_rate", 0.2),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
        )

    @override
    def compute_loss(self, batch_x):
        reconstruction, mu, logvar, _ = self.model(batch_x)
        kl_weight = self.config.get("kl_weight", 0.001)

        return vae_loss(
            reconstruction,
            batch_x,
            mu,
            logvar,
            kl_weight=kl_weight,
        )

    @override
    def get_latent_representation(self, batch_x):
        # Use mu rather than a random sample z so clustering is deterministic.
        mu, _ = self.model.encode(batch_x)
        return mu