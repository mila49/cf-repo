import torch
from typing import override
from src.pipelines.embedding import EmbeddingPipeline

from src.models import AE, ae_loss


class AEPipeline(EmbeddingPipeline):
    """
    Pipeline for training an Autoencoder (AE) on single-cell data.
    """
    def __init__(self, config_path: str):
        super().__init__(config_path)
    

    @override
    def setup_model(self):
        input_dim = self.adata.shape[1]

        self.model = AE(
            input_dim=input_dim,
            latent_dim=self.config["latent_dim"],
            dropout_rate=self.config.get("dropout_rate", 0.3),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
        )
    

    @override
    def compute_loss(self, batch_x):
        decoded = self.model(batch_x)

        return ae_loss(decoded, batch_x)


    @override
    def get_latent_representation(self, batch_x):
        return self.model.encoder(batch_x)