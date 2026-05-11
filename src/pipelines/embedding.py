import anndata as ad
import yaml
import torch
from torch.utils.data import DataLoader

from src.utils.data_loader import load_data, preprocess_data
from src.utils.datasets import SparseAnnDataset
from src.models.vae import VAE, vae_loss


class EmbeddingPipeline:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)

        self.device = self.config.get("device", "cpu")

        self.adata = None
        self.dataset = None
        self.loader = None
        self.model = None
        self.optimizer = None

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_data(self):
        self.adata = load_data(self.config["data_path"])

        self.adata = preprocess_data(
            self.adata,
            n_top_genes=self.config["n_top_genes"],
        )

        self.dataset = SparseAnnDataset(self.adata.X)

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
        )

    def setup_model(self):
        input_dim = self.adata.shape[1]

        self.model = VAE(
            input_dim=input_dim,
            latent_dim=self.config["latent_dim"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
        )

    def train(self):
        epochs = self.config["epochs"]
        kl_weight = self.config.get("kl_weight", 0.001)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_x in self.loader:
                batch_x = batch_x.to(self.device)

                x_hat, mu, logvar, z = self.model(batch_x)

                loss = vae_loss(
                    x_hat,
                    batch_x,
                    mu,
                    logvar,
                    kl_weight=kl_weight,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_x.size(0)

            avg_loss = total_loss / len(self.dataset)
            print(f"Epoch {epoch + 1:03d} | loss = {avg_loss:.4f}")

    def generate_embeddings(self):
        self.model.eval()
        latents = []

        eval_loader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        with torch.no_grad():
            for batch_x in eval_loader:
                batch_x = batch_x.to(self.device)

                mu, logvar = self.model.encode(batch_x)
                latents.append(mu.cpu())

        self.adata.obsm["X_vae"] = torch.cat(latents, dim=0).numpy()

    def save_outputs(self):
        ad.settings.allow_write_nullable_strings = True

        self.adata.write(self.config["output_adata_path"])
        torch.save(
            self.model.state_dict(),
            self.config["output_model_path"],
        )

        print("Saved:")
        print(self.config["output_adata_path"])
        print(self.config["output_model_path"])

    def run(self):
        self.setup_data()
        self.setup_model()
        self.train()
        self.generate_embeddings()
        self.save_outputs()
