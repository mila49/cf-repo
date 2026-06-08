import os
from abc import abstractmethod

import anndata as ad
from anndata import AnnData
import torch
from torch.nn import Module
from torch.optim import Optimizer
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.utils.data_loader import load_data, preprocess_data
from src.utils.datasets import SparseAnnDataset
from ..main import Pipeline


class EmbeddingPipeline(Pipeline):
    def __init__(self, config_file: str):
        super().__init__(config_file)

        self.device = self.config.get("device", "cpu")

        self.adata: AnnData = None
        self.dataset: SparseAnnDataset = None
        self.loader: DataLoader = None
        self.model: type[Module] = None
        self.optimizer: type[Optimizer] = None


    def setup_data(self) -> None:
        """
        Setup the data for training.
        """
        self.adata = load_data(self.root_dir / self.config["data_path"])

        self.adata = preprocess_data(
            self.adata,
            n_top_genes=self.config["n_top_genes"],
        )

        self.dataset = SparseAnnDataset(self.adata.X)

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
        )


    @abstractmethod
    def setup_model(self):
        """
        Setup the model and optimizer.
        """
        raise NotImplementedError("Must be implemented in subclass")


    @abstractmethod
    def compute_loss(self, **kwargs):
        """
        Compute the loss for a given batch of data.
        """
        raise NotImplementedError("Must be implemented in subclass")


    @abstractmethod
    def get_latent_representation(self, **kwargs):
        """
        Get the latent representation for a given batch of data.
        """
        raise NotImplementedError("Must be implemented in subclass")


    def _run_training_impl(self):
        """
        Internal implementation of the complete training pipeline.
        Called either directly (if no sweep) or within sweep context.
        """
        self.setup_data()
        self.setup_model()
        self.train()
        self.generate_embeddings()
        self.save_outputs()


    def train(self):
        """
        Train the embedding model.
        Only executes the training loop. Setup and save must be called separately
        or within _run_training_impl() when using sweeps.
        """
        epochs = self.config["epochs"]

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_x in self.loader:
                batch_x = batch_x.to(self.device)

                loss = self.compute_loss(batch_x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_x.size(0)

            avg_loss = total_loss / len(self.dataset)
            print(f"Epoch {epoch + 1:03d} | loss = {avg_loss:.4f}")
            wandb.log({"loss": avg_loss})


    def generate_embeddings(self):
        """
        Generate the latent embeddings for all cells and store them in adata.obsm["X_emb"].
        """
        self.model.eval()
        latents = [] 

        eval_loader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 0),
        )

        with torch.no_grad():
            for batch_x in eval_loader:
                batch_x = batch_x.to(self.device)

                latent = self.get_latent_representation(batch_x)
                latents.append(latent.cpu())

        key_name = self.config.get("output_obsm_key", "X_emb")
        self.adata.obsm[key_name] = torch.cat(latents, dim=0).numpy()


    def save_outputs(self):
        """
        Save the AnnData object with embeddings and the trained model weights.
        """
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
        """
        Run the embedding pipeline.
        
        If wb_sweep is enabled, delegates to run_with_sweep() which handles sweep initialization.
        Otherwise, executes the full pipeline directly with W&B logging.
        """
        load_dotenv()
        
        if self.wb_sweep:
            self.run_with_sweep(
                train_function=self._run_training_impl,
                project=os.environ["WANDB_PROJECT"],
                entity=os.environ["WANDB_ENTITY"],
            )
        else:
            run = wandb.init(
                entity=os.environ["WANDB_ENTITY"],
                project=os.environ["WANDB_PROJECT"],
                job_type="embedding",
                config=self.config,
            )
            
            self._run_training_impl()
            run.finish()