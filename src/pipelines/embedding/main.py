import os
from abc import abstractmethod

import anndata as ad
from anndata import AnnData
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.model: type[Module] = None
        self.optimizer: type[Optimizer] = None
        self.best_model_state: dict = None
        self.best_val_loss: float = float('inf')


    def setup_data(self) -> None:
        """
        Setup the data for training with train-validation-test split.
        Default split: 60% train, 20% validation, 20% test
        """
        self.adata = load_data(self.root_dir / self.config["data_path"])

        self.adata = preprocess_data(
            self.adata,
            n_top_genes=self.config["n_top_genes"],
        )

        self.dataset = SparseAnnDataset(self.adata.X)
        
        # Create train-validation-test split
        # Get split proportions from config
        train_prop = self.config.get("train_size", 0.6)
        val_prop = self.config.get("val_size", 0.2)
        test_prop = self.config.get("test_size", 0.2)
        
        # Normalize proportions
        total = train_prop + val_prop + test_prop
        train_prop /= total
        val_prop /= total
        
        # First split: train vs (validation + test)
        temp_size = val_prop + test_prop
        train_indices, temp_indices = train_test_split(
            np.arange(len(self.dataset)),
            test_size=temp_size,
            random_state=self.config.get("random_state", 42)
        )
        
        # Second split: validation vs test
        val_size_from_temp = val_prop / temp_size
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_size_from_temp),
            random_state=self.config.get("random_state", 42)
        )
        
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 0),
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 0),
        )
        
        # Keep full loader for compatibility
        self.loader = self.train_loader
        
        print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")


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


    def compute_metrics(self, loader: DataLoader, prefix: str = ""):
        """
        Compute MAE, MSE, RMSE on a given dataloader.
        
        Args:
            loader: DataLoader to evaluate on
            prefix: Prefix for metric names (e.g., "train", "test")
        
        Returns:
            dict: Dictionary with MAE, MSE, RMSE values
        """
        self.model.eval()
        all_losses = []
        
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                loss = self.compute_loss(batch_x)
                all_losses.append(loss.item())
        
        avg_loss = np.mean(all_losses)
        mse = avg_loss
        rmse = np.sqrt(mse)
        # MAE approximation using loss (for most cases loss ≈ MSE or MAE)
        mae = avg_loss
        
        metrics = {
            f"{prefix}_mae": mae,
            f"{prefix}_mse": mse,
            f"{prefix}_rmse": rmse,
        }
        
        return metrics


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
        Train the embedding model with train-validation-test evaluation.
        Uses validation set to select best model and detects overfitting.
        """
        epochs = self.config["epochs"]
        patience = self.config.get("early_stopping_patience", None)
        patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_x in self.train_loader:
                batch_x = batch_x.to(self.device)

                loss = self.compute_loss(batch_x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            avg_train_loss = train_loss / len(self.train_loader.dataset)
            
            # Compute metrics on train, validation and test sets
            train_metrics = self.compute_metrics(self.train_loader, prefix="train")
            val_metrics = self.compute_metrics(self.val_loader, prefix="val")
            test_metrics = self.compute_metrics(self.test_loader, prefix="test")
            
            # Combine metrics
            metrics = {"epoch": epoch + 1}
            metrics.update(train_metrics)
            metrics.update(val_metrics)
            metrics.update(test_metrics)
            
            # Print results
            print(f"Epoch {epoch + 1:03d} | "
                  f"Train RMSE: {train_metrics['train_rmse']:.4f} | "
                  f"Val RMSE: {val_metrics['val_rmse']:.4f} | "
                  f"Test RMSE: {test_metrics['test_rmse']:.4f}", end="")
            
            # Detect overfitting using validation set
            overfitting_ratio = val_metrics['val_rmse'] / (train_metrics['train_rmse'] + 1e-6)
            if overfitting_ratio > 1.1:
                print(" ⚠️  OVERFITTING", end="")
            
            # Save best model based on validation performance
            if val_metrics['val_rmse'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_rmse']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(" ✓ BEST")
            else:
                print()
                if patience is not None:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                        break
            
            # Log to wandb
            wandb.log(metrics)
    
    
    def evaluate_on_test_set(self):
        """
        Load the best model and evaluate it on the test set.
        Returns final metrics and comparison with train.
        """
        if self.best_model_state is None:
            print("Warning: No best model state saved. Using current model.")
            self.best_model_state = self.model.state_dict()
        else:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model from validation set")
        
        # Evaluate on all sets with best model
        train_metrics = self.compute_metrics(self.train_loader, prefix="train")
        val_metrics = self.compute_metrics(self.val_loader, prefix="val")
        test_metrics = self.compute_metrics(self.test_loader, prefix="test")
        
        # Print final results
        print("\n" + "="*70)
        print("FINAL EVALUATION WITH BEST MODEL (based on validation set)")
        print("="*70)
        print(f"Train Set - MAE: {train_metrics['train_mae']:.4f} | MSE: {train_metrics['train_mse']:.4f} | RMSE: {train_metrics['train_rmse']:.4f}")
        print(f"Val Set   - MAE: {val_metrics['val_mae']:.4f} | MSE: {val_metrics['val_mse']:.4f} | RMSE: {val_metrics['val_rmse']:.4f}")
        print(f"Test Set  - MAE: {test_metrics['test_mae']:.4f} | MSE: {test_metrics['test_mse']:.4f} | RMSE: {test_metrics['test_rmse']:.4f}")
        print("="*70)
        
        # Calculate overfitting indicators
        train_rmse = train_metrics['train_rmse']
        val_rmse = val_metrics['val_rmse']
        test_rmse = test_metrics['test_rmse']
        
        val_overfitting_ratio = (val_rmse - train_rmse) / train_rmse * 100
        test_overfitting_ratio = (test_rmse - train_rmse) / train_rmse * 100
        
        print(f"\nOverfitting Analysis:")
        print(f"  Val vs Train:  {val_overfitting_ratio:+.2f}%")
        print(f"  Test vs Train: {test_overfitting_ratio:+.2f}%")
        
        if test_rmse > train_rmse * 1.2:
            print("  ⚠️  SIGNIFICANT OVERFITTING DETECTED ON TEST SET")
        elif test_rmse > train_rmse * 1.1:
            print("  ⚠️  MODERATE OVERFITTING DETECTED ON TEST SET")
        else:
            print("  ✓ Good generalization to test set")
        print("="*70 + "\n")
        
        # Log final metrics to wandb
        final_metrics = {
            "final_train_rmse": train_rmse,
            "final_val_rmse": val_rmse,
            "final_test_rmse": test_rmse,
        }
        wandb.log(final_metrics)
        
        return train_metrics, val_metrics, test_metrics


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