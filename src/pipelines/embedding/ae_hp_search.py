import copy
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split

from src.pipelines.clustering.clustering_search import run_clustering_search

from src.pipelines.embedding.ae import AEPipeline


sweep_config = {
    "method": "grid",
    "metric": {
        "name": "silhouette_latent",
        "goal": "maximize",
    },
    "parameters": {
        "output_obsm_key": {
            "values": ["X_ae"],
        },
        "output_adata_path": {
            "values": ["adata_ae.h5ad"],
        },
        "output_model_path": {
            "values": ["ae_model.pt"],
        },
        "data_path": {
            "values": ["Dataset/raw/matrix.mtx"],
        },

        # Preprocessing
        "n_top_genes": {
            "values": [5000, 7000],
        },

        # Autoencoder hyperparameters
        "batch_size": {
            "values": [32, 64, 128],
        },
        "latent_dim": {
            "values": [8, 16, 32, 64, 100],
        },
        "learning_rate": {
            "values": [1e-4, 1e-3],
        },
        "dropout_rate": {
            "values": [0.1, 0.2, 0.3],
        },
    },
}


def sweep_worker():
    load_dotenv()

    run = wandb.init()

    # Use the same seed in every run for fair comparison
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load base AE configuration and overwrite it with
    # the current W&B hyperparameter combination
    pipeline = AEPipeline(
        config_path="embeddings/ae_embedding.yml"
    )

    pipeline.config.update(dict(wandb.config))

    # Load and preprocess the data
    pipeline.setup_data()

    dataset_size = len(pipeline.dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    if val_size < 3:
        raise ValueError(
            "The validation set must contain at least three samples."
        )

    # Use the same train/validation split in every run
    train_dataset, val_dataset = random_split(
        pipeline.dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create the AE after loading the current sweep parameters
    pipeline.setup_model()

    batch_size = pipeline.config["batch_size"]

    # BatchNorm cannot train with a batch containing only one sample
    drop_last_train = train_size % batch_size == 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last_train,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Early stopping configuration
    max_epochs = 100
    patience = 10
    min_delta = 1e-4

    best_val_mse = float("inf")
    best_train_mse = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    # Train the AE
    for epoch in range(max_epochs):

        # Training phase
        pipeline.model.train()

        total_train_loss = 0.0
        processed_train_samples = 0

        for batch_x in train_loader:
            batch_x = batch_x.to(pipeline.device)

            pipeline.optimizer.zero_grad()

            loss = pipeline.compute_loss(batch_x)

            loss.backward()
            pipeline.optimizer.step()

            batch_samples = batch_x.size(0)

            total_train_loss += loss.item() * batch_samples
            processed_train_samples += batch_samples

        avg_train_mse = (
            total_train_loss / processed_train_samples
        )

        # Validation phase
        pipeline.model.eval()

        total_val_loss = 0.0

        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(pipeline.device)

                loss = pipeline.compute_loss(batch_x)

                total_val_loss += (
                    loss.item() * batch_x.size(0)
                )

        avg_val_mse = total_val_loss / len(val_dataset)

        # Check whether validation MSE has improved
        if avg_val_mse < best_val_mse - min_delta:
            best_val_mse = avg_val_mse
            best_train_mse = avg_train_mse
            best_epoch = epoch + 1
            patience_counter = 0

            best_model_state = copy.deepcopy(
                pipeline.model.state_dict()
            )

        else:
            patience_counter += 1

        # Log the training curves
        wandb.log({
            "epoch": epoch + 1,
            "train_mse_epoch": avg_train_mse,
            "val_mse_epoch": avg_val_mse,
        })

        print(
            f"Epoch {epoch + 1:03d}/{max_epochs} | "
            f"Train MSE: {avg_train_mse:.4f} | "
            f"Val MSE: {avg_val_mse:.4f} | "
            f"Best Val MSE: {best_val_mse:.4f}"
        )

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch}"
            )
            break

    # Restore the AE from the epoch with the best validation MSE
    if best_model_state is None:
        raise RuntimeError(
            "No valid AE model state was saved during training."
        )

    pipeline.model.load_state_dict(best_model_state)

    # Generate embeddings for the validation set
    print("\nExtracting validation embeddings...")

    pipeline.model.eval()

    val_embeddings = []

    with torch.no_grad():
        for batch_x in val_loader:
            batch_x = batch_x.to(pipeline.device)

            latent = pipeline.get_latent_representation(
                batch_x
            )

            val_embeddings.append(
                latent.cpu().numpy()
            )

    val_embeddings = np.vstack(val_embeddings)

    # Test all clustering methods and configurations
    print("\nRunning clustering search...")

    all_clustering_results, best_clustering = (
        run_clustering_search(
            embeddings=val_embeddings,
            random_state=seed,
            scale_embeddings=True,
        )
    )

    # Create a W&B table containing every clustering result
    clustering_table = wandb.Table(
        columns=[
            "method",
            "n_neighbors",
            "resolution",
            "n_clusters_requested",
            "covariance_type",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "n_clusters",
            "min_cluster_size",
            "max_cluster_size",
        ]
    )

    for result in all_clustering_results:
        clustering_table.add_data(
            result["method"],
            result.get("n_neighbors"),
            result.get("resolution"),
            result.get("n_clusters_requested"),
            result.get("covariance_type"),
            result["silhouette"],
            result["davies_bouldin"],
            result["calinski_harabasz"],
            result["n_clusters"],
            result["min_cluster_size"],
            result["max_cluster_size"],
        )

    # Main metrics for this AE run
    metrics_to_log = {
        # AE metrics
        "train_mse": best_train_mse,
        "val_mse": best_val_mse,
        "best_epoch": best_epoch,

        # Best clustering found for this AE
        "silhouette_latent": best_clustering["silhouette"],
        "davies_bouldin": best_clustering["davies_bouldin"],
        "calinski_harabasz": (
            best_clustering["calinski_harabasz"]
        ),
        "n_clusters": best_clustering["n_clusters"],

        # Best clustering method
        "best_clustering_method": best_clustering["method"],

        # Table with all clustering configurations
        "clustering_results": clustering_table,
    }

    # Only log parameters that apply to the winning method
    if best_clustering.get("n_neighbors") is not None:
        metrics_to_log["best_n_neighbors"] = (
            best_clustering["n_neighbors"]
        )

    if best_clustering.get("resolution") is not None:
        metrics_to_log["best_resolution"] = (
            best_clustering["resolution"]
        )

    if best_clustering.get("n_clusters_requested") is not None:
        metrics_to_log["best_n_clusters_requested"] = (
            best_clustering["n_clusters_requested"]
        )

    if best_clustering.get("covariance_type") is not None:
        metrics_to_log["best_covariance_type"] = (
            best_clustering["covariance_type"]
        )

    wandb.log(metrics_to_log)

    print("\nFinal AE Metrics:")
    print(f"  Train MSE: {best_train_mse:.4f}")
    print(f"  Best Val MSE: {best_val_mse:.4f}")
    print(f"  Best Epoch: {best_epoch}")

    print("\nBest clustering for this AE configuration:")
    print(f"  Method: {best_clustering['method']}")
    print(
        f"  Silhouette: "
        f"{best_clustering['silhouette']:.4f}"
    )
    print(
        f"  Davies-Bouldin: "
        f"{best_clustering['davies_bouldin']:.4f}"
    )
    print(
        f"  Calinski-Harabasz: "
        f"{best_clustering['calinski_harabasz']:.1f}"
    )
    print(
        f"  Number of clusters: "
        f"{best_clustering['n_clusters']}"
    )

    print("\nBest clustering parameters:")

    for key in [
        "n_neighbors",
        "resolution",
        "n_clusters_requested",
        "covariance_type",
    ]:
        value = best_clustering.get(key)

        if value is not None:
            print(f"  {key}: {value}")

    run.finish()


if __name__ == "__main__":
    project_root = (
        Path(__file__).parent.parent.parent.parent
    )

    load_dotenv(project_root / ".env")

    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")

    print(f"W&B project: {wandb_project}")
    print(f"W&B entity: {wandb_entity}")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=wandb_project,
        entity=wandb_entity,
    )

    # Execute the complete grid
    wandb.agent(
        sweep_id,
        function=sweep_worker,
    )

    # Retrieve the best complete AE + clustering run
    api = wandb.Api()

    sweep_path = (
        f"{wandb_entity}/"
        f"{wandb_project}/"
        f"{sweep_id}"
    )

    sweep = api.sweep(sweep_path)
    best_run = sweep.best_run()

    print("\n" + "=" * 50)
    print(
        f"Best Run: {best_run.name} "
        f"({best_run.id})"
    )

    print("\nOptimization Metric:")
    print(
        "  Silhouette Score: "
        f"{best_run.summary.get('silhouette_latent', np.nan):.5f}"
    )

    print("\nAE Metrics:")
    print(
        "  Validation MSE: "
        f"{best_run.summary.get('val_mse', np.nan):.5f}"
    )
    print(
        "  Best Epoch: "
        f"{best_run.summary.get('best_epoch', 0)}"
    )

    print("\nBest Clustering:")
    print(
        "  Method: "
        f"{best_run.summary.get('best_clustering_method', 'N/A')}"
    )
    print(
        "  Davies-Bouldin: "
        f"{best_run.summary.get('davies_bouldin', np.nan):.5f}"
    )
    print(
        "  Calinski-Harabasz: "
        f"{best_run.summary.get('calinski_harabasz', np.nan):.1f}"
    )
    print(
        "  Number of clusters: "
        f"{best_run.summary.get('n_clusters', 0)}"
    )

    print("\nBest AE hyperparameters:")

    for parameter, value in best_run.config.items():
        print(f"  {parameter}: {value}")

    print("=" * 50)

