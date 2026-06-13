import os
import numpy as np
import torch
import yaml
import wandb
import scanpy as sc
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.pipelines.embedding.ae import AEPipeline  

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'silhouette_latent',
        'goal': 'maximize'
    },
    'parameters': {
        'output_obsm_key': {
            'values': ["X_ae"]
        },
        'output_adata_path': {
            'values': ["adata_ae.h5ad"]
        },
        'output_model_path': {
            'values': ["ae_model.pt"]
        },
        'data_path': {
            'values': ["Dataset/raw/matrix.mtx"]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'n_top_genes': {
            'values': [5000, 7000, 9000]
        },
        'epochs': {
            'values': [30, 40, 50, 60]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'latent_dim': {
            'values': [16, 32, 64, 128]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }
    }
}

def sweep_worker():
    load_dotenv()
    
    run = wandb.init()
    
    pipeline = AEPipeline(config_path="ae_embedding.yml")
    pipeline.config.update(wandb.config)
    
    # Load clustering config for Leiden resolution
    clustering_config_path = Path(__file__).parent.parent.parent.parent / "config" / "clustering" / "leiden_clustering.yml"
    with open(clustering_config_path, "r") as f:
        clustering_config = yaml.safe_load(f)
    leiden_resolution = clustering_config.get("resolution", 0.5)
    n_neighbors = clustering_config.get("n_neighbors", 15)
    
    pipeline.setup_data()
    pipeline.setup_model()
    
    # Split data into train/val (80/20)
    dataset_size = len(pipeline.dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    from torch.utils.data import random_split, DataLoader
    train_dataset, val_dataset = random_split(
        pipeline.dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=pipeline.config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=pipeline.config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    epochs = pipeline.config["epochs"]
    
    # Track best validation loss for early stopping awareness
    best_val_mse = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        pipeline.model.train()
        total_train_loss = 0
        
        for batch_x in train_loader:
            batch_x = batch_x.to(pipeline.device)
            
            loss = pipeline.compute_loss(batch_x)
            
            pipeline.optimizer.zero_grad()
            loss.backward()
            pipeline.optimizer.step()
            
            total_train_loss += loss.item() * batch_x.size(0)
        
        avg_train_mse = total_train_loss / len(train_dataset)
        
        # Validation phase
        pipeline.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(pipeline.device)
                loss = pipeline.compute_loss(batch_x)
                total_val_loss += loss.item() * batch_x.size(0)
        
        avg_val_mse = total_val_loss / len(val_dataset)
        
        # Track best validation loss
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
        
        print(f"Epoch {epoch + 1:03d}/{epochs} | Train MSE: {avg_train_mse:.4f} | Val MSE: {avg_val_mse:.4f}")
    
    # After all epochs, compute final clustering metrics
    print("\nComputing final clustering metrics...")
    pipeline.model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch_x in train_loader:
            batch_x = batch_x.to(pipeline.device)
            latent = pipeline.get_latent_representation(batch_x)
            embeddings.append(latent.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    # Create temporary AnnData object for clustering
    import anndata as ad
    adata_temp = ad.AnnData(X=embeddings)
    
    # Compute neighborhood graph using config parameters
    sc.pp.neighbors(adata_temp, n_neighbors=n_neighbors, use_rep='X')
    
    # Run Leiden clustering with config resolution
    sc.tl.leiden(adata_temp, resolution=leiden_resolution, key_added='leiden')
    
    cluster_labels = adata_temp.obs['leiden'].astype(int).values
    n_clusters = len(np.unique(cluster_labels))
    
    # Compute clustering metrics
    silhouette = silhouette_score(embeddings, cluster_labels)
    db_score = davies_bouldin_score(embeddings, cluster_labels)
    ch_score = calinski_harabasz_score(embeddings, cluster_labels)
    
    # Compute supervised metrics if reference labels are available
    # Common column names for cell type annotations: 'cell_type', 'celltype', 'cluster', 'leiden', 'louvain'
    reference_label_cols = ['cell_type', 'celltype', 'CellType', 'cluster', 'leiden', 'louvain', 'labels']
    reference_labels = None
    reference_col = None
    
    for col in reference_label_cols:
        if col in pipeline.adata.obs.columns:
            reference_col = col
            # Get labels for train dataset only (matching embeddings)
            all_labels = pipeline.adata.obs[col].values
            train_indices = train_dataset.indices
            reference_labels = all_labels[train_indices]
            break
    
    metrics_to_log = {
        "train_mse": avg_train_mse,
        "val_mse": best_val_mse,
        "silhouette_latent": silhouette,
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "n_clusters": n_clusters,
        "leiden_resolution": leiden_resolution,
        "n_neighbors": n_neighbors,
    }
    
    if reference_labels is not None:
        # Compute ARI and NMI
        ari = adjusted_rand_score(reference_labels, cluster_labels)
        nmi = normalized_mutual_info_score(reference_labels, cluster_labels)
        
        metrics_to_log["ari_reference"] = ari
        metrics_to_log["nmi_reference"] = nmi
        
        print(f"\nSupervised Metrics (using '{reference_col}' as reference):")
        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")
    
    # Log final metrics ONCE per hyperparameter combination
    wandb.log(metrics_to_log)
    
    print(f"\nFinal Unsupervised Metrics:")
    print(f"  Train MSE: {avg_train_mse:.4f}")
    print(f"  Best Val MSE: {best_val_mse:.4f}")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Davies-Bouldin: {db_score:.4f}")
    print(f"  Calinski-Harabasz: {ch_score:.1f}")
    print(f"  N Clusters: {n_clusters}")
    
    run.finish()


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

    print(os.environ.get("WANDB_PROJECT"))
    print(os.environ.get("WANDB_ENTITY"))
    
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY")
    )
    
    wandb.agent(sweep_id, function=sweep_worker, count=3)
    
    api = wandb.Api()
    sweep_path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{sweep_id}"
    sweep = api.sweep(sweep_path)
    
    best_run = sweep.best_run()
    
    print("\n" + "="*50)
    print(f"Best Run: {best_run.name} ({best_run.id})")
    print(f"\nOptimization Metric:")
    print(f"  Silhouette Score: {best_run.summary.get('silhouette_latent', 0):.5f}")
    print(f"\nUnsupervised Metrics:")
    print(f"  Validation MSE: {best_run.summary.get('val_mse', 0):.5f}")
    print(f"  Davies-Bouldin: {best_run.summary.get('davies_bouldin', 0):.5f}")
    print(f"  Calinski-Harabasz: {best_run.summary.get('calinski_harabasz', 0):.1f}")
    print(f"  Optimal Clusters: {best_run.summary.get('n_clusters', 0)}")
    
    # Show supervised metrics if available
    if 'ari_reference' in best_run.summary:
        print(f"\nSupervised Metrics (validation):")
        print(f"  ARI: {best_run.summary.get('ari_reference', 0):.5f}")
        print(f"  NMI: {best_run.summary.get('nmi_reference', 0):.5f}")
    
    print("\nBest hyperparameters:")
    for param, value in best_run.config.items():
        print(f"  {param}: {value}")

    print("="*50)
