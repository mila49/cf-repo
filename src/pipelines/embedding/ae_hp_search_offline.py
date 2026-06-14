"""
Offline Hyperparameter Search for Autoencoder
No internet connection required - runs locally and saves results to CSV

Evaluation Strategy:
- Train autoencoder on training split
- Extract validation embeddings (shuffle=False for alignment)
- Build kNN graph on validation embeddings (fixed n_neighbors)
- Run Leiden clustering (fixed resolution)
- Select best model by VALIDATION SILHOUETTE (unsupervised)
- Use ARI/NMI only for posterior evaluation (not for model selection)
"""
import os
import numpy as np
import torch
import yaml
import scanpy as sc
import pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from src.pipelines.embedding.ae import AEPipeline  


# Define hyperparameter grid

PARAM_GRID = {
    "learning_rate": [3e-5, 6e-5, 1e-4],
    "n_top_genes": [5000],
    "epochs": [5, 10],
    "batch_size": [32, 64, 128],
    "latent_dim": [12, 16, 24, 32],
    "dropout_rate": [0.2],
}

# Medium search: 96 combinations (~4-6 hours) - uncomment to use
# PARAM_GRID = {
#     'learning_rate': [1e-4, 1e-3],
#     'n_top_genes': [5000, 7000],
#     'epochs': [40, 50],
#     'batch_size': [64, 128],
#     'latent_dim': [32, 64],
#     'dropout_rate': [0.2, 0.3, 0.4],
# }

# Configuration for embedding standardization
SCALE_EMBEDDINGS = True  # Standardize embeddings before graph construction


def train_single_config(config_dict, run_id, total_runs):
    """
    Train a single hyperparameter configuration.
    Evaluates on VALIDATION set using unsupervised clustering metrics.
    """
    print(f"\n{'='*80}")
    print(f"Run {run_id}/{total_runs}")
    print(f"Configuration: {config_dict}")
    print(f"{'='*80}\n")
    
    # Create pipeline
    pipeline = AEPipeline(config_path="embeddings/ae_embedding.yml")
    
    # Update config with current hyperparameters
    for key, value in config_dict.items():
        pipeline.config[key] = value
    
    # Load clustering config (FIXED parameters for fair comparison)
    clustering_config_path = Path(__file__).parent.parent.parent.parent / "config" / "clustering" / "leiden_clustering.yml"
    with open(clustering_config_path, "r") as f:
        clustering_config = yaml.safe_load(f)
    leiden_resolution = clustering_config.get("resolution", 0.5)
    n_neighbors = clustering_config.get("n_neighbors", 400)
    
    # Setup
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
    best_val_mse = float('inf')
    
    # Training loop
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
        
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs} | Train MSE: {avg_train_mse:.4f} | Val MSE: {avg_val_mse:.4f}")
    
    # Extract VALIDATION embeddings (shuffle=False for proper alignment)
    print("\nExtracting validation embeddings...")
    pipeline.model.eval()
    val_embeddings = []
    
    # Create embedding loader with shuffle=False to maintain order
    embedding_loader = DataLoader(
        val_dataset,
        batch_size=pipeline.config["batch_size"],
        shuffle=False,  # CRITICAL: keep order for alignment
        num_workers=0,
    )
    
    with torch.no_grad():
        for batch_x in embedding_loader:
            batch_x = batch_x.to(pipeline.device)
            latent = pipeline.get_latent_representation(batch_x)
            val_embeddings.append(latent.cpu().numpy())
    
    val_embeddings = np.vstack(val_embeddings)
    
    # Optionally standardize embeddings before graph construction
    if SCALE_EMBEDDINGS:
        scaler = StandardScaler()
        val_embeddings_scaled = scaler.fit_transform(val_embeddings)
    else:
        val_embeddings_scaled = val_embeddings
    
    # Build kNN graph on validation embeddings
    print(f"Building kNN graph (n_neighbors={n_neighbors})...")
    import anndata as ad
    adata_temp = ad.AnnData(X=val_embeddings_scaled)
    sc.pp.neighbors(adata_temp, n_neighbors=n_neighbors, use_rep='X')
    
    # Run Leiden clustering with FIXED resolution
    print(f"Running Leiden clustering (resolution={leiden_resolution})...")
    sc.tl.leiden(adata_temp, resolution=leiden_resolution, key_added='leiden')
    
    cluster_labels = adata_temp.obs['leiden'].astype(int).values
    n_clusters = len(np.unique(cluster_labels))
    n_samples = len(cluster_labels)
    
    # Cluster size diagnostics
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    min_cluster_size = int(cluster_counts.min())
    max_cluster_size = int(cluster_counts.max())
    median_cluster_size = int(np.median(cluster_counts))
    cluster_size_ratio = max_cluster_size / min_cluster_size if min_cluster_size > 0 else np.inf
    
    # Initialize clustering metrics
    silhouette_val = np.nan
    db_score_val = np.nan
    ch_score_val = np.nan
    
    # Guard against invalid clustering (need at least 2 clusters and less than n_samples)
    if n_clusters > 1 and n_clusters < n_samples:
        try:
            # Use ORIGINAL embeddings (not scaled) for metric computation
            silhouette_val = silhouette_score(val_embeddings, cluster_labels)
            db_score_val = davies_bouldin_score(val_embeddings, cluster_labels)
            ch_score_val = calinski_harabasz_score(val_embeddings, cluster_labels)
        except Exception as e:
            print(f"Warning: Could not compute clustering metrics: {e}")
    else:
        print(f"Warning: Invalid clustering with {n_clusters} clusters (need 2 <= n_clusters < {n_samples})")
    
    # Posterior evaluation with reference labels (NOT used for model selection)
    # We compute ARI/NMI separately for both major and minor annotations.

    reference_label_cols = ["major", "minor"]

    posterior_metrics = {}

    val_indices = val_dataset.indices

    for col in reference_label_cols:
        ari_col = np.nan
        nmi_col = np.nan

        if col in pipeline.adata.obs.columns:
            all_labels = pipeline.adata.obs[col].values
            reference_labels_val = all_labels[val_indices]

            try:
                ari_col = adjusted_rand_score(reference_labels_val, cluster_labels)
                nmi_col = normalized_mutual_info_score(reference_labels_val, cluster_labels)
            except Exception as e:
                print(f"Warning: Could not compute ARI/NMI for {col}: {e}")

        else:
            print(f"Warning: Reference label column '{col}' not found in adata.obs")

        posterior_metrics[f"ari_reference_{col}_val"] = ari_col
        posterior_metrics[f"nmi_reference_{col}_val"] = nmi_col
    
    # Collect results
    results = {
            **config_dict,
            "train_mse": avg_train_mse,
            "val_mse": avg_val_mse,
            "best_val_mse": best_val_mse,

            "silhouette_val": silhouette_val,
            "davies_bouldin_val": db_score_val,
            "calinski_harabasz_val": ch_score_val,

            "n_clusters_val": n_clusters,
            "min_cluster_size_val": min_cluster_size,
            "max_cluster_size_val": max_cluster_size,
            "median_cluster_size_val": median_cluster_size,
            "cluster_size_ratio_val": cluster_size_ratio,

            "leiden_resolution": leiden_resolution,
            "n_neighbors": n_neighbors,
            "scale_embeddings": SCALE_EMBEDDINGS,

            **posterior_metrics,
        }
    
    print(f"\nReconstruction Metrics:")
    print(f"  Train MSE: {avg_train_mse:.4f}")
    print(f"  Val MSE: {avg_val_mse:.4f}")
    print(f"  Best Val MSE: {best_val_mse:.4f}")
    
    print(f"\nValidation Clustering Metrics (Unsupervised):")
    print(f"  Silhouette: {silhouette_val:.4f}" if not np.isnan(silhouette_val) else f"  Silhouette: NaN")
    print(f"  Davies-Bouldin: {db_score_val:.4f}" if not np.isnan(db_score_val) else f"  Davies-Bouldin: NaN")
    print(f"  Calinski-Harabasz: {ch_score_val:.1f}" if not np.isnan(ch_score_val) else f"  Calinski-Harabasz: NaN")
    
    print(f"\nCluster Diagnostics:")
    print(f"  N Clusters: {n_clusters}")
    print(f"  Min Cluster Size: {min_cluster_size}")
    print(f"  Max Cluster Size: {max_cluster_size}")
    print(f"  Median Cluster Size: {median_cluster_size}")
    print(f"  Size Ratio (max/min): {cluster_size_ratio:.2f}")
    
    posterior_keys = [
    key for key in posterior_metrics.keys()
    if key.startswith("ari_reference_") or key.startswith("nmi_reference_")
]

    has_posterior_metrics = any(
        not np.isnan(value)
        for key, value in posterior_metrics.items()
        if key in posterior_keys
    )

    if has_posterior_metrics:
        print("\nPosterior Reference-Label Evaluation:")
        print("  (These are for evaluation only, NOT used for model selection)")

        for col in ["major", "minor"]:
            ari_key = f"ari_reference_{col}_val"
            nmi_key = f"nmi_reference_{col}_val"

            ari_value = posterior_metrics.get(ari_key, np.nan)
            nmi_value = posterior_metrics.get(nmi_key, np.nan)

            if not np.isnan(ari_value):
                print(f"\n  Reference Column: {col}")
                print(f"    ARI (Validation): {ari_value:.5f}")
                print(f"    NMI (Validation): {nmi_value:.5f}")
        
        return results


def main():
    """
    Run grid search over all hyperparameter combinations.
    """
    print("\n" + "="*80)
    print("OFFLINE HYPERPARAMETER SEARCH FOR AUTOENCODER")
    print("="*80)
    
    # Generate all combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))
    
    total_runs = len(all_combinations)
    print(f"\nTotal hyperparameter combinations: {total_runs}")
    print(f"Parameter grid:")
    for key, values in PARAM_GRID.items():
        print(f"  {key}: {values}")
    print()
    
    # Confirm before running
    response = input(f"This will run {total_runs} training runs. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run all combinations
    all_results = []
    start_time = datetime.now()
    
    for i, combination in enumerate(all_combinations, 1):
        config_dict = dict(zip(param_names, combination))
        
        try:
            results = train_single_config(config_dict, i, total_runs)
            all_results.append(results)
            
            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv("hp_search_results_temp.csv", index=False)
            
        except Exception as e:
            print(f"\nERROR in run {i}: {e}")
            print("Continuing with next configuration...\n")
            continue
    
    # Save final results
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"hp_search_results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    
    elapsed_time = datetime.now() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*80)
    print(f"Total runs: {len(all_results)}/{total_runs}")
    print(f"Time elapsed: {elapsed_time}")
    print(f"Results saved to: {output_file}")
    
    # Check if any runs succeeded
    if len(all_results) == 0:
        print("\n" + "="*80)
        print("ERROR: All runs failed. Please check:")
        print("  1. Data files exist (matrix.mtx, genes.csv, metadata.csv)")
        print("  2. Paths in config/embeddings/ae_embedding.yml are correct")
        print("  3. Virtual environment has all dependencies installed")
        print("="*80)
        return
    
    # Filter valid configurations (avoid degenerate clusterings)
    print("\n" + "="*80)
    print("FILTERING VALID CONFIGURATIONS")
    print("="*80)
    
    valid_results = []
    for res in all_results:
        # Filter criteria:
        # 1. Silhouette is not NaN
        # 2. At least 2 clusters
        # 3. Minimum cluster size >= 5 (avoid tiny clusters)
        # 4. Not too many clusters (< 100)
        is_valid = (
            not np.isnan(res['silhouette_val']) and
            res['n_clusters_val'] >= 2 and
            res['min_cluster_size_val'] >= 5 and
            res['n_clusters_val'] < 100
        )
        
        if is_valid:
            valid_results.append(res)
    
    print(f"Valid configurations: {len(valid_results)}/{len(all_results)}")
    
    if len(valid_results) == 0:
        print("\nWARNING: No valid configurations found!")
        print("All configurations produced degenerate clusterings.")
        print("Consider adjusting hyperparameter ranges or clustering parameters.")
        return
    
    # Convert to DataFrame
    valid_df = pd.DataFrame(valid_results)
    
    # Find best configuration by VALIDATION SILHOUETTE
    best_idx = valid_df['silhouette_val'].idxmax()
    best_config = valid_df.loc[best_idx]
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION BY VALIDATION SILHOUETTE")
    print("="*80)
    print(f"\nPrimary Optimization Metric (Unsupervised):")
    print(f"  Silhouette (Validation): {best_config['silhouette_val']:.5f}")
    
    print(f"\nBest Hyperparameters:")
    for param in param_names:
        if param in best_config:
            print(f"  {param}: {best_config[param]}")
    
    print(f"\nReconstruction Metrics:")
    print(f"  Train MSE: {best_config['train_mse']:.5f}")
    print(f"  Val MSE: {best_config['val_mse']:.5f}")
    print(f"  Best Val MSE: {best_config['best_val_mse']:.5f}")
    
    print(f"\nValidation Clustering Metrics (Unsupervised):")
    print(f"  Davies-Bouldin: {best_config['davies_bouldin_val']:.5f}")
    print(f"  Calinski-Harabasz: {best_config['calinski_harabasz_val']:.1f}")
    
    print(f"\nCluster Diagnostics (Validation):")
    print(f"  N Clusters: {int(best_config['n_clusters_val'])}")
    print(f"  Min Cluster Size: {int(best_config['min_cluster_size_val'])}")
    print(f"  Max Cluster Size: {int(best_config['max_cluster_size_val'])}")
    print(f"  Median Cluster Size: {int(best_config['median_cluster_size_val'])}")
    print(f"  Size Ratio (max/min): {best_config['cluster_size_ratio_val']:.2f}")
    
    print(f"\nFixed Clustering Parameters:")
    print(f"  Leiden Resolution: {best_config['leiden_resolution']}")
    print(f"  kNN Neighbors: {int(best_config['n_neighbors'])}")
    print(f"  Standardize Embeddings: {best_config['scale_embeddings']}")
    
    # Posterior ARI/NMI (may be stored per reference column, e.g. ari_reference_major_val)
    posterior_columns = [c for c in best_config.index if c.startswith('ari_reference_') or c.startswith('nmi_reference_')]
    has_posterior = any(not pd.isna(best_config.get(c, np.nan)) for c in posterior_columns)
    if has_posterior:
        print(f"\n" + "-"*80)
        print("POSTERIOR REFERENCE-LABEL EVALUATION")
        print("(NOT used for model selection, only for evaluation)")
        print("-"*80)
        # Print any recognized posterior metrics (major/minor)
        for col in ['major', 'minor']:
            ari_k = f'ari_reference_{col}_val'
            nmi_k = f'nmi_reference_{col}_val'
            ari_v = best_config.get(ari_k, np.nan)
            nmi_v = best_config.get(nmi_k, np.nan)
            if not pd.isna(ari_v):
                print(f"  Reference Column: {col}")
                print(f"    ARI (Validation): {ari_v:.5f}")
                if not pd.isna(nmi_v):
                    print(f"    NMI (Validation): {nmi_v:.5f}")
    
    print("\n" + "="*80)
    
    # Show top 5 configurations
    print("\nTOP 5 CONFIGURATIONS (by Validation Silhouette):")
    print("="*80)
    top5 = valid_df.nlargest(5, 'silhouette_val')
    display_cols = ['learning_rate', 'latent_dim', 'batch_size', 'epochs', 
                    'silhouette_val', 'val_mse', 'n_clusters_val', 
                    'min_cluster_size_val', 'max_cluster_size_val']
    print(top5[display_cols].to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
