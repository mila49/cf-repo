import yaml
import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def load_config(config_path: str) -> dict:
    """Load clustering/comparison parameters from a YAML config file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    # Load config
    config = load_config("config/compare_clusters.yml")

    # Load clustered AnnData file
    adata = sc.read_h5ad(config["output_adata_path"])

    # Get column names from config
    leiden_key = config["leiden_key"]
    reference_label = config["reference_label"]

    # Check that required columns exist
    if leiden_key not in adata.obs.columns:
        raise ValueError(f"Column '{leiden_key}' not found in adata.obs.")

    if reference_label not in adata.obs.columns:
        raise ValueError(f"Column '{reference_label}' not found in adata.obs.")

    # Show number of cells per VAE cluster
    print("\nNumber of cells per VAE Leiden cluster:")
    print(adata.obs[leiden_key].value_counts().sort_index())

    # Show number of cells per reference major group
    print(f"\nNumber of cells per reference '{reference_label}' group:")
    print(adata.obs[reference_label].value_counts())

    # Count how many cells from each major group fall into each VAE cluster
    counts = pd.crosstab(
        adata.obs[reference_label],
        adata.obs[leiden_key],
    )

    print(f"\nCounts: reference '{reference_label}' groups vs VAE Leiden clusters")
    print(counts)
    counts.to_csv("./crosstab_counts.csv")

    # Calculate percentages by reference group
    # This shows whether cells from the same paper group are grouped together
    percentages = pd.crosstab(
        adata.obs[reference_label],
        adata.obs[leiden_key],
        normalize="index",
    ) * 100

    print(f"\nPercentages by reference '{reference_label}' group:")
    print(percentages.round(1))

     # Compute clustering similarity metrics
    ari = adjusted_rand_score(
        adata.obs[reference_label],
        adata.obs[leiden_key],
    )

    nmi = normalized_mutual_info_score(
        adata.obs[reference_label],
        adata.obs[leiden_key],
    )

    print("\nClustering comparison metrics:")
    print(f"ARI {reference_label} vs VAE Leiden: {ari:.4f}")
    print(f"NMI {reference_label} vs VAE Leiden: {nmi:.4f}")
    # Visual comparison: same UMAP colored by VAE clusters and paper major labels

    embedding_config = load_config("config/ae_embedding.yml")
    n_top_genes = embedding_config["n_top_genes"]
    resolution = config["resolution"]

    title_base = f"n_top_genes={n_top_genes} | res={resolution}\nARI={ari:.4f} | NMI={nmi:.4f}"
    # Visual comparison: same UMAP colored by VAE clusters and paper major labels
    sc.pl.umap(
        adata,
        color=[leiden_key, reference_label],
        title=[f"AE Leiden — {title_base}", reference_label],
    )


if __name__ == "__main__":
    main()