import yaml
import scanpy as sc
import anndata as ad


def load_config(config_path: str) -> dict:
    """Load clustering parameters from a YAML config file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    # Allow AnnData to save nullable string columns
    ad.settings.allow_write_nullable_strings = True

    # Load clustering config
    config = load_config("config/leiden_clustering.yml")

    # Load AnnData file containing VAE embeddings and metadata
    adata = sc.read_h5ad(config["input_adata_path"])

    # Get config parameters
    embedding_key = config["embedding_key"]
    leiden_key = config["leiden_key"]
    n_neighbors = config["n_neighbors"]
    resolution = config["resolution"]

    # Check that the selected embedding exists
    if embedding_key not in adata.obsm.keys():
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm.")

    # Build nearest-neighbor graph using the VAE latent embeddings
    sc.pp.neighbors(
        adata,
        use_rep=embedding_key,
        n_neighbors=n_neighbors,
    )

    # Run Leiden clustering on the neighbor graph
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=leiden_key,
        flavor="igraph"
    )

    # Compute UMAP coordinates for visualization
    sc.tl.umap(adata)

    # Save AnnData with clusters and UMAP
    adata.write(config["output_adata_path"])

    print(f"Saved clustered AnnData to: {config['output_adata_path']}")

    # Plot VAE clusters
    sc.pl.umap(adata, color=leiden_key)


if __name__ == "__main__":
    main()