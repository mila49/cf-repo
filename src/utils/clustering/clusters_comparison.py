import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


CLUSTERING_CONFIG_PATH = "config/clustering.yml"
EMBEDDING_CONFIG_PATH = "config/embedding.yml"
REPRESENTATION_KEY = "X_vae"
TABLE_OUTPUT_PATH = "./src/utils/clustering/vae_contingency_counts_table.png"


def load_config(config_path: str) -> dict:
    """Load parameters from a YAML config file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def validate_obs_columns(adata, columns: list[str]) -> None:
    """Check that required columns exist in adata.obs."""
    missing = [col for col in columns if col not in adata.obs.columns]

    if missing:
        raise ValueError(
            f"Missing columns in adata.obs: {missing}. "
            f"Available columns are: {list(adata.obs.columns)}"
        )


def make_contingency_tables(
    adata,
    reference_label: str,
    cluster_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create counts and row-normalized percentage contingency tables."""
    counts = pd.crosstab(
        adata.obs[reference_label],
        adata.obs[cluster_label],
    )

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    return counts, percentages


def compute_clustering_metrics(
    adata,
    reference_label: str,
    cluster_label: str,
    representation_key: str = REPRESENTATION_KEY,
) -> dict[str, float]:
    """Compute ARI, NMI, and silhouette score."""
    reference = adata.obs[reference_label].astype(str)
    clusters = adata.obs[cluster_label].astype(str)

    ari = adjusted_rand_score(reference, clusters)
    nmi = normalized_mutual_info_score(reference, clusters)

    if representation_key not in adata.obsm:
        raise ValueError(
            f"Representation '{representation_key}' not found in adata.obsm. "
            f"Available representations are: {list(adata.obsm.keys())}"
        )

    if clusters.nunique() < 2:
        silhouette = float("nan")
        print("\nSilhouette score could not be calculated because there is only one cluster.")
    else:
        silhouette = silhouette_score(adata.obsm[representation_key], clusters)

    return {
        "ari": ari,
        "nmi": nmi,
        "silhouette": silhouette,
    }


def plot_dataframe_table(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    figsize_width: float = 12,
    row_height: float = 0.45,
    font_size: int = 9,
) -> None:
    """Plot a pandas DataFrame as a styled table figure."""
    fig_height = row_height * len(df) + 2

    fig, ax = plt.subplots(figsize=(figsize_width, fig_height))
    ax.axis("off")

    table = pd.plotting.table(
        ax,
        df,
        loc="center",
        cellLoc="center",
        colWidths=[0.08] * len(df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("lightgray")
        cell.set_linewidth(0.6)

        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAF2")

        if col == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F5F5F5")

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"\nSaved contingency table plot as '{output_path}'")


def print_summary(
    adata,
    leiden_key: str,
    reference_label: str,
    counts: pd.DataFrame,
    percentages: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    """Print cluster counts, contingency tables, and metrics."""
    print("\nNumber of cells per VAE Leiden cluster:")
    print(adata.obs[leiden_key].value_counts().sort_index())

    print(f"\nNumber of cells per reference '{reference_label}' group:")
    print(adata.obs[reference_label].value_counts())

    print(f"\nCounts: reference '{reference_label}' groups vs VAE Leiden clusters")
    print(counts)

    print(f"\nPercentages by reference '{reference_label}' group:")
    print(percentages.round(1))

    print("\nClustering comparison metrics:")
    print(f"ARI {reference_label} vs VAE Leiden: {metrics['ari']:.4f}")
    print(f"NMI {reference_label} vs VAE Leiden: {metrics['nmi']:.4f}")
    print(f"Silhouette score VAE Leiden: {metrics['silhouette']:.4f}")


def main() -> None:
    clustering_config = load_config(CLUSTERING_CONFIG_PATH)["vae"]
    embedding_config = load_config(EMBEDDING_CONFIG_PATH)

    adata = sc.read_h5ad(clustering_config["output_adata_path"])

    leiden_key = clustering_config["leiden_key"]
    reference_label = clustering_config["reference_label"]
    resolution = clustering_config["resolution"]
    n_top_genes = embedding_config["n_top_genes"]

    validate_obs_columns(adata, [leiden_key, reference_label])

    counts, percentages = make_contingency_tables(
        adata=adata,
        reference_label=reference_label,
        cluster_label=leiden_key,
    )

    metrics = compute_clustering_metrics(
        adata=adata,
        reference_label=reference_label,
        cluster_label=leiden_key,
        representation_key=REPRESENTATION_KEY,
    )

    print_summary(
        adata=adata,
        leiden_key=leiden_key,
        reference_label=reference_label,
        counts=counts,
        percentages=percentages,
        metrics=metrics,
    )

    title_base = (
        f"n_top_genes={n_top_genes} | res={resolution}\n"
        f"ARI={metrics['ari']:.4f} | "
        f"NMI={metrics['nmi']:.4f} | "
        f"silhouette={metrics['silhouette']:.4f}"
    )

    sc.pl.umap(
        adata,
        color=[leiden_key, reference_label],
        title=[
            f"VAE Leiden: {title_base}",
            reference_label,
        ],
    )

    plot_dataframe_table(
        df=counts,
        title=f"Contingency table: {reference_label} vs {leiden_key}",
        output_path=TABLE_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
