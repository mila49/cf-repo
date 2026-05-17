import os
from anndata import AnnData
import matplotlib.pyplot as plt
from typing import override
import pandas as pd
import wandb
from dotenv import load_dotenv
import scanpy as sc

from src.clustering import Clustering
from ..main import Pipeline
from src.utils.data_loader import load_data


class ClusteringPipeline(Pipeline):
    """
    Pipeline for clustering single-cell data.

    Steps:
        1. Load preprocessed data (AnnData object).
        2. Run clustering algorithm (e.g., Leiden, Louvain) and store cluster labels in adata.obs.
        3. Generate UMAP coordinates for visualization.
        4. Log results to Weights & Biases (W&B) including UMAP plots colored by cluster labels.

    Note: This pipeline assumes that the input data is already preprocessed and ready for clustering.
    """
    def __init__(self, config_file: str, clustering_method: Clustering):
        super().__init__(config_file)

        self.clustering_method: Clustering = clustering_method
        self.adata: AnnData = None


    def load_processed_data(self) -> None:
        """
        Load preprocessed data from the specified path in the configuration.
        The data is expected to be in a format compatible with AnnData (e.g., .h5ad or .mtx).
        """
        self.adata = load_data(self.root_dir / self.config["data_path"])


    def run_clustering(self) -> str:
        """
        Run the clustering algorithm.

        Returns:
            str: The key in adata.obs where the cluster labels are stored (e.g., "leiden_clusters").
        """
        return self.clustering_method.fit_predict(self.adata, self.config)


    def generate_umap(self) -> None:
        """
        Generate UMAP coordinates for visualization. The coordinates will be
        stored in adata.obsm["X_umap"].
        """
        sc.tl.umap(self.adata)


    def log_to_wandb(self, cluster_key: str) -> None:
        """
        Log UMAP plots to Weights & Biases (W&B) with clusters colored by the specified cluster key.

        Args:
            cluster_key: The key in adata.obs where the cluster labels are stored.
        """
        load_dotenv()
        run = wandb.init(
            entity=os.environ["WANDB_ENTITY"],
            project=os.environ["WANDB_PROJECT"],
            job_type="clustering",
            config=self.config
        )

        umap_coords = self.adata.obsm["X_umap"]
        df_plot = pd.DataFrame(umap_coords, columns=["UMAP_1", "UMAP_2"])
        df_plot["Cluster"] = self.adata.obs[cluster_key].astype(str)
        
        wb_table = wandb.Table(dataframe=df_plot)
        sc_plot = wandb.plot.scatter(wb_table, "UMAP_1", "UMAP_2", title=f"Clustering: {cluster_key}")
        run.log({"interactive_umap": sc_plot})

        fig, ax = plt.subplots(figsize=(6, 5))
        sc.pl.umap(self.adata, color=cluster_key, ax=ax, show=False)
        run.log({"scanpy_umap_image": wandb.Image(fig)})
        plt.close(fig)

        run.finish()


    @override
    def run(self):
        self.load_processed_data()
        cluster_key = self.run_clustering()
        self.generate_umap()
        self.log_to_wandb(cluster_key)