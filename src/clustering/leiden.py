import anndata as ad
import scanpy as sc
from typing import override

from . import Clustering


class LeidenClustering(Clustering):
    @override
    def fit_predict(self, adata: ad.AnnData, config: dict) -> str:
        use_rep = config.get("use_rep", "X_ae")
        n_neighbors = config.get("n_neighbors", 15)
        resolution = config.get("resolution", 0.5)
        
        print(f"Computing KNN graph using representation: {use_rep}...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
        
        print(f"Computing Leiden with resolution: {resolution}...")
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added="leiden_clusters",
            flavor="igraph"
        )

        return "leiden_clusters"