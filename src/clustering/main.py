from abc import ABC, abstractmethod
import anndata as ad


class Clustering(ABC):
    """
    Abstract base class for clustering algorithms.
    Each clustering method should implement the fit_predict method, which takes an AnnData object
    and a configuration dictionary, and returns the key for the cluster labels in the AnnData object.
    """
    @abstractmethod
    def fit_predict(self, adata: ad.AnnData, config: dict) -> str:
        """
        Fits the clustering algorithm and predicts cluster labels for the given data.

        Args:
            adata: AnnData object containing the data to cluster.
            config: Configuration parameters for the clustering algorithm.

        Returns:
            str: The key for the cluster labels in the AnnData object.
        """
        pass