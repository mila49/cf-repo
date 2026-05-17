from src.clustering import Clustering
from src.pipelines.clustering import ClusteringPipeline


class LeidenClusteringPipeline(ClusteringPipeline):
    """
    Pipeline for Leiden clustering.
    """
    def __init__(self, config_file: str, clustering_method: Clustering):
        super().__init__(config_file, clustering_method)
