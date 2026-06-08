from typing import Dict, override
from src.clustering import ClusterComparison
from ..main import Pipeline


class ClusterComparisonPipeline(Pipeline):
    """
    Pipeline for comparing clustering results against reference labels.

    Steps:
        1. Load clustering comparison configuration.
        2. Load clustered AnnData data.
        3. Compute clustering metrics (ARI, NMI).
        4. Generate cross-tabulation analysis.
        5. Create visualizations comparing clusters with reference labels.

    Note: This pipeline assumes that the input data is already clustered and contains
    both cluster labels and reference labels in adata.obs.
    """

    def __init__(self, config_file: str):
        """
        Initialize the ClusterComparisonPipeline.

        Args:
            config_file: Path to the comparison configuration YAML file.
        """
        super().__init__(config_file)
        self.comparator: ClusterComparison = None
        self.results: Dict = None


    def initialize_comparator(self) -> None:
        """
        Initialize the ClusterComparison object with the configuration file.
        """
        self.comparator = ClusterComparison(self.config)


    def run_comparison(self) -> Dict:
        """
        Run the complete cluster comparison analysis.

        Returns:
            Dictionary containing all analysis results.
        """
        self.results = self.comparator.compare()
        return self.results


    def get_metrics(self) -> Dict[str, float]:
        """
        Get the computed clustering metrics.

        Returns:
            Dictionary with ARI and NMI scores.
        """
        if self.results is None:
            raise ValueError("Pipeline has not been run yet. Call run() first.")
        return self.results.get("metrics")


    def get_crosstab(self) -> any:
        """
        Get the cross-tabulation results.

        Returns:
            DataFrame with cross-tabulation counts.
        """
        if self.results is None:
            raise ValueError("Pipeline has not been run yet. Call run() first.")
        return self.results.get("crosstab_counts")


    @override
    def run(self):
        """
        Execute the cluster comparison pipeline.
        """
        self.initialize_comparator()
        self.run_comparison()
