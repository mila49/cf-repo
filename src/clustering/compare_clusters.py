import yaml
import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import Dict


class ClusterComparison:
    """
    Compares clustering results against a reference label using various metrics and visualizations.
    """

    def __init__(self, config: dict):
        """
        Initialize the ClusterComparison with configuration.

        Args:
            config: Dictionary containing the clustering comparison configuration.
        """
        self.config = config
        self.adata = None
        self.leiden_key = self.config.get("leiden_key")
        self.reference_label = self.config.get("reference_label")


    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load clustering/comparison parameters from a YAML config file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)


    def load_data(self) -> None:
        """Load the clustered AnnData file from the configured path."""
        adata_path = self.config.get("output_adata_path")
        self.adata = sc.read_h5ad(adata_path)
        self._validate_columns()


    def _validate_columns(self) -> None:
        """Validate that required columns exist in the AnnData object."""
        if self.leiden_key not in self.adata.obs.columns:
            raise ValueError(f"Column '{self.leiden_key}' not found in adata.obs.")

        if self.reference_label not in self.adata.obs.columns:
            raise ValueError(f"Column '{self.reference_label}' not found in adata.obs.")


    def get_cluster_counts(self) -> pd.DataFrame:
        """
        Get counts of cells for each leiden cluster.

        Returns:
            DataFrame with cell counts per cluster.
        """
        print("\nNumber of cells per VAE Leiden cluster:")
        counts = self.adata.obs[self.leiden_key].value_counts().sort_index()
        print(counts)
        return counts


    def get_reference_counts(self) -> pd.Series:
        """
        Get counts of cells for each reference label group.

        Returns:
            Series with cell counts per reference group.
        """
        print(f"\nNumber of cells per reference '{self.reference_label}' group:")
        counts = self.adata.obs[self.reference_label].value_counts()
        print(counts)
        return counts


    def get_crosstab_counts(self, output_path: str = "./crosstab_counts.csv") -> pd.DataFrame:
        """
        Get cross-tabulation between reference labels and leiden clusters.

        Args:
            output_path: Path where to save the crosstab CSV file.

        Returns:
            DataFrame with cross-tabulation counts.
        """
        counts = pd.crosstab(
            self.adata.obs[self.reference_label],
            self.adata.obs[self.leiden_key],
        )

        print(f"\nCounts: reference '{self.reference_label}' groups vs VAE Leiden clusters")
        print(counts)
        counts.to_csv(output_path)
        return counts


    def get_crosstab_percentages(self) -> pd.DataFrame:
        """
        Get cross-tabulation percentages by reference group.

        Returns:
            DataFrame with percentages normalized by reference group.
        """
        percentages = pd.crosstab(
            self.adata.obs[self.reference_label],
            self.adata.obs[self.leiden_key],
            normalize="index",
        ) * 100

        print(f"\nPercentages by reference '{self.reference_label}' group:")
        print(percentages.round(1))
        return percentages


    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute clustering similarity metrics.

        Returns:
            Dictionary containing ARI and NMI scores.
        """
        ari = adjusted_rand_score(
            self.adata.obs[self.reference_label],
            self.adata.obs[self.leiden_key],
        )

        nmi = normalized_mutual_info_score(
            self.adata.obs[self.reference_label],
            self.adata.obs[self.leiden_key],
        )

        print("\nClustering comparison metrics:")
        print(f"ARI {self.reference_label} vs VAE Leiden: {ari:.4f}")
        print(f"NMI {self.reference_label} vs VAE Leiden: {nmi:.4f}")

        return {"ARI": ari, "NMI": nmi}


    def _get_title_base(self, metrics: Dict[str, float]) -> str:
        """
        Generate title base string with metrics and parameters.

        Args:
            metrics: Dictionary containing ARI and NMI scores.

        Returns:
            Formatted title string.
        """
        embedding_config = self._load_config("config/embeddings/ae_embedding.yml")
        n_top_genes = embedding_config.get("n_top_genes")
        resolution = self.config.get("resolution")
        ari = metrics["ARI"]
        nmi = metrics["NMI"]

        return f"n_top_genes={n_top_genes} | res={resolution}\nARI={ari:.4f} | NMI={nmi:.4f}"


    def plot_comparison(self, show: bool = True, reference_labels: list = None) -> None:
        """
        Create visual comparison of clusters using UMAP.

        Args:
            show: Whether to display the plot.
            reference_labels: List of reference label columns to compare. 
                            If None, uses self.reference_label only.
        """
        if reference_labels is None:
            reference_labels = [self.reference_label]
        
        # Plot for each reference label
        for ref_label in reference_labels:
            if ref_label not in self.adata.obs.columns:
                print(f"Warning: Reference label '{ref_label}' not found in adata.obs. Skipping.")
                continue
            
            # Compute metrics for this reference label
            metrics = self._compute_metrics_for_label(ref_label)
            title_base = self._get_title_base_for_label(metrics, ref_label)

            sc.pl.umap(
                self.adata,
                color=[self.leiden_key, ref_label],
                title=[f"AE Leiden — {title_base}", ref_label],
                show=show,
            )
    
    
    def _compute_metrics_for_label(self, reference_label: str) -> Dict[str, float]:
        """
        Compute clustering similarity metrics for a specific reference label.

        Args:
            reference_label: Name of the reference label column.

        Returns:
            Dictionary containing ARI and NMI scores.
        """
        ari = adjusted_rand_score(
            self.adata.obs[reference_label],
            self.adata.obs[self.leiden_key],
        )

        nmi = normalized_mutual_info_score(
            self.adata.obs[reference_label],
            self.adata.obs[self.leiden_key],
        )

        return {"ARI": ari, "NMI": nmi}
    
    
    def _get_title_base_for_label(self, metrics: Dict[str, float], reference_label: str) -> str:
        """
        Generate title base string with metrics and parameters for a specific reference label.

        Args:
            metrics: Dictionary containing ARI and NMI scores.
            reference_label: Name of the reference label.

        Returns:
            Formatted title string.
        """
        embedding_config = self._load_config("config/embeddings/ae_embedding.yml")
        n_top_genes = embedding_config.get("n_top_genes")
        resolution = self.config.get("resolution")
        ari = metrics["ARI"]
        nmi = metrics["NMI"]

        return f"n_top_genes={n_top_genes} | res={resolution}\n{reference_label}: ARI={ari:.4f} | NMI={nmi:.4f}"


    def compare(self) -> Dict:
        """
        Run the complete comparison analysis.

        Returns:
            Dictionary containing all results.
        """
        self.load_data()
        
        results = {
            "cluster_counts": self.get_cluster_counts(),
            "reference_counts": self.get_reference_counts(),
            "crosstab_counts": self.get_crosstab_counts(),
            "crosstab_percentages": self.get_crosstab_percentages(),
            "metrics": self.compute_metrics(),
        }

        # Compute and print metrics for both major and minor
        print("\n" + "="*80)
        print("CLUSTERING METRICS FOR ALL REFERENCE LABELS")
        print("="*80)
        
        for ref_label in ["major", "minor"]:
            if ref_label in self.adata.obs.columns:
                metrics = self._compute_metrics_for_label(ref_label)
                print(f"\n{ref_label.upper()} annotation:")
                print(f"  ARI: {metrics['ARI']:.4f}")
                print(f"  NMI: {metrics['NMI']:.4f}")
        
        print("\n" + "="*80)
        
        # Plot comparisons for both major and minor reference labels
        self.plot_comparison(reference_labels=["major", "minor"])

        return results