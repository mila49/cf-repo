from src.pipelines.clustering import ClusterComparisonPipeline


def main():
    """Run cluster comparison analysis using the ClusterComparisonPipeline."""
    pipeline = ClusterComparisonPipeline("clustering/compare_clusters.yml")
    pipeline.run()


if __name__ == "__main__":
    main()