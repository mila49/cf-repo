from src.pipelines.embedding import VaeClusteringPipeline

if __name__ == "__main__":
    pipeline = VaeClusteringPipeline(
        config_path="config/embedding.yml",
        preprocessing_config_path="config/preprocessing.yml",
    )

    pipeline.run()
