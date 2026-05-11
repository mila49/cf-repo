from src.pipelines.embedding import EmbeddingPipeline


if __name__ == "__main__":
    pipeline = EmbeddingPipeline(
        config_path="config/embedding.yml"
    )

    pipeline.run()
