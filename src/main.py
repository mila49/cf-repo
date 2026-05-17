from .pipelines.embedding import VAEPipeline, AEPipeline
from .pipelines.clustering import LeidenClusteringPipeline

from src.clustering import LeidenClustering


if __name__ == "__main__":
    # vae_pipeline = VAEPipeline(
    #     config_path="vae_embedding.yml"
    # )
    ae_pipeline = AEPipeline(
        config_path="ae_embedding.yml"
    )

    # vae_pipeline.run()
    ae_pipeline.run()