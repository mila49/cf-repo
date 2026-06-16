from .pipelines.embedding import VAEPipeline, AEPipeline, DAEPipeline
from .pipelines.clustering import LeidenClusteringPipeline

from src.clustering import LeidenClustering


if __name__ == "__main__":
    # vae_pipeline = VAEPipeline(
    #     config_path="vae_embedding.yml"
    # )
    ae_pipeline = AEPipeline(
        config_path="embeddings/ae_embedding.yml"
    )
    # dae_pipeline = DAEPipeline(
    #     config_path="dae_embedding.yml"
    # )

    # vae_pipeline.run()
    ae_pipeline.run()
    # dae_pipeline.run()