import os
import wandb
from dotenv import load_dotenv
from pathlib import Path

from src.pipelines.embedding.ae import AEPipeline  

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'mse',
        'goal': 'minimize'
    },
    'parameters': {
        'output_obsm_key': {
            'values': ["X_ae"]
        },
        'output_adata_path': {
            'values': ["adata_ae.h5ad"]
        },
        'output_model_path': {
            'values': ["ae_model.pt"]
        },
        'data_path': {
            'values': ["Dataset/raw/matrix.mtx"]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'n_top_genes': {
            'values': [5000, 7000, 9000]
        },
        'epochs': {
            'values': [30, 40, 50, 60]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'latent_dim': {
            'values': [16, 32, 64, 128]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }
    }
}

def sweep_worker():
    load_dotenv()
    
    run = wandb.init()
    
    pipeline = AEPipeline(config_path="ae_embedding.yml")
    
    pipeline.config.update(wandb.config)
    
    pipeline.setup_data()
    pipeline.setup_model()
    
    epochs = pipeline.config["epochs"]
    
    for epoch in range(epochs):
        pipeline.model.train()
        total_loss = 0
        
        for batch_x in pipeline.loader:
            batch_x = batch_x.to(pipeline.device)
            
            loss = pipeline.compute_loss(batch_x)
            
            pipeline.optimizer.zero_grad()
            loss.backward()
            pipeline.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
        avg_mse = total_loss / len(pipeline.dataset)
        print(f"Epoch {epoch + 1:03d} | MSE = {avg_mse:.4f}")
        
        wandb.log({"mse": avg_mse, "epoch": epoch + 1})
        
    run.finish()

if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

    print(os.environ.get("WANDB_PROJECT"))
    print(os.environ.get("WANDB_ENTITY"))
    
    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY")
    )
    
    wandb.agent(sweep_id, function=sweep_worker, count=3)
    
    api = wandb.Api()
    sweep_path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{sweep_id}"
    sweep = api.sweep(sweep_path)
    
    best_run = sweep.best_run()
    
    print("\n" + "="*50)
    print(f"Best Run: {best_run.name} ({best_run.id})")
    print(f"Best MSE: {best_run.summary.get('mse'):.5f}")
    print("\nBest hyperparameters:")
    for param, value in best_run.config.items():
        print(f"{param}: {value}")

    print("="*50)