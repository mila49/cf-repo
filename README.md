# CF-Repo: Single-Cell RNA-seq Analysis Pipeline

A complete pipeline for single-cell RNA sequencing (scRNA-seq) analysis using deep learning models (Autoencoder, Variational Autoencoder, Denoising Autoencoder) for dimensionality reduction and clustering.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Basic Training](#basic-training)
  - [Hyperparameter Search](#hyperparameter-search)
  - [Running Specific Models](#running-specific-models)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.8 or higher
- ~1GB disk space for data
- (Optional) Weights & Biases account for experiment tracking

---

## Installation

### Step 1: Clone the repository
```powershell
git clone https://github.com/mila49/cf-repo
cd cf-repo
```

### Step 2: Create and activate virtual environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate virtual environment (Windows CMD)
.\.venv\Scripts\activate.bat

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate
```

### Step 3: Install dependencies
```powershell
pip install -r requirements.txt
```

---

## Data Setup

### Step 1: Prepare your data files
You need the following files in the root directory:
- `matrix.mtx` - Gene expression matrix (~1GB)
- `genes.csv` - Gene names/IDs
- `metadata.csv` - Cell metadata

**Note:** If your data is in a different location, update the `data_path` in the config files.

### Step 2: Verify data files
```powershell
# Check if required files exist
ls matrix.mtx, genes.csv, metadata.csv
```

---

## Configuration

### Step 1: Set up Weights & Biases (Optional but recommended)

Create a `.env` file from the example:
```powershell
cp .env.example .env
```

Edit `.env` and add your credentials:
```bash
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_ENTITY="your-wandb-username-or-team"
export WANDB_PROJECT="cf-repo"
```

### Step 2: Configure model parameters

Edit configuration files in `config/embeddings/`:
- `ae_embedding.yml` - Autoencoder settings
- `vae_embedding.yml` - VAE settings
- `dae_embedding.yml` - Denoising Autoencoder settings

Example configuration (`config/embeddings/ae_embedding.yml`):
```yaml
wb_sweep: false  # Set to true for hyperparameter search
data_path: "Dataset/raw/matrix.mtx"

parameters:
  batch_size: [64]
  latent_dim: [30]
  epochs: [40]
  learning_rate: [0.001]
  dropout_rate: [0.3]

n_top_genes: 5000
output_obsm_key: "X_ae"
output_adata_path: "adata_ae.h5ad"
output_model_path: "ae_model.pt"
device: "cpu"  # or "cuda" if GPU available
```

---

## Usage

### Basic Training

#### Train Autoencoder (AE)
```powershell
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Run AE training
python -m src.main
```

#### Train VAE or DAE
Edit `src/main.py` to uncomment the desired model:
```python
if __name__ == "__main__":
    # Uncomment the model you want to train:
    
    # vae_pipeline = VAEPipeline(config_path="embeddings/vae_embedding.yml")
    # vae_pipeline.run()
    
    ae_pipeline = AEPipeline(config_path="embeddings/ae_embedding.yml")
    ae_pipeline.run()
    
    # dae_pipeline = DAEPipeline(config_path="embeddings/dae_embedding.yml")
    # dae_pipeline.run()
```

Then run:
```powershell
python -m src.main
```

---

### Hyperparameter Search

#### Method 1: Built-in Grid Search (Recommended)

**Step 1:** Edit `config/embeddings/ae_embedding.yml`:
```yaml
wb_sweep: true  # Enable hyperparameter search

parameters:
  batch_size:
    - 64
    - 128
  latent_dim:
    - 30
    - 40
    - 50
  epochs:
    - 40
    - 50
  learning_rate:
    - 0.001
    - 0.0005
  dropout_rate:
    - 0.3
    - 0.4
```

**Step 2:** Run the pipeline (it will automatically run all combinations):
```powershell
python -m src.main
```

**Step 3:** View results in your Weights & Biases dashboard

---

#### Method 2: Bayesian Optimization

**Step 1:** Edit `src/pipelines/embedding/ae_hp_search.py` to configure search space:
```python
sweep_config = {
    'method': 'bayes',  # or 'random', 'grid'
    'metric': {
        'name': 'mse',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'latent_dim': {
            'values': [16, 32, 64, 128]
        },
        # Add more parameters...
    }
}
```

**Step 2:** Set the number of trials:
```python
# In ae_hp_search.py, modify this line:
wandb.agent(sweep_id, function=sweep_worker, count=10)  # Run 10 trials
```

**Step 3:** Run the hyperparameter search:
```powershell
python -m src.pipelines.embedding.ae_hp_search
```

**Step 4:** The script will print the best hyperparameters:
```
==================================================
Best Run: <run-name> (<run-id>)
Best MSE: 0.12345
Best hyperparameters:
batch_size: 64
latent_dim: 50
learning_rate: 0.001
...
==================================================
```

---

### Running Specific Models

#### Run only VAE:
```powershell
python -c "from src.pipelines.embedding import VAEPipeline; VAEPipeline('embeddings/vae_embedding.yml').run()"
```

#### Run only DAE:
```powershell
python -c "from src.pipelines.embedding import DAEPipeline; DAEPipeline('embeddings/dae_embedding.yml').run()"
```

#### Run clustering pipeline:
```powershell
python -m src.clustering.leiden
```

---

## Project Structure

```
cf-repo/
├── config/                          # Configuration files
│   ├── embeddings/                  # Model-specific configs
│   │   ├── ae_embedding.yml        # Autoencoder config
│   │   ├── vae_embedding.yml       # VAE config
│   │   └── dae_embedding.yml       # DAE config
│   ├── clustering/                  # Clustering configs
│   ├── preprocessing.yml            # Data preprocessing settings
│   └── graph.yml                    # Graph construction settings
│
├── src/                             # Source code
│   ├── models/                      # Model architectures
│   │   ├── ae.py                   # Autoencoder
│   │   ├── vae.py                  # Variational Autoencoder
│   │   └── dae.py                  # Denoising Autoencoder
│   │
│   ├── pipelines/                   # Training pipelines
│   │   ├── main.py                 # Base Pipeline class
│   │   ├── embedding/              # Embedding pipelines
│   │   │   ├── main.py            # EmbeddingPipeline base
│   │   │   ├── ae.py              # AE training pipeline
│   │   │   ├── vae.py             # VAE training pipeline
│   │   │   ├── dae.py             # DAE training pipeline
│   │   │   └── ae_hp_search.py    # Hyperparameter search
│   │   └── clustering/             # Clustering pipelines
│   │
│   ├── utils/                       # Utility functions
│   │   ├── data_loader.py          # Data loading and preprocessing
│   │   └── datasets.py             # PyTorch dataset classes
│   │
│   └── main.py                      # Main entry point
│
├── matrix.mtx                       # Gene expression data (~1GB)
├── genes.csv                        # Gene names/IDs
├── metadata.csv                     # Cell metadata
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Outputs

After running the pipeline, you will find:

### Model Files
- `ae_model.pt` / `vae_model.pt` / `dae_model.pt` - Trained model weights
- `adata_ae.h5ad` / `adata_vae.h5ad` - AnnData objects with embeddings

### AnnData Structure
The output `.h5ad` files contain:
- `adata.X` - Log-normalized gene expression (highly variable genes only)
- `adata.obsm['X_ae']` - Autoencoder embeddings (latent representation)
- `adata.obs` - Cell metadata
- `adata.var` - Gene metadata
- `adata.layers['counts']` - Raw count data
- `adata.layers['lognorm']` - Log-normalized full-gene matrix

### Weights & Biases Dashboard
View your experiments at: `https://wandb.ai/<entity>/<project>`
- Loss curves over epochs
- Hyperparameter comparisons
- Best model configurations

---

## Troubleshooting

### Issue: Virtual environment won't activate
**Solution (PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Issue: `ModuleNotFoundError`
**Solution:**
```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: matrix.mtx`
**Solution:**
- Ensure `matrix.mtx`, `genes.csv`, and `metadata.csv` are in the root directory
- Or update `data_path` in your config file:
```yaml
data_path: "path/to/your/matrix.mtx"
```

### Issue: CUDA out of memory
**Solution:**
- Reduce batch size in config:
```yaml
parameters:
  batch_size: [32]  # Instead of 64
```
- Or use CPU:
```yaml
device: "cpu"
```

### Issue: Weights & Biases authentication error
**Solution:**
```powershell
# Login to wandb
wandb login

# Or set API key directly
$env:WANDB_API_KEY="your-api-key"
```

### Issue: Import errors when running `python -m src.main`
**Solution:**
```powershell
# Run from the repository root directory
cd cf-repo
python -m src.main
```

---

## Advanced Usage

### Custom Model Configuration

Create a new config file in `config/embeddings/`:
```yaml
wb_sweep: false
data_path: "Dataset/raw/matrix.mtx"

parameters:
  batch_size: [128]
  latent_dim: [64]
  epochs: [100]
  learning_rate: [0.0001]
  dropout_rate: [0.2]

n_top_genes: 7000
output_obsm_key: "X_ae_custom"
output_adata_path: "adata_ae_custom.h5ad"
output_model_path: "ae_model_custom.pt"
device: "cuda"
```

### Using GPU
```yaml
device: "cuda"  # or "cuda:0" for specific GPU
```

### Preprocessing Options
Edit `config/preprocessing.yml`:
```yaml
n_top_genes: 5000      # Number of highly variable genes
min_genes: 200         # Min genes per cell
min_cells: 3           # Min cells expressing a gene
max_genes: 6000        # Max genes per cell (filter doublets)
max_pct_mt: 15.0       # Max mitochondrial %
target_sum: 10000      # Normalization target
hvg_flavor: "seurat"   # HVG selection method
```

---

## Quick Reference

### Common Commands
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Train model
python -m src.main

# Hyperparameter search
python -m src.pipelines.embedding.ae_hp_search

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# View data info
python -c "import anndata as ad; adata = ad.read_h5ad('adata_ae.h5ad'); print(adata)"
```

---

## Appendix: AnnData Object Reference

**AnnData** is a Python library for handling annotated data matrices used in single-cell analysis.

### Key Components

#### 1. **X** (Expression Matrix)
```python
adata.X  # Gene expression matrix (cells × genes)
```
- Main data matrix containing gene expression measurements
- Shape: (n_cells, n_genes)

#### 2. **obs** (Cell Metadata)
```python
adata.obs  # DataFrame with cell metadata
```
- Cell type annotations, quality metrics, batch information, clustering results

#### 3. **var** (Gene Metadata)
```python
adata.var  # DataFrame with gene metadata
```
- Gene symbols, IDs, expression statistics, quality metrics

#### 4. **obsm** (Embeddings)
```python
adata.obsm['X_ae']   # Autoencoder embeddings
adata.obsm['X_pca']  # PCA coordinates
adata.obsm['X_umap'] # UMAP coordinates
```
- Low-dimensional representations of cells

#### 5. **layers** (Alternative Data Views)
```python
adata.layers['counts']   # Raw count data
adata.layers['lognorm']  # Log-normalized data
```
- Different preprocessing versions of the expression data

### Basic Operations
```python
import anndata as ad

# Load data
adata = ad.read_h5ad('adata_ae.h5ad')

# View structure
print(adata)

# Access embeddings
embeddings = adata.obsm['X_ae']

# Access metadata
cell_types = adata.obs['cell_type']

# Save data
adata.write('output.h5ad')
```

---

**Happy analyzing! 🧬**
