# CF-Repo: Single-Cell RNA-seq Analysis Pipeline

## 📋 Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Basic Training](#basic-training)
  - [Hyperparameter Search](#hyperparameter-search)
  - [Running Specific Models](#running-specific-models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Outputs](#outputs)

---

## Introduction

Cystic fibrosis (CF) is a lethal genetic disease that affects the respiratory system and causes progressive damage to the airway epithelium. This pathology alters both the relative abundance of epithelial cell populations and their gene expression profiles.

The study by Carraro et al. primarily aimed to identify the epithelial cell types and subtypes present in the airways to determine how their composition and transcriptional states vary between cystic fibrosis donors and control donors. To carry out this analysis, the authors followed a classic single-cell RNA sequencing (single-cell RNA-seq) analysis workflow.

The main objective of this project is to evaluate the feasibility of implementing the core pipeline of this analysis using advanced Deep Learning techniques, exploring whether these tools can optimize or provide new perspectives to the traditional methodology used in the reference study.

## Objectives

This project focuses on reproducing and enhancing the cell-clustering results of the reference study through advanced Deep Learning approaches. To achieve this, the workflow will shift from traditional methods by utilizing a larger number of genes instead of restricting the analysis solely to highly variable genes. While maintaining a strictly unsupervised clustering framework, the project aims to improve the separation of finer, more nuanced cell subpopulations and significantly increase the biological interpretability of the resulting clusters.


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

## Experiments

This section summarizes the main experimental workflows implemented under [src/pipelines](src/pipelines). Each experiment is designed to test a specific hypothesis about representation learning, graph refinement, or clustering quality for single-cell data.

### Autoencoder embedding experiment
A standard autoencoder is used to compress high-dimensional gene-expression data into a compact latent space while preserving the essential structure of the input. The workflow in [src/pipelines/embedding/ae.py](src/pipelines/embedding/ae.py) trains the model on the preprocessed AnnData object, minimizes reconstruction error with Adam, and stores the resulting embedding for downstream clustering.

### Variational autoencoder experiment
This experiment evaluates whether a variational autoencoder can produce a more regularized and structured latent space by encouraging the embeddings to follow a Gaussian prior. In [src/pipelines/embedding/vae.py](src/pipelines/embedding/vae.py), the model is trained with a reconstruction loss plus KL regularization, and the posterior mean is used as the embedding representation.

### Denoising autoencoder experiment
The denoising autoencoder experiment tests whether forcing the model to reconstruct corrupted inputs helps it learn more robust features from noisy single-cell data. The workflow in [src/pipelines/embedding/dae.py](src/pipelines/embedding/dae.py) applies masked inputs during training, optimizes reconstruction error, and extracts latent embeddings from the encoder.

### Graph refinement experiment
This experiment investigates whether refining the initial embeddings with a graph-based neural model can improve local neighborhood structure and make the latent space more suitable for clustering. In [src/pipelines/embedding/graph_refinement.py](src/pipelines/embedding/graph_refinement.py), a k-nearest-neighbor graph is built from the initial embeddings and a graph attention refiner is trained to produce improved representations.

### Autoencoder hyperparameter search
This experiment explores how sensitive the autoencoder is to key hyperparameters such as latent dimension, learning rate, batch size, and dropout. The sweep in [src/pipelines/embedding/ae_hp_search.py](src/pipelines/embedding/ae_hp_search.py) evaluates multiple configurations, trains each model with early stopping, and logs the results for comparison.

### Variational autoencoder hyperparameter search
This search examines how the balance between reconstruction quality and KL regularization affects the quality of the learned representations. The workflow in [src/pipelines/embedding/vae_hp_search.py](src/pipelines/embedding/vae_hp_search.py) evaluates different VAE configurations and tracks validation loss together with clustering-related metrics.

### Denoising autoencoder hyperparameter search
This experiment studies how denoising strength and optimization choices influence the generalization ability of the denoising autoencoder on noisy gene-expression data. The sweep in [src/pipelines/embedding/dae_hp_search.py](src/pipelines/embedding/dae_hp_search.py) searches over multiple parameter settings and assesses the latent representations they produce.

### Clustering search experiment
This experiment evaluates whether different graph construction settings and clustering resolutions lead to more meaningful and stable partitions of the data. The workflow in [src/pipelines/clustering/clustering_search.py](src/pipelines/clustering/clustering_search.py) compares multiple Leiden and Louvain parameter combinations over the learned embeddings and reports clustering metrics.

### Leiden clustering experiment
This experiment tests whether Leiden clustering can recover meaningful cell communities from the learned embedding space when the neighborhood graph and resolution are properly tuned. The pipeline in [src/pipelines/clustering/leiden.py](src/pipelines/clustering/leiden.py) runs Leiden clustering and produces cluster assignments for downstream interpretation.

### Annotation and signature scoring experiment
This experiment assesses whether biological signatures can serve as an external validation signal for the learned embeddings and clusters. The workflow in [src/pipelines/annotations/score_cell_types.py](src/pipelines/annotations/score_cell_types.py) loads signature definitions, scores cells against them, and compares the resulting biological signal with the clustering output.

---

## Results

This section summarizes the outcomes of applying the embedding, clustering, and graph-refinement pipelines above to the reference dataset.

### Metrics

Model configurations were compared using a composite score:

```
Composite Score = Base Quality × Cluster Resolution × Cluster Balance
```

which jointly captures three questions: are the clusters well separated, did the model avoid collapsing clusters together, and are the cluster sizes reasonable?

| Composite score | AE | DAE |
|---|---|---|
| Without GAT embeddings | **0.28** | 0.17 |
| With GAT embeddings | 0.26 | 0.22 |

The standard Autoencoder (AE) consistently outperformed the Denoising Autoencoder (DAE) across the explored hyperparameter search space, both with and without graph-based refinement.

Comparing the best configuration against the ground-truth cell-type labels (Adjusted Rand Index and Normalized Mutual Information, both bounded between 0 and 1):

| Best config | ARI | NMI |
|---|---|---|
| Major cell types | 0.515 | 0.483 |
| Minor cell types | 0.331 | 0.419 |

Performance is noticeably stronger at the coarse ("major") cell-type level than at the fine-grained ("minor") subtype level. This is a **clustering granularity** effect: the finer the distinction the model is asked to recover, the harder it becomes to separate biologically similar subtypes within the learned latent space.

The best configuration found during the hyperparameter sweep (`n_top_genes=9000`, `latent_dim=64`, `learning_rate=3e-5`, `batch_size=128`, `dropout_rate=0.3`, `leiden_resolution=0.35`, `n_neighbors=50`) reached the composite score of 0.284 reported above.

### Clustering

At the **major cell-type level**, unsupervised Leiden clustering on the AE embedding produced 11 clusters that closely mirror the true major cell-type structure (Basal, Ciliated, Secretory, etc.), without the model ever seeing the ground-truth labels during training. An independent, label-free biological signature-scoring step — assigning each cell a predicted type based on marker-gene expression — reconstructs a very similar structure, providing external validation that the AE embedding captures real biological signal rather than noise.

At the **minor cell-type level**, Leiden proposes 20 clusters, but both clustering and signature-scoring recover the finer subtypes less cleanly, consistent with the ARI/NMI drop reported above.

### Interpretability

To probe what the model has learned, embeddings were refined with a **Graph Attention Network (GAT)** (see [src/models/gat_refiner.py](src/models/gat_refiner.py) and [src/pipelines/embedding/graph_refinement.py](src/pipelines/embedding/graph_refinement.py)):

- A **k-nearest-neighbor graph is built over cells** (not genes), connecting each cell to its `knn_k` nearest neighbors (default 15) in the AE's latent space.
- A two-layer GAT is trained **self-supervised**, with no labels involved: its only objective is to reconstruct each cell's own input embedding from its neighbors' features, learning in the process how much attention to assign to each neighboring cell.
- The resulting per-edge attention weights are averaged by grouping cells according to their cell-type label, producing a cluster-to-cluster attention matrix.

At the **major cell-type level**, attention is fairly uniform (~0.12–0.13) across all cluster pairs, indicating the model treats broad cell types as roughly equally connected. At the **minor subtype level**, attention becomes sparse and selective — several cluster pairs show zero attention, while others (e.g. Ciliated2 → Secretory5, 0.190) stand out clearly — showing the model differentiates biologically meaningful relationships once given fine-enough resolution to work with.

### Conclusions & Future Work

- A simple AE-based unsupervised pipeline is feasible and biologically meaningful, but its performance is limited by representation quality and clustering granularity.
- The standard AE consistently outperformed the DAE across the explored search space.
- The pipeline reproduces the reference study's clustering results while making them more interpretable, thanks to a wider gene context and biological signature validation.
- The GAT attention analysis shows this interpretability gain is resolution-dependent, becoming more selective and biologically specific at finer clustering resolutions.

Future work includes: a broader/higher-resolution hyperparameter search, testing whether larger gene sets continue to help or start adding noise, trying additional clustering algorithms beyond Leiden/Louvain, building a stronger interpretability pipeline grounded more directly in biological marker expression, and extending to a **Variational Autoencoder (VAE)** to impose a more structured latent space — directly addressing the granularity limitation seen at the minor subtype level.

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
