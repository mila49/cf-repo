# Deep Learning-Based Cell Type Clustering in Cystic Fibrosis Airway Epithelium

> **Project Type**: Postgraduate Research - Advanced Deep Learning for Bioinformatics  
> **Domain**: Single-Cell RNA Sequencing (scRNA-seq) Analysis  
> **Objective**: Unsupervised cell-type discovery using Autoencoders and Graph Attention Networks

---

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Biological Context](#11-biological-context)
  - [1.2 Motivation](#12-motivation)
  - [1.3 Objectives](#13-objectives)
- [2. Methodology](#2-methodology)
  - [2.1 System Architecture Overview](#21-system-architecture-overview)
  - [2.2 Data Preprocessing](#22-data-preprocessing)
  - [2.3 Dimensionality Reduction Models](#23-dimensionality-reduction-models)
  - [2.4 Graph Attention Network Refinement](#24-graph-attention-network-refinement)
  - [2.5 Clustering](#25-clustering)
  - [2.6 Hyperparameter Optimisation](#26-hyperparameter-optimisation)
  - [2.7 Evaluation Strategy](#27-evaluation-strategy)
  - [2.8 Cell Type Annotation](#28-cell-type-annotation)
- [3. Project Structure](#3-project-structure)
- [4. Installation and Setup](#4-installation-and-setup)
- [5. Usage](#5-usage)
  - [5.1 Single Model Training](#51-single-model-training)
  - [5.2 Hyperparameter Search](#52-hyperparameter-search)
  - [5.3 Visualization and Reporting](#53-visualization-and-reporting)
- [6. Configuration Reference](#6-configuration-reference)
- [7. Outputs](#7-outputs)
- [8. References](#8-references)

---

## Abstract

Cystic fibrosis (CF) is a lethal genetic disease that causes progressive damage to the airway epithelium. Understanding the altered cellular composition in CF airways requires accurate identification of epithelial cell populations from single-cell RNA sequencing data. This project implements a Deep Learning pipeline that replaces traditional dimensionality reduction techniques (PCA) with autoencoder-based models coupled with Graph Attention Networks (GAT) to produce biologically meaningful cell embeddings. The system operates in a fully unsupervised manner, using Leiden community detection for clustering and a composite scoring framework for model selection. The pipeline supports three autoencoder variants-standard (AE), variational (VAE), and denoising (DAE)-each followed by optional GAT-based embedding refinement, with automated hyperparameter search across the full parameter space including clustering parameters.

---

## 1. Introduction

### 1.1 Biological Context

Cystic fibrosis is caused by mutations in the CFTR gene, which encodes a chloride channel protein critical for maintaining airway surface liquid homeostasis. The disease alters both the relative abundance of epithelial cell populations and their gene expression profiles. Carraro et al. (2021) performed single-cell RNA-seq on CF and control donor airways, identifying diverse epithelial cell types (Basal, Secretory, Ciliated, FOXN4+, and Neuroendocrine) and their subtypes through a traditional analysis pipeline consisting of PCA, kNN graph construction, and Louvain clustering.

### 1.2 Motivation

The traditional scRNA-seq workflow relies on:
1. Filtering to highly variable genes (HVGs) - typically 2,000–3,000 genes
2. Linear dimensionality reduction via PCA
3. Graph-based clustering (Louvain/Leiden)

This project explores whether deep learning-based non-linear dimensionality reduction can:
- Operate on a broader gene set (5,000–10,000 genes) rather than restricting to HVGs
- Learn richer latent representations that preserve fine-grained biological structure
- Improve separation of rare and closely related cell subpopulations
- Incorporate graph-aware attention mechanisms to enhance local neighbourhood structure

### 1.3 Objectives

1. **Replace PCA with autoencoder-based embeddings** - Evaluate AE, VAE, and DAE architectures for non-linear dimensionality reduction of scRNA-seq count matrices.
2. **Introduce GAT-based embedding refinement** - Apply Graph Attention Networks as a post-encoding step that leverages cell neighbourhood topology to produce context-aware embeddings.
3. **Maintain a strictly unsupervised framework** - Model selection and clustering are driven exclusively by unsupervised metrics; reference labels are used only for posterior evaluation.
4. **Automate hyperparameter optimisation** - Perform systematic grid searches over both model and clustering hyperparameters, including the Leiden resolution and number of neighbours.
5. **Improve biological interpretability** - Annotate discovered clusters using marker gene scoring and compare against published cell type assignments.

---

## 2. Methodology

### 2.1 System Architecture Overview

The pipeline follows a modular, multi-stage architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                      │
│    matrix.mtx (gene expression) + metadata.csv + genes.csv             │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                                       │
│    • Top-N gene selection (5,000–10,000)                               │
│    • Library-size normalisation                                         │
│    • Log1p transformation                                               │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│              DIMENSIONALITY REDUCTION (Autoencoder)                      │
│                                                                         │
│    Input (N genes) → Encoder → Latent Space (d dims) → Decoder → Output│
│                                                                         │
│    Variants: AE | VAE | DAE                                             │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│              GAT REFINEMENT (Optional)                                   │
│                                                                         │
│    1. Build kNN graph from latent embeddings                            │
│    2. Multi-head Graph Attention (neighbourhood aggregation)            │
│    3. Self-supervised reconstruction objective                          │
│    4. Output: graph-aware refined embeddings Z1                         │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│              CLUSTERING (Leiden Community Detection)                     │
│                                                                         │
│    1. Build kNN graph (configurable n_neighbors)                        │
│    2. Leiden algorithm (configurable resolution)                         │
│    3. Cluster assignment per cell                                        │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│              EVALUATION & ANNOTATION                                     │
│                                                                         │
│    • Unsupervised: Silhouette, Davies-Bouldin, Calinski-Harabasz,      │
│                    DBCVI, Edge Purity, Modularity                        │
│    • Posterior:    ARI, NMI vs. reference labels (major/minor)          │
│    • Annotation:  Marker gene scoring per cluster                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Preprocessing

The input data consists of a sparse gene expression matrix from the Carraro et al. study, containing approximately 10,000 airway epithelial cells.

**Preprocessing steps:**
1. **Gene selection**: The top N highly expressed genes are retained (configurable: 5,000–10,000), broadening the feature space beyond the traditional HVG approach.
2. **Normalisation**: Library-size normalisation followed by log1p transformation standardises expression counts across cells.
3. **Tensor conversion**: The normalised matrix is converted to a PyTorch dataset for minibatch training.

**Input files:**
| File | Description |
|------|-------------|
| `matrix.mtx` | Sparse gene expression matrix (Market Matrix format) |
| `genes.csv` | Gene identifiers mapping matrix columns to gene names |
| `metadata.csv` | Cell-level metadata including donor, condition, and reference cell-type annotations |

### 2.3 Dimensionality Reduction Models

Three autoencoder architectures are implemented, all sharing a common training pipeline interface:

#### 2.3.1 Standard Autoencoder (AE)

A symmetric encoder–decoder network with batch normalisation, ReLU activations, and dropout regularisation:

```
Encoder: Input(N) → 128 → BN → ReLU → Dropout → 64 → BN → ReLU → Dropout → Latent(d)
Decoder: Latent(d) → 64 → BN → ReLU → 128 → BN → ReLU → Output(N)
Loss:    MSE(x̂, x)
```

#### 2.3.2 Variational Autoencoder (VAE)

Extends the AE with a probabilistic latent space using the reparameterisation trick:

```
Encoder: Input(N) → 128 → ReLU → 64 → ReLU → [μ(d), log σ²(d)]
Latent:  z = μ + ε·σ,  ε ~ N(0, I)
Decoder: z(d) → 64 → ReLU → 128 → ReLU → Output(N)
Loss:    MSE(x̂, x) + β · KL(q(z|x) || p(z))
```

The KL divergence term regularises the latent space toward a standard normal distribution, promoting smoothness and interpolation.

#### 2.3.3 Denoising Autoencoder (DAE)

Introduces two robustness mechanisms that force the encoder to learn denoising capabilities:

```
Training:  x̃ = x ⊙ Bernoulli(1 - mask_rate)    [input corruption]
Encoder:   x̃ → 256 → BN → ReLU → Dropout → 128 → BN → ReLU → Dropout → Latent(d)
Decoder:   Latent(d) → 128 → BN → ReLU → 256 → BN → ReLU → Output(N)
Loss:      MSE(x̂, x_clean)                      [reconstruct ORIGINAL input]
```

Key properties:
- **Masking** (noise injection): Randomly zeros a configurable fraction of input genes during training, preventing trivial identity mappings.
- **Dropout**: Regularises internal representations by randomly deactivating neurons.
- **Inference**: Both masking and dropout are disabled during embedding extraction, yielding clean, deterministic representations.

### 2.4 Graph Attention Network Refinement

After obtaining latent embeddings from the autoencoder, an optional GAT refinement stage incorporates cell neighbourhood structure:

**Step 1 - kNN Graph Construction:**  
A k-nearest-neighbour graph is built in the embedding space (Euclidean distance). Edges are symmetrised and self-loops added to ensure bidirectional message passing.

**Step 2 - Multi-Head Graph Attention:**

```
GAT Layer 1:  Z₀ (d) → Multi-Head Attention (H heads, hidden_dim) → ELU → Dropout
              Output: (n_cells, H �- hidden_dim)

GAT Layer 2:  → Single-Head Attention → Output Z₁ (d)
              Refined embeddings capturing graph structure

Reconstruction Head:  Z₁ → Linear → Ẑ₀
              Loss: MSE(Ẑ₀, Z₀)    [self-supervised]
```

The self-supervised reconstruction objective (predicting original embeddings from refined ones) ensures the GAT preserves information while injecting graph context. The attention mechanism learns to weight neighbour contributions differently, allowing the model to identify biologically relevant cell–cell relationships.

**Attention interpretability**: The learned attention weights can be aggregated at the cluster level to produce cluster-to-cluster attention maps, revealing which cell populations the model considers most related.

### 2.5 Clustering

Clustering is performed using the **Leiden algorithm** (Traag et al., 2019), a community detection method that optimises modularity on the cell–cell graph:

1. A kNN graph is constructed from the (optionally refined and scaled) embeddings using configurable `n_neighbors`.
2. The Leiden algorithm partitions cells into communities at configurable `resolution`.
3. Higher resolution produces more clusters; lower resolution yields coarser groupings.

Both `n_neighbors` and `resolution` are treated as hyperparameters in the search grid, as they critically influence the final clustering granularity.

### 2.6 Hyperparameter Optimisation

The project implements an **offline grid search** pipeline that jointly optimises:

| Category | Parameters |
|----------|-----------|
| **Autoencoder** | `learning_rate`, `latent_dim`, `batch_size`, `epochs`, `dropout_rate`, `n_top_genes` |
| **DAE-specific** | `mask_rate` |
| **GAT refinement** | `knn_k`, `gat_heads`, `gat_learning_rate`, `gat_dropout`, `gat_epochs` |
| **Clustering** | `leiden_resolution`, `n_neighbors` |

**Search protocol:**
1. Generate all hyperparameter combinations (Cartesian product).
2. For each configuration:
   - Split data into 80% training / 20% validation (fixed seed for reproducibility).
   - Train the autoencoder on the training set.
   - Extract validation embeddings (`shuffle=False` to maintain cell order alignment).
   - Apply optional GAT refinement on validation embeddings.
   - Standardise embeddings (zero mean, unit variance).
   - Run Leiden clustering on validation embeddings.
   - Compute unsupervised clustering metrics.
3. Select the best configuration by **composite score** (see Section 2.7).
4. Retrain the best model on the **full dataset** (all cells) for final posterior evaluation.

**Model selection criteria** - Only unsupervised metrics drive model selection via a **composite score**. Configurations are filtered to exclude degenerate solutions (n_clusters < 2, n_clusters > 100, min_cluster_size < 5).

### 2.7 Evaluation Strategy

The evaluation framework explicitly separates model selection from biological validation:

#### Composite Score (used for model selection)

The best configuration is selected by a multiplicative composite score that ensures a model must exhibit good geometry, sufficient resolution, and reasonable cluster balance to rank highly:

```
Composite Score = Base Quality x Cluster Resolution x Cluster Balance
```

| Component | What it measures | Role in selection |
|-----------|-----------------|-------------------|
| **Silhouette Score** | How well each cell fits within its assigned cluster vs. other clusters | Rewards internally cohesive, well-separated clusters |
| **Davies-Bouldin Index** | Cluster compactness and separation (lower is better, inverted before use) | Penalises overlapping or poorly separated clusters |
| **Calinski-Harabasz Score** | Between-cluster separation vs. within-cluster dispersion (normalised) | Rewards global separation without dominating the score |
| **Base Quality** | Weighted combination of Silhouette, inverse Davies-Bouldin, and Calinski-Harabasz | Captures the geometric quality of the clustering |
| **Cluster Resolution** | Whether the clustering has enough effective structure | Prevents favouring overly coarse solutions (e.g. 3 broad clusters) |
| **Cluster Balance** | Whether cluster sizes are reasonably distributed | Penalises one giant dominant cluster or many tiny unstable clusters |

In summary:
- **Base Quality** = Are the clusters well separated?
- **Cluster Resolution** = Did the model avoid collapsing into too few groups?
- **Cluster Balance** = Are the cluster sizes reasonable?

#### Additional Unsupervised Metrics (logged for analysis)

| Metric | Interpretation |
|--------|---------------|
| **DBCVI** [0, 1] | Density-based cluster validity (lower is better) |
| **Edge Purity** [0, 1] | Fraction of kNN graph edges connecting same-cluster cells |
| **Modularity** [-0.5, 1] | Graph density within clusters vs. random expectation |

#### Posterior Reference Metrics (used ONLY for evaluation, never for selection)

| Metric | Interpretation |
|--------|---------------|
| **Adjusted Rand Index (ARI)** [-1, 1] | Agreement with reference annotations (chance-adjusted) |
| **Normalised Mutual Information (NMI)** [0, 1] | Information overlap with reference labels |

These are computed against both `major` (coarse cell types) and `minor` (fine-grained subtypes) reference annotations from the original study.

### 2.8 Cell Type Annotation

Discovered clusters are annotated post-hoc using marker gene scoring:

1. **Signature gene sets** are defined for each known cell type and subtype (derived from the literature and Supplementary Table 2 of the reference study).
2. **Scanpy's `sc.tl.score_genes`** computes per-cell signature scores.
3. **Per-cluster aggregation** assigns each cluster the cell type whose signature score is highest.

This provides biological interpretability without influencing the unsupervised clustering process.

---

## 3. Project Structure

```
cf-repo/
├── config/
│   ├── embeddings/
│   │   ├── ae_embedding.yml          # AE training configuration
│   │   ├── ae_hp_search.yml          # AE hyperparameter search grid
│   │   ├── dae_embedding.yml         # DAE training configuration
│   │   └── vae_embedding.yml         # VAE training configuration
│   ├── annotations/                   # Cell-type annotation configs
│   ├── clustering/                    # Clustering pipeline configs
│   ├── clustering.yml                 # Leiden clustering parameters
│   ├── graph.yml                      # Graph construction settings
│   └── preprocessing.yml              # Preprocessing parameters
│
├── src/
│   ├── models/
│   │   ├── ae.py                      # Standard Autoencoder
│   │   ├── vae.py                     # Variational Autoencoder
│   │   ├── dae.py                     # Denoising Autoencoder
│   │   ├── gat_refiner.py            # Graph Attention Network (pure PyTorch)
│   │   └── graph_representation.py    # Graph data structures
│   │
│   ├── pipelines/
│   │   ├── embedding/
│   │   │   ├── ae.py                  # AE training pipeline
│   │   │   ├── vae.py                 # VAE training pipeline
│   │   │   ├── dae.py                 # DAE training pipeline
│   │   │   ├── ae_hp_search_offline.py    # AE offline hyperparameter search
│   │   │   ├── dae_hp_search_offline.py   # DAE offline hyperparameter search
│   │   │   ├── ae_hp_search.py        # W&B-based HP search (online)
│   │   │   ├── dae_hp_search.py       # W&B-based DAE HP search (online)
│   │   │   └── graph_refinement.py    # GAT refinement sub-pipeline
│   │   ├── clustering/                # Clustering pipelines
│   │   └── annotations/              # Cell type scoring pipelines
│   │
│   ├── clustering/
│   │   ├── leiden.py                  # Leiden community detection
│   │   └── compare_clusters.py        # Cluster comparison analysis
│   │
│   ├── interpretability/
│   │   ├── visualize_attention.py     # GAT attention heatmaps and graphs
│   │   └── cluster_attention.py       # Cluster-level attention analysis
│   │
│   ├── utils/
│   │   ├── data.py                    # Data splitting, loaders, config I/O
│   │   ├── datasets.py                # PyTorch dataset classes
│   │   ├── data_loader.py             # Data loading and preprocessing
│   │   ├── graph.py                   # kNN graph, Leiden, scaling utilities
│   │   ├── metrics.py                 # All clustering evaluation metrics
│   │   └── wandb.py                   # Weights & Biases integration
│   │
│   └── main.py                        # Entry point for single model training
│
├── preprocessing/
│   ├── build_h5ad_object_from_seurat.py   # Convert Seurat objects to AnnData
│   ├── extract_data_from_rds.R            # Extract data from R objects
│   └── signature_genes_*.csv              # Marker gene lists
│
├── notebooks/
│   └── reporting.ipynb                # Results analysis notebook
│
├── reporting/
│   └── embedding_optim.ipynb          # Embedding optimisation analysis
│
├── tests/                             # Unit tests
├── scripts/                           # Utility scripts
│
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
│
└── Data files (root, not version-controlled)
    ├── matrix.mtx                     # Sparse gene expression matrix (~1 GB)
    ├── genes.csv                      # Gene identifiers
    └── metadata.csv                   # Cell metadata with annotations
```

---

## 4. Installation and Setup

### Prerequisites

- Python ≥ 3.8
- ~1 GB disk space for data files
- (Optional) CUDA-capable GPU for accelerated training
- (Optional) Weights & Biases account for experiment tracking

### Step 1: Clone and create environment

```powershell
git clone https://github.com/mila49/cf-repo
cd cf-repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate          # Linux/Mac
```

### Step 2: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Prepare data

Place the following files in the repository root:
- `matrix.mtx` - sparse expression matrix (Market Matrix format)
- `genes.csv` - gene name mapping
- `metadata.csv` - cell metadata with `major` and `minor` annotation columns

### Step 4: Configure environment (optional)

```powershell
cp .env.example .env
# Edit .env with your WANDB_API_KEY and WANDB_ENTITY
```

---

## 5. Usage

### 5.1 Single Model Training

Train a single autoencoder with fixed hyperparameters defined in the YAML configuration:

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Train AE (default in src/main.py)
python -m src.main
```

To switch between models (AE / VAE / DAE), edit `src/main.py` and uncomment the desired pipeline.

### 5.2 Hyperparameter Search

Run the offline grid search (no internet connection required):

```powershell
# AE + GAT hyperparameter search
python -m src.pipelines.embedding.ae_hp_search_offline

# DAE + GAT hyperparameter search
python -m src.pipelines.embedding.dae_hp_search_offline
```

The search pipeline:
1. Iterates over all parameter combinations defined in `config/embeddings/ae_hp_search.yml`
2. Saves intermediate results to `*_hp_search_results_temp.csv` (crash-safe)
3. Selects the best configuration by validation silhouette score
4. Retrains on full data and saves posterior evaluation to `*_hp_search_best_full_posterior_*.csv`

For online hyperparameter search with Weights & Biases tracking:
```powershell
python -m src.pipelines.embedding.ae_hp_search
python -m src.pipelines.embedding.dae_hp_search
```

### 5.3 Visualization and Reporting

```powershell
# Generate UMAP visualizations of best configuration
python visualize_best_config.py
python visualize_dae_umap.py
```

Jupyter notebooks in `notebooks/` and `reporting/` provide interactive analysis of hyperparameter search results and embedding quality.

---

## 6. Configuration Reference

### Embedding Configuration (`config/embeddings/ae_embedding.yml`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_top_genes` | Number of genes to retain after selection | 5000 |
| `latent_dim` | Dimensionality of latent space | 16 |
| `learning_rate` | Adam optimiser learning rate | 3e-5 |
| `batch_size` | Training minibatch size | 64 |
| `epochs` | Number of training epochs | 5 |
| `dropout_rate` | Encoder dropout probability | 0.2 |
| `use_graph_refinement` | Enable GAT post-processing | true |
| `knn_k` | Neighbours for GAT graph construction | 15 |
| `gat_heads` | Number of attention heads | 4 |
| `gat_hidden_dim` | GAT hidden layer dimensionality | 32 |
| `gat_epochs` | GAT training epochs | 50 |
| `gat_learning_rate` | GAT optimiser learning rate | 0.001 |
| `device` | Compute device (`cpu` / `cuda`) | cpu |

### Hyperparameter Search Grid (`config/embeddings/ae_hp_search.yml`)

Defines parameter lists for grid search. All combinations are evaluated:

| Parameter | Search Space |
|-----------|-------------|
| `n_top_genes` | [5000, 6000, 7000, 8000, 9000, 10000] |
| `learning_rate` | [5e-6, 1e-5, 2e-5, 3e-5] |
| `latent_dim` | [12, 24, 32, 64] |
| `batch_size` | [64, 128] |
| `epochs` | [15, 25] |
| `dropout_rate` | [0.1, 0.2, 0.3] |
| `leiden_resolution` | [0.10, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0] |
| `n_neighbors` | [50, 100, 200] |
| `gat_heads` | [4, 8] |
| `knn_k` | [5, 15, 50] |

---

## 7. Outputs

### Generated Files

| File Pattern | Description |
|-------------|-------------|
| `hp_search_results_YYYYMMDD_HHMMSS.csv` | All HP search configurations with metrics |
| `hp_search_best_full_posterior_*.csv` | Full-dataset posterior evaluation of best model |
| `dae_hp_search_results_*.csv` | DAE-specific search results |
| `adata_ae.h5ad` / `adata_vae.h5ad` | AnnData objects with latent embeddings |
| `best_config_*_umap.png` | UMAP visualisations coloured by cluster |
| `best_config_*_umap_major.png` | UMAP coloured by major cell type |
| `best_config_*_umap_minor.png` | UMAP coloured by minor subtype |
| `best_config_*_attention_heatmap.png` | GAT cluster-level attention weights |
| `best_config_*_attention_graph.png` | GAT attention as directed graph |
| `ae_model.pt` / `vae_model.pt` | Trained model weights |
| `crosstab_counts.csv` | Cluster �- cell-type contingency table |

### AnnData Output Structure

The output `.h5ad` files follow the AnnData convention:
- `adata.X` - Log-normalised gene expression (selected genes)
- `adata.obsm['X_ae']` - Autoencoder embeddings
- `adata.obsm['X_ae_gat']` - GAT-refined embeddings
- `adata.obs['leiden_clusters']` - Cluster assignments
- `adata.obs` - Cell metadata (major/minor types, donor, condition)

### Experiment Tracking (W&B)

When configured, Weights & Biases logs training curves, hyperparameter configurations, and clustering metrics for each run.

---

## 8. References

1. Carraro, G. et al. (2021). Transcriptional analysis of cystic fibrosis airways at single-cell resolution reveals altered epithelial cell states and composition. *Nature Medicine*, 27, 806–814.
2. Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9, 5233.
3. Veličković, P. et al. (2018). Graph Attention Networks. *ICLR 2018*.
4. Vincent, P. et al. (2010). Stacked Denoising Autoencoders. *JMLR*, 11, 3371–3408.
5. Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*, 19, 15.

---

## License

This project is developed as part of a postgraduate programme in Artificial Intelligence and Deep Learning.
