# Hyperparameter Search Pipeline - Complete Documentation

## 🎯 Overall Goal

The pipeline performs an **offline grid search** to find the optimal hyperparameters for an **Autoencoder + Graph Attention Network (GAT) refinement** system for single-cell RNA-seq data clustering.

---

## 📋 Pipeline Stages Overview

### Stage 1️⃣: Configuration & Setup

#### 1.1 Hyperparameter Grid Definition

```python
PARAM_GRID = {
    # Autoencoder hyperparameters
    "learning_rate": [3e-5, 4e-5, 6e-5],
    "latent_dim": [16, 24],
    "batch_size": [64],
    "epochs": [10],
    
    # GAT hyperparameters
    "gat_heads": [4],
    "gat_learning_rate": [1e-3],
    "gat_dropout": [0.2],
    "knn_k": [5, 10, 15, 20, 30],  # Number of neighbors for kNN graph
}
```

- **Purpose**: Defines all hyperparameter combinations to test
- **Cartesian Product**: All combinations are generated (e.g., 3 learning rates × 2 latent dims × 5 knn_k = 30 configs)

#### 1.2 Global Configuration Flags

- **`SCALE_EMBEDDINGS = True`**: Standardize embeddings (zero mean, unit variance) before graph construction
- **`USE_GAT_REFINEMENT = True`**: Apply GAT-based refinement to embeddings
- **`COMPUTE_FULL_DATA_POSTERIOR = True`**: After finding best config, retrain on full dataset

---

### Stage 2️⃣: For Each Hyperparameter Configuration

#### 2.1 Data Loading & Splitting

```
┌─────────────────────────────────┐
│  Load full dataset              │
│  (e.g., 10,000 cells)           │
└────────────┬────────────────────┘
             │
             ├─► 80% Training (8,000 cells)
             │   • Used to train autoencoder
             │   • Shuffled batches
             │
             └─► 20% Validation (2,000 cells)
                 • Used for hyperparameter evaluation
                 • NOT shuffled (critical for alignment!)
```

**Why 80/20 split?**
- Training set: Learn embeddings
- Validation set: Evaluate clustering quality WITHOUT overfitting to training data
- **Fixed seed (42)**: Ensures reproducibility across runs

---

#### 2.2 Autoencoder Training

```
For each epoch:
  ┌──────────────────────────────┐
  │ TRAINING PHASE               │
  │ • Forward pass through AE    │
  │ • Compute reconstruction MSE │
  │ • Backpropagation            │
  │ • Update weights             │
  └──────────────────────────────┘
  
  ┌──────────────────────────────┐
  │ VALIDATION PHASE             │
  │ • Forward pass (no gradients)│
  │ • Compute validation MSE     │
  │ • Track best validation MSE  │
  └──────────────────────────────┘
```

**Outputs**:
- Trained autoencoder model
- `avg_train_mse`: Average reconstruction error on training set
- `best_val_mse`: Best validation reconstruction error

---

#### 2.3 Validation Embedding Extraction

```python
# CRITICAL: shuffle=False maintains cell order!
embedding_loader = DataLoader(val_dataset, shuffle=False)

val_embeddings = []
for batch in embedding_loader:
    latent = model.encode(batch)  # Extract latent representation
    val_embeddings.append(latent)

val_embeddings = np.vstack(val_embeddings)  # Shape: (2000, 16)
```

**Why shuffle=False?**
- We need to align cluster labels with the original cell indices
- Shuffling would break the correspondence between embeddings and metadata (major/minor labels)

---

#### 2.4 Optional GAT Refinement 🔥

If `USE_GAT_REFINEMENT = True`:

##### Step A: Build kNN Graph from Validation Embeddings

```
For each cell i:
  ┌────────────────────────────────────┐
  │ Find k nearest neighbors in       │
  │ embedding space (Euclidean)       │
  │                                    │
  │ cell_i → [cell_23, cell_87, ...]  │
  └────────────────────────────────────┘

Create edge list:
  • Source → Target (directed edges)
  • Make symmetric (add reverse edges)
  • Add self-loops (cell → cell)
  • Remove duplicates

Result: edge_index = [[src, src, ...],
                       [dst, dst, ...]]
```

**Example with k=5**:
```
Cell 0: neighbors = [1, 5, 12, 34, 89]
Edges:  0→1, 0→5, 0→12, 0→34, 0→89
        1→0, 5→0, 12→0, 34→0, 89→0  (symmetric)
        0→0 (self-loop)
```

##### Step B: Train GAT Model

```
GAT Architecture:
  ┌──────────────────────────────────────┐
  │ Input: X (val_embeddings)            │
  │ Shape: (2000, 16)                    │
  └─────────────┬────────────────────────┘
                │
                ▼
  ┌──────────────────────────────────────┐
  │ GAT Layer 1 (Multi-Head Attention)   │
  │ • hidden_dim = 32                    │
  │ • heads = 4                          │
  │ • Output: (2000, 32×4) = (2000, 128) │
  │ • ELU activation                     │
  │ • Dropout                            │
  └─────────────┬────────────────────────┘
                │
                ▼
  ┌──────────────────────────────────────┐
  │ GAT Layer 2 (Single-Head)            │
  │ • out_dim = 16                       │
  │ • heads = 1 (average pooling)        │
  │ • Output Z1: (2000, 16)              │
  └─────────────┬────────────────────────┘
                │
                ▼
  ┌──────────────────────────────────────┐
  │ Reconstruction Head                  │
  │ • Linear layer: 16 → 16              │
  │ • Output Z0_hat: (2000, 16)          │
  │ • MSE Loss: ||Z0_hat - X||²          │
  └──────────────────────────────────────┘
```

**Training Loop**:
```python
for gat_epoch in range(20):  # gat_epochs from config
    Z1, Z0_hat, attention = GAT(X, edge_index)
    loss = MSE(Z0_hat, X)  # Reconstruct original embeddings
    loss.backward()
    optimizer.step()
```

**Output**: 
- **Z1 (refined embeddings)**: Shape (2000, 16)
- These capture **graph-aware** structure (neighbor relationships)

---

#### 2.5 Embedding Standardization

If `SCALE_EMBEDDINGS = True`:

```python
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings_for_clustering)

# Each feature now has mean=0, std=1
```

**Why standardize?**
- Ensures all dimensions contribute equally to distance calculations
- Critical for kNN graph construction and clustering

---

#### 2.6 Clustering Evaluation

##### Step A: Build kNN Graph (for Leiden)

```python
# Using Scanpy's neighbor function
sc.pp.neighbors(adata, n_neighbors=400)
# Builds connectivity graph in adata.obsp['connectivities']
```

##### Step B: Run Leiden Clustering

```python
sc.tl.leiden(adata, resolution=0.5)
# Community detection algorithm
# Finds clusters that maximize modularity
```

**Result**: `cluster_labels` = [0, 2, 1, 1, 0, ...]  (one label per cell)

---

#### 2.7 Metrics Computation 📊

##### Unsupervised Clustering Metrics (PRIMARY)

Computed on **ORIGINAL (non-scaled) validation embeddings**:

**1. Silhouette Score** (range: -1 to 1, higher is better)

```python
silhouette = silhouette_score(val_embeddings, cluster_labels)
```

- Measures how similar cells are to their own cluster vs. other clusters
- **THIS IS THE PRIMARY METRIC FOR MODEL SELECTION**

**2. Davies-Bouldin Index** (lower is better)

```python
db_score = davies_bouldin_score(val_embeddings, cluster_labels)
```

- Average similarity ratio of each cluster with its most similar cluster
- Lower values mean better-separated clusters

**3. Calinski-Harabasz Score** (higher is better)

```python
ch_score = calinski_harabasz_score(val_embeddings, cluster_labels)
```

- Ratio of between-cluster to within-cluster variance
- Also known as Variance Ratio Criterion

##### Posterior Reference-Label Metrics (EVALUATION ONLY)

Computed by comparing clustering to ground-truth annotations:

**For EACH reference column (major, minor)**:

```python
# Get ground truth labels for validation cells
reference_labels = adata.obs['major'][val_indices]

# Adjusted Rand Index (range: -1 to 1, higher is better)
ari = adjusted_rand_score(reference_labels, cluster_labels)

# Normalized Mutual Information (range: 0 to 1, higher is better)
nmi = normalized_mutual_info_score(reference_labels, cluster_labels)
```

**CRITICAL**: These are **NOT** used for model selection! Only for evaluation.

---

#### 2.8 Results Collection

All metrics are saved to a dictionary:

```python
results = {
    # Hyperparameters
    "learning_rate": 3e-5,
    "latent_dim": 16,
    "knn_k": 5,
    ...
    
    # Reconstruction metrics
    "train_mse": 0.1234,
    "val_mse": 0.1456,
    "best_val_mse": 0.1400,
    
    # Unsupervised clustering (VALIDATION)
    "silhouette_val": 0.3245,
    "davies_bouldin_val": 1.2345,
    "calinski_harabasz_val": 1234.5,
    
    # Cluster diagnostics
    "n_clusters_val": 15,
    "min_cluster_size_val": 23,
    "max_cluster_size_val": 456,
    
    # Posterior evaluation (VALIDATION)
    "ari_reference_major_val": 0.6789,
    "nmi_reference_major_val": 0.7123,
    "ari_reference_minor_val": 0.5432,
    "nmi_reference_minor_val": 0.6111,
    
    # GAT metrics
    "gat_reconstruction_loss": 0.0234,
}
```

---

### Stage 3️⃣: Best Model Selection

After all configurations complete:

#### 3.1 Filter Valid Configurations

```python
valid_configs = [config for config in all_results if:
    silhouette is not NaN AND
    n_clusters >= 2 AND
    n_clusters < 100 AND
    min_cluster_size >= 5
]
```

**Why filter?**
- Avoid degenerate clusterings (e.g., all cells in one cluster)
- Avoid tiny clusters (statistical instability)

#### 3.2 Select Best by Validation Silhouette

```python
best_config = valid_configs[argmax(silhouette_val)]
```

**Why Silhouette?**
- **Unsupervised**: Doesn't require ground truth labels
- **Robust**: Works well across different cluster sizes
- **Interpretable**: Clear meaning (cohesion vs. separation)

---

### Stage 4️⃣: Full-Dataset Posterior Evaluation

If `COMPUTE_FULL_DATA_POSTERIOR = True`:

#### 4.1 Retrain on FULL Dataset

```
┌─────────────────────────────────────┐
│ Use best hyperparameters            │
│ Train on ALL cells (100%)           │
│ No train/val split                  │
└─────────────────────────────────────┘
```

**Why retrain?**
- Use all available data for final model
- Get best possible embeddings
- Evaluate on entire dataset

#### 4.2 Extract All Embeddings

```python
all_embeddings = model.encode(all_cells)  # Shape: (10000, 16)
```

#### 4.3 Cluster All Cells

```python
sc.pp.neighbors(adata, n_neighbors=400)
sc.tl.leiden(adata, resolution=0.5)
```

#### 4.4 Compute Full-Dataset Metrics

**Unsupervised**:
- Silhouette (full)
- Davies-Bouldin (full)
- Calinski-Harabasz (full)

**Supervised (Reference Labels)**:

```python
for ref_col in ['major', 'minor']:
    reference_labels = adata.obs[ref_col]
    ari_full = adjusted_rand_score(reference_labels, cluster_labels)
    nmi_full = normalized_mutual_info_score(reference_labels, cluster_labels)
```

---

### Stage 5️⃣: Output & Reporting

#### 5.1 CSV Files

1. **`hp_search_results_YYYYMMDD_HHMMSS.csv`**: All configurations + metrics
2. **`hp_search_best_full_posterior_YYYYMMDD_HHMMSS.csv`**: Full-dataset metrics for best config
3. **`hp_search_results_temp.csv`**: Intermediate results (updated after each run)

#### 5.2 Console Output

**For Each Configuration**:

```
================================
Run 5/30
Configuration: {lr: 3e-5, latent: 16, knn_k: 5}
================================

Reconstruction Metrics:
  Train MSE: 0.1234
  Val MSE: 0.1456

Validation Clustering Metrics (Unsupervised):
  Silhouette: 0.3245
  Davies-Bouldin: 1.2345
  Calinski-Harabasz: 1234.5

Posterior Reference-Label Evaluation:
  Reference Column: major
    ARI (Validation): 0.6789
    NMI (Validation): 0.7123
```

**Final Summary**:

```
================================
BEST CONFIGURATION
================================
Silhouette (Validation): 0.3456

Best Hyperparameters:
  learning_rate: 3e-5
  latent_dim: 16
  knn_k: 5

TOP 5 CONFIGURATIONS:
[Table with all metrics]

POST-HOC FULL-DATA EVALUATION:
Full Dataset Clustering Metrics:
  Silhouette: 0.3567
  ARI (major): 0.6890
  NMI (major): 0.7234
```

---

## 🔑 Key Design Principles

### 1. Train/Val Split Strategy

- **Training (80%)**: Learn embeddings
- **Validation (20%)**: Evaluate clustering quality
- **Prevents overfitting** to clustering metrics

### 2. Two-Level Optimization

- **Primary (Unsupervised)**: Silhouette score
  - Used for model selection
  - No ground truth required
- **Secondary (Supervised)**: ARI/NMI
  - Used for evaluation only
  - Measures agreement with biological labels

### 3. Fixed Clustering Parameters

- **Leiden resolution**: Fixed at 0.5
- **kNN neighbors**: Fixed at 400
- **Why?** Ensures fair comparison across AE hyperparameters

### 4. Embedding vs. Graph Scaling

- **Metrics computed on original embeddings**: Fair comparison
- **Graph built on scaled embeddings**: Better kNN construction

### 5. Robust Error Handling

- Invalid clusterings → NaN metrics
- Failed runs → logged and skipped
- Intermediate results saved → no data loss

---

## 📊 Metrics Summary Table

| **Metric** | **Range** | **Better** | **Type** | **Purpose** |
|------------|-----------|------------|----------|-------------|
| **Silhouette** | -1 to 1 | Higher | Unsupervised | **Model Selection** |
| Davies-Bouldin | 0 to ∞ | Lower | Unsupervised | Cluster separation |
| Calinski-Harabasz | 0 to ∞ | Higher | Unsupervised | Variance ratio |
| **ARI** | -1 to 1 | Higher | Supervised | Clustering agreement |
| **NMI** | 0 to 1 | Higher | Supervised | Information overlap |
| MSE | 0 to ∞ | Lower | Reconstruction | AE quality |

---

## 🚀 Usage

### Running the Pipeline

```bash
python src/pipelines/embedding/ae_hp_search_offline.py
```

### Analyzing Results (A Posteriori)

```bash
# Analyze most recent results
python scripts/analyze_hp_results.py

# Analyze specific file
python scripts/analyze_hp_results.py --file hp_search_results_20260621_143052.csv

# Show top 10 configurations
python scripts/analyze_hp_results.py --top 10
```

---

## 📁 File Locations

- **Main script**: `src/pipelines/embedding/ae_hp_search_offline.py`
- **Analysis script**: `scripts/analyze_hp_results.py`
- **Config files**: `config/embeddings/ae_embedding.yml`, `config/clustering/leiden_clustering.yml`
- **GAT model**: `src/models/gat_refiner.py`
- **Graph refinement**: `src/pipelines/embedding/graph_refinement.py`

---

## 🎓 Pipeline Philosophy

This pipeline provides a **comprehensive, statistically rigorous framework** for finding optimal hyperparameters while:

✅ **Avoiding overfitting** through train/validation splits  
✅ **Using unsupervised metrics** for model selection  
✅ **Evaluating biological relevance** with reference labels  
✅ **Maintaining reproducibility** with fixed seeds  
✅ **Ensuring robustness** through error handling  
✅ **Providing interpretability** with multiple metrics  

---

**Last Updated**: 2026-06-21
