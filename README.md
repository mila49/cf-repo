# Activate virutal environemnt

```python
.\.venv\Scripts\Activate.ps1
```

# AnnData Object Structure

## Overview

**AnnData** (Annotated Data) is a Python library designed to handle annotated data matrices commonly used in bioinformatics, particularly for single-cell RNA-sequencing (scRNA-seq) analysis. An AnnData object is a container that stores gene expression data along with associated metadata and analysis results in an organized and efficient manner.

## Core Components and Properties

### 1. **X** (Expression Matrix)
- **Type**: Matrix (numpy array, scipy sparse matrix, or similar)
- **Definition**: The main data matrix containing gene expression measurements
- **Dimensions**: (n_obs, n_vars) where:
  - `n_obs` = number of observations (cells)
  - `n_vars` = number of variables (genes)
- **Example**: Raw counts, normalized expression values, or log-transformed data

```python
adata.X  # Gene expression matrix (cells × genes)
```

### 2. **obs** (Observations Metadata)
- **Type**: pandas DataFrame
- **Definition**: Metadata associated with observations (typically cells)
- **Dimensions**: (n_obs, n_meta) where n_meta is the number of metadata features
- **Common Contents**: 
  - Cell type annotations
  - Cell quality metrics
  - Sample/batch information
  - Clustering results
  - Cell cycle phase

```python
adata.obs  # DataFrame with cell metadata (rows=cells)
```

### 3. **var** (Variables Metadata)
- **Type**: pandas DataFrame
- **Definition**: Metadata associated with variables (typically genes)
- **Dimensions**: (n_vars, n_meta) where n_meta is the number of metadata features
- **Common Contents**:
  - Gene symbols and IDs
  - Gene biotype information
  - Expression statistics (mean, variance, dispersion)
  - Quality control metrics

```python
adata.var  # DataFrame with gene metadata (rows=genes)
```

### 4. **obsm** (Observations Multidimensional Matrix)
- **Type**: Dictionary of numpy arrays or sparse matrices
- **Definition**: Multidimensional data associated with observations
- **Dimensions**: Keys map to arrays of shape (n_obs, n_components)
- **Common Contents**:
  - PCA embeddings (`'X_pca'`)
  - UMAP embeddings (`'X_umap'`)
  - t-SNE embeddings (`'X_tsne'`)
  - Other dimensionality reduction results

```python
adata.obsm  # Dictionary of cell embeddings
adata.obsm['X_pca']   # PCA coordinates
adata.obsm['X_umap']  # UMAP coordinates
```

### 5. **varm** (Variables Multidimensional Matrix)
- **Type**: Dictionary of numpy arrays or sparse matrices
- **Definition**: Multidimensional data associated with variables
- **Dimensions**: Keys map to arrays of shape (n_vars, n_components)
- **Common Contents**:
  - Principal components loadings
  - Feature contributions to embeddings

```python
adata.varm  # Dictionary of gene-level embeddings
```

### 6. **obsp** (Observations Pairwise Matrix)
- **Type**: Dictionary of sparse matrices
- **Definition**: Pairwise relationships between observations (cells)
- **Dimensions**: Keys map to (n_obs, n_obs) matrices
- **Common Contents**:
  - Cell-cell distance matrices
  - Connectivity graphs (k-nearest neighbors)
  - Correlation matrices

```python
adata.obsp  # Pairwise cell relationships
adata.obsp['distances']    # Distance matrix
adata.obsp['connectivities']  # Connectivity graph
```

### 7. **varp** (Variables Pairwise Matrix)
- **Type**: Dictionary of sparse matrices
- **Definition**: Pairwise relationships between variables (genes)
- **Dimensions**: Keys map to (n_vars, n_vars) matrices
- **Common Contents**:
  - Gene-gene correlation matrices
  - Co-expression networks

```python
adata.varp  # Pairwise gene relationships
```

### 8. **layers** (Additional Data Layers)
- **Type**: Dictionary of matrices
- **Definition**: Alternative representations or layers of the main expression data
- **Dimensions**: Each layer has shape (n_obs, n_vars)
- **Common Contents**:
  - Raw counts (`'raw'`)
  - Normalized counts (`'normalized'`)
  - Batch-corrected data
  - Different preprocessing versions

```python
adata.layers  # Dictionary of expression matrices
adata.layers['raw']        # Raw count data
adata.layers['normalized'] # Normalized expression
```

### 9. **uns** (Unstructured Metadata)
- **Type**: Dictionary
- **Definition**: Unstructured and miscellaneous metadata that doesn't fit other categories
- **Common Contents**:
  - Parameter settings used in analyses
  - Figure configurations
  - Analysis notes and descriptions
  - Reference information

```python
adata.uns  # Unstructured annotations
adata.uns['neighbors']     # Neighborhood graph parameters
adata.uns['pca']           # PCA parameters
```

## Example Structure

```
AnnData object with n_obs × n_vars = 5000 × 20000
    obs: 'cell_type', 'batch', 'n_counts', 'n_genes'
    var: 'gene_names', 'gene_ids', 'n_cells', 'mean_counts', 'dispersions'
    obsm: 'X_pca', 'X_umap'
    varm: 'PCs'
    obsp: 'distances', 'connectivities'
    layers: 'raw', 'normalized'
    uns: 'neighbors', 'pca', 'umap'
```

## Basic Operations

### Accessing Data
```python
# Access expression matrix
expression = adata.X

# Access cell metadata
cell_types = adata.obs['cell_type']

# Access gene metadata
gene_means = adata.var['mean_counts']

# Access embeddings
umap_coords = adata.obsm['X_umap']
```

### Saving and Loading
```python
# Save to H5AD format (HDF5-based)
adata.write('data.h5ad')

# Load from H5AD
adata = anndata.read_h5ad('data.h5ad')
```

## Data Types and Formats

| Component | Type | Shape | Sparse Support |
|-----------|------|-------|-----------------|
| **X** | Matrix | (n_obs, n_vars) | Yes |
| **obs** | DataFrame | (n_obs, n_meta) | No |
| **var** | DataFrame | (n_vars, n_meta) | No |
| **obsm** | Dict[ndarray] | (n_obs, k) | Yes |
| **varm** | Dict[ndarray] | (n_vars, k) | Yes |
| **obsp** | Dict[sparse] | (n_obs, n_obs) | Typically |
| **varp** | Dict[sparse] | (n_vars, n_vars) | Typically |
| **layers** | Dict[matrix] | (n_obs, n_vars) | Yes |
| **uns** | Dict | - | - |
