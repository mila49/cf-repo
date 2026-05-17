from scipy.io import mmread
import anndata as ad
import numpy as np
import scanpy as sc
from pathlib import Path


def load_data(file_path: Path) -> ad.AnnData:
    """
    Load either a .mtx or .h5ad file and return an AnnData object.
    - .mtx files are transposed because they are usually genes x cells,
      while AnnData expects cells x genes.
    - .h5ad files are loaded directly as they already contain proper AnnData structure.
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()

    if file_extension == ".mtx":
        # Load .mtx matrix file
        matrix = mmread(file_path).T.tocsr().astype(np.float32)
        adata = ad.AnnData(X=matrix)
        adata.obs_names = [f"Cell_{i}" for i in range(adata.n_obs)]
        adata.var_names = [f"Gene_{i}" for i in range(adata.n_vars)]

    elif file_extension == ".h5ad":
        # Load .h5ad file directly
        adata = ad.read_h5ad(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            f"Supported formats: .mtx, .h5ad"
        )

    print(f"Loaded data from {file_path.name}:", adata)

    return adata


def preprocess_data(
    adata: ad.AnnData,
    n_top_genes: int,
    min_genes: int = 200,
    min_cells: int = 3,
) -> ad.AnnData:
    """
    Basic preprocessing for single-cell data.
    Keeps the matrix sparse to reduce RAM usage.
    """
    print("Before preprocessing:", adata)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat",
    )

    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = adata.X.tocsr().astype(np.float32)

    print("After preprocessing:", adata)

    return adata
