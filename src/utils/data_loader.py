from scipy.io import mmread
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc


def load_data(file_path: str) -> ad.AnnData:
    """
    Load a .mtx file and return an AnnData object.
    Matrix is transposed because .mtx files are often genes x cells,
    while AnnData expects cells x genes.
    We also load gene names and metadata from separate CSV files and 
    attach them to the AnnData object.

    """
    matrix = mmread(file_path).T.tocsr().astype(np.float32)

    adata = ad.AnnData(X=matrix)

    genes = pd.read_csv("genes.csv", header=None)
    metadata = pd.read_csv("metadata.csv", index_col=0)

    adata.var_names = genes[0].astype(str).values
    adata.obs = metadata

    print("Loaded data:", adata)

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
