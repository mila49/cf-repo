from scipy.io import mmread
from scipy import sparse
import anndata as ad
import numpy as np
import pandas as pd
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
    n_top_genes: int = 2000,
    min_genes: int = 200,
    min_cells: int = 3,
    max_genes: int | None = 6000,
    max_pct_mt: float | None = 15.0,
    target_sum: float = 1e4,
    hvg_flavor: str = "seurat",
    batch_key: str | None = None,
    copy: bool = True,
) -> ad.AnnData:
    """
    Preprocess scRNA-seq data for clustering / representation learning.

    Expected input:
        adata.X = raw counts, cells × genes

    Steps:
        1. Save raw counts in adata.layers["counts"]
        2. Basic cell/gene filtering
        3. Compute QC metrics
        4. Filter high-mito / possible low-quality cells
        5. Normalize total counts per cell
        6. Log1p transform
        7. Select highly variable genes
        8. Subset to HVGs
        9. Convert X to CSR float32 for lower RAM usage

    Returns:
        Preprocessed AnnData object containing log-normalized HVG expression.
    """

    if copy:
        adata = adata.copy()

    print("Before preprocessing:", adata)

    # Ensure sparse format for memory efficiency
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    # Use float32 to reduce RAM usage
    adata.X = adata.X.astype(np.float32)

    # Save raw counts before modifying X
    adata.layers["counts"] = adata.X.copy()

    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Mitochondrial genes.
    # Human genes often start with "MT-", mouse genes often with "mt-".
    adata.var["mt"] = (
        adata.var_names.str.startswith("MT-")
        | adata.var_names.str.startswith("mt-")
    )

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Optional QC filtering
    if max_pct_mt is not None:
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()

    if max_genes is not None:
        adata = adata[adata.obs["n_genes_by_counts"] < max_genes].copy()

    # Normalize and log-transform.
    # normalize_total corrects for different sequencing depth per cell.
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Save log-normalized full-gene matrix before HVG subsetting
    adata.layers["lognorm"] = adata.X.copy()

    # Highly variable genes
    if hvg_flavor == "seurat_v3":
        # seurat_v3 expects count data, so use the counts layer.
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            layer="counts",
            batch_key=batch_key,
        )
    else:
        # seurat expects log-normalized data in adata.X.
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=hvg_flavor,
            batch_key=batch_key,
        )

    # Subset to HVGs
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Keep matrix sparse and memory efficient
    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = sparse.csr_matrix(adata.X.astype(np.float32))

    print("After preprocessing:", adata)

    return adata
