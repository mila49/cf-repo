from scipy.io import mmread
from scipy import sparse
import anndata as ad
import numpy as np

def load_data(file_path):
    """Load the data from a .mtx file and return an AnnData object as sparse matrix."""
    matrix = mmread(file_path).T.tocsr().astype(np)
    adata = ad.AnnData(X=matrix)
    return adata