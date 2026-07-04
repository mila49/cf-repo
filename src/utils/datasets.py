import numpy as np
import torch
from torch.utils.data import Dataset


class SparseAnnDataset(Dataset):
    """
    Dataset for sparse AnnData matrices.

    It keeps the full matrix sparse and only converts one row
    to dense when PyTorch asks for a sample.
    """

    def __init__(self, X):
        self.X = X.tocsr()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Convert idx to int to handle numpy/tensor indices
        if isinstance(idx, (np.ndarray, torch.Tensor)):
            idx = int(idx.item()) if hasattr(idx, 'item') else int(idx)
        row = self.X[idx].toarray().squeeze().astype(np.float32)
        return torch.from_numpy(row)