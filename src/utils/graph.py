"""
Graph Construction and Clustering Utilities

This module provides utilities for:
- Building kNN graphs from embeddings
- Running graph-based clustering algorithms (Leiden, Louvain)
- Embedding normalization/scaling
"""

import numpy as np
import scanpy as sc
from typing import Optional, Union
from sklearn.preprocessing import StandardScaler
import anndata


def build_knn_graph(
    adata: anndata.AnnData,
    use_rep: str = 'X',
    n_neighbors: int = 15,
    **kwargs
) -> anndata.AnnData:
    """
    Build k-nearest neighbor graph on AnnData object.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing embeddings
    use_rep : str, default='X'
        Key in adata.obsm to use for graph construction
        Use 'X' for adata.X, or key like 'X_pca' for adata.obsm['X_pca']
    n_neighbors : int, default=15
        Number of neighbors for kNN graph
    **kwargs : optional
        Additional arguments passed to sc.pp.neighbors
        
    Returns
    -------
    adata : anndata.AnnData
        AnnData object with graph stored in adata.obsp['connectivities']
    """
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        **kwargs
    )
    return adata


def run_leiden_clustering(
    adata: anndata.AnnData,
    resolution: float = 0.5,
    n_neighbors: Optional[int] = None,
    use_rep: str = 'X',
    key_added: str = 'leiden',
    **kwargs
) -> anndata.AnnData:
    """
    Run Leiden clustering on AnnData object.
    
    If graph doesn't exist, builds kNN graph first.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
    resolution : float, default=0.5
        Resolution parameter for Leiden algorithm
        Higher values yield more clusters
    n_neighbors : int, optional
        Number of neighbors if graph needs to be built
        If None, uses existing graph or defaults to 15
    use_rep : str, default='X'
        Representation to use if graph needs to be built
    key_added : str, default='leiden'
        Key in adata.obs where results will be stored
    **kwargs : optional
        Additional arguments passed to sc.tl.leiden
        
    Returns
    -------
    adata : anndata.AnnData
        AnnData object with clustering results in adata.obs[key_added]
    """
    # Build graph if it doesn't exist
    if 'connectivities' not in adata.obsp:
        if n_neighbors is None:
            n_neighbors = 15
        build_knn_graph(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    
    # Run Leiden clustering
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=key_added,
        **kwargs
    )
    
    return adata


def run_louvain_clustering(
    adata: anndata.AnnData,
    resolution: float = 0.5,
    n_neighbors: Optional[int] = None,
    use_rep: str = 'X',
    key_added: str = 'louvain',
    **kwargs
) -> anndata.AnnData:
    """
    Run Louvain clustering on AnnData object.
    
    If graph doesn't exist, builds kNN graph first.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
    resolution : float, default=0.5
        Resolution parameter for Louvain algorithm
        Higher values yield more clusters
    n_neighbors : int, optional
        Number of neighbors if graph needs to be built
        If None, uses existing graph or defaults to 15
    use_rep : str, default='X'
        Representation to use if graph needs to be built
    key_added : str, default='louvain'
        Key in adata.obs where results will be stored
    **kwargs : optional
        Additional arguments passed to sc.tl.louvain
        
    Returns
    -------
    adata : anndata.AnnData
        AnnData object with clustering results in adata.obs[key_added]
    """
    # Build graph if it doesn't exist
    if 'connectivities' not in adata.obsp:
        if n_neighbors is None:
            n_neighbors = 15
        build_knn_graph(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    
    # Run Louvain clustering
    sc.tl.louvain(
        adata,
        resolution=resolution,
        key_added=key_added,
        **kwargs
    )
    
    return adata


def scale_embeddings(
    embeddings: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> tuple[np.ndarray, StandardScaler]:
    """
    Scale embeddings using StandardScaler.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix to scale
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, creates new scaler
    fit : bool, default=True
        Whether to fit the scaler on the data
        Set to False when using a pre-fitted scaler
        
    Returns
    -------
    scaled_embeddings : np.ndarray
        Scaled embedding matrix
    scaler : StandardScaler
        The StandardScaler object (fitted if fit=True)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        scaled_embeddings = scaler.fit_transform(embeddings)
    else:
        scaled_embeddings = scaler.transform(embeddings)
    
    return scaled_embeddings, scaler


def create_anndata_from_embeddings(
    embeddings: np.ndarray,
    obs: Optional[dict] = None,
    var: Optional[dict] = None,
    obsm_key: str = 'X'
) -> anndata.AnnData:
    """
    Create AnnData object from embedding matrix.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix (n_samples, n_features)
    obs : dict, optional
        Observation metadata (e.g., cell types, batches)
    var : dict, optional
        Variable metadata (e.g., gene names)
    obsm_key : str, default='X'
        If 'X', embeddings go to adata.X
        Otherwise, embeddings go to adata.obsm[obsm_key]
        
    Returns
    -------
    adata : anndata.AnnData
        AnnData object with embeddings
    """
    if obsm_key == 'X':
        adata = anndata.AnnData(X=embeddings, obs=obs, var=var)
    else:
        adata = anndata.AnnData(
            X=np.zeros((embeddings.shape[0], 1)),  # Placeholder
            obs=obs,
            var=var
        )
        adata.obsm[obsm_key] = embeddings
    
    return adata


def extract_graph_from_adata(
    adata: anndata.AnnData,
    graph_key: str = 'connectivities'
):
    """
    Extract graph adjacency matrix from AnnData object.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing graph
    graph_key : str, default='connectivities'
        Key in adata.obsp containing the graph
        
    Returns
    -------
    adjacency_matrix : sparse matrix
        Graph adjacency matrix
        
    Raises
    ------
    KeyError
        If graph_key not found in adata.obsp
    """
    if graph_key not in adata.obsp:
        raise KeyError(f"Graph '{graph_key}' not found in adata.obsp. "
                      f"Available keys: {list(adata.obsp.keys())}")
    
    return adata.obsp[graph_key]