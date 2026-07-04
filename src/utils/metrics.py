"""
Clustering Evaluation Metrics

This module provides comprehensive clustering quality metrics including:
- Distance-based metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
- Density-based metrics (DBCVI)
- Graph-based metrics (edge purity, local entropy, conductance, modularity)
- Reference-based metrics (ARI, NMI)
"""

import numpy as np
from typing import Optional, Dict, Any
from scipy.sparse import csr_matrix
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def compute_dbcvi(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Density-Based Clustering Validation Index (DBCVI).
    
    DBCVI measures clustering quality by combining:
    - Within-cluster compactness (lower is better)
    - Between-cluster separation (higher is better)
    - Density considerations
    
    Lower DBCVI values indicate better clustering.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The embedding/feature matrix
    labels : np.ndarray, shape (n_samples,)
        Cluster labels for each sample
        
    Returns
    -------
    dbcvi : float
        The DBCVI score (lower is better, NaN if computation fails)
    """
    try:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return np.nan
        
        # Compute pairwise distances
        distances = euclidean_distances(X)
        
        # For each cluster, compute density and separation
        cluster_metrics = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = X[cluster_mask]
            n_points = cluster_points.shape[0]
            
            if n_points < 2:
                continue
            
            # Within-cluster compactness: average distance to cluster centroid
            centroid = cluster_points.mean(axis=0)
            compactness = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
            
            # Between-cluster separation: min distance to other cluster centroids
            other_centroids = []
            for other_label in unique_labels:
                if other_label != label:
                    other_mask = labels == other_label
                    other_points = X[other_mask]
                    if other_points.shape[0] > 0:
                        other_centroid = other_points.mean(axis=0)
                        other_centroids.append(other_centroid)
            
            if len(other_centroids) == 0:
                continue
            
            other_centroids = np.array(other_centroids)
            separations = np.linalg.norm(other_centroids - centroid, axis=1)
            min_separation = np.min(separations)
            
            # Density estimate: inverse of average nearest-neighbor distance within cluster
            if n_points > 1:
                cluster_distances = euclidean_distances(cluster_points)
                np.fill_diagonal(cluster_distances, np.inf)
                nn_distances = np.min(cluster_distances, axis=1)
                avg_nn_dist = np.mean(nn_distances[np.isfinite(nn_distances)])
                density = 1.0 / (avg_nn_dist + 1e-10) if avg_nn_dist > 0 else 0
            else:
                density = 0
            
            # Cluster validity: ratio of compactness to separation, weighted by density
            if min_separation > 0 and density > 0:
                validity = compactness / (min_separation * density)
                cluster_metrics.append(validity)
        
        if len(cluster_metrics) == 0:
            return np.nan
        
        # DBCVI: average validity across all clusters (lower is better)
        dbcvi = np.mean(cluster_metrics)
        
        return float(dbcvi)
        
    except Exception:
        return np.nan


def compute_edge_purity(adjacency_matrix, labels: np.ndarray) -> float:
    """
    Compute Edge Purity.
    
    Measures the fraction of edges connecting nodes with the same label.
    Higher values indicate better clustering (range: 0 to 1).
    
    Parameters
    ----------
    adjacency_matrix : sparse matrix or array
        Graph adjacency matrix
    labels : np.ndarray
        Cluster labels
        
    Returns
    -------
    edge_purity : float
        Edge purity score (higher is better, 0-1 range)
    """
    try:
        if not isinstance(adjacency_matrix, csr_matrix):
            adjacency_matrix = csr_matrix(adjacency_matrix)
        
        # Get edges (non-zero elements)
        rows, cols = adjacency_matrix.nonzero()
        
        # Remove self-loops
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        
        if len(rows) == 0:
            return np.nan
        
        # Count edges where both nodes have same label
        same_label = labels[rows] == labels[cols]
        edge_purity = np.mean(same_label)
        
        return float(edge_purity)
        
    except Exception:
        return np.nan


def compute_local_label_entropy(adjacency_matrix, labels: np.ndarray) -> float:
    """
    Compute Local Label Entropy.
    
    Measures the average entropy of cluster labels in each node's neighborhood.
    Lower values indicate better clustering (nodes surrounded by same-label neighbors).
    
    Parameters
    ----------
    adjacency_matrix : sparse matrix or array
        Graph adjacency matrix
    labels : np.ndarray
        Cluster labels
        
    Returns
    -------
    local_entropy : float
        Average local label entropy (lower is better)
    """
    try:
        if not isinstance(adjacency_matrix, csr_matrix):
            adjacency_matrix = csr_matrix(adjacency_matrix)
        
        n_nodes = adjacency_matrix.shape[0]
        unique_labels = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        entropies = []
        
        for node in range(n_nodes):
            # Get neighbors
            neighbors = adjacency_matrix[node].nonzero()[1]
            
            # Remove self-loop if present
            neighbors = neighbors[neighbors != node]
            
            if len(neighbors) == 0:
                continue
            
            # Count label distribution in neighborhood
            neighbor_labels = labels[neighbors]
            label_counts = np.zeros(len(unique_labels))
            
            for neighbor_label in neighbor_labels:
                if neighbor_label in label_to_idx:
                    label_counts[label_to_idx[neighbor_label]] += 1
            
            # Compute entropy
            if label_counts.sum() > 0:
                label_probs = label_counts / label_counts.sum()
                node_entropy = scipy_entropy(label_probs, base=2)
                entropies.append(node_entropy)
        
        if len(entropies) == 0:
            return np.nan
        
        return float(np.mean(entropies))
        
    except Exception:
        return np.nan


def compute_conductance(adjacency_matrix, labels: np.ndarray) -> float:
    """
    Compute Conductance.
    
    Measures the fraction of edges leaving each cluster relative to cluster size.
    Lower values indicate better clustering (fewer edges crossing cluster boundaries).
    
    Parameters
    ----------
    adjacency_matrix : sparse matrix or array
        Graph adjacency matrix
    labels : np.ndarray
        Cluster labels
        
    Returns
    -------
    conductance : float
        Average conductance across clusters (lower is better)
    """
    try:
        if not isinstance(adjacency_matrix, csr_matrix):
            adjacency_matrix = csr_matrix(adjacency_matrix)
        
        unique_labels = np.unique(labels)
        conductances = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Edges within cluster
            cluster_subgraph = adjacency_matrix[cluster_indices][:, cluster_mask]
            internal_edges = cluster_subgraph.nnz / 2  # Divide by 2 for undirected
            
            # Edges leaving cluster (cut)
            cluster_total = adjacency_matrix[cluster_indices].nnz
            cut_edges = cluster_total - cluster_subgraph.nnz
            
            # Volume: sum of degrees in cluster
            volume = adjacency_matrix[cluster_indices].sum()
            
            # Conductance: cut / min(volume_S, volume_not_S)
            total_volume = adjacency_matrix.sum()
            volume_complement = total_volume - volume
            
            if min(volume, volume_complement) > 0:
                cond = cut_edges / min(volume, volume_complement)
                conductances.append(cond)
        
        if len(conductances) == 0:
            return np.nan
        
        return float(np.mean(conductances))
        
    except Exception:
        return np.nan


def compute_modularity(adjacency_matrix, labels: np.ndarray) -> float:
    """
    Compute Modularity.
    
    Measures the density of edges within clusters compared to random expectation.
    Higher values indicate better clustering (range: -0.5 to 1).
    
    Parameters
    ----------
    adjacency_matrix : sparse matrix or array
        Graph adjacency matrix
    labels : np.ndarray
        Cluster labels
        
    Returns
    -------
    modularity : float
        Modularity score (higher is better)
    """
    try:
        if not isinstance(adjacency_matrix, csr_matrix):
            adjacency_matrix = csr_matrix(adjacency_matrix)
        
        # Total number of edges
        m = adjacency_matrix.nnz / 2  # Divide by 2 for undirected graph
        
        if m == 0:
            return np.nan
        
        # Compute degree for each node
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        
        unique_labels = np.unique(labels)
        modularity = 0.0
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Edges within cluster
            cluster_subgraph = adjacency_matrix[cluster_indices][:, cluster_mask]
            e_ii = cluster_subgraph.nnz / (2 * m)
            
            # Expected edges (sum of degrees in cluster)^2 / (4m^2)
            degree_sum = degrees[cluster_mask].sum()
            a_i = degree_sum / (2 * m)
            
            # Modularity contribution: e_ii - a_i^2
            modularity += (e_ii - a_i ** 2)
        
        return float(modularity)
        
    except Exception:
        return np.nan


def compute_bounded_conductance(adjacency_matrix, labels: np.ndarray) -> float:
    """
    Compute Bounded Conductance.
    
    A normalized version of conductance that is bounded between 0 and 1.
    Measures cluster quality by normalizing the cut size by the cluster volume.
    Lower values indicate better clustering.
    
    Parameters
    ----------
    adjacency_matrix : sparse matrix or array
        Graph adjacency matrix
    labels : np.ndarray
        Cluster labels
        
    Returns
    -------
    bounded_conductance : float
        Average bounded conductance across clusters (lower is better, 0-1 range)
    """
    try:
        if not isinstance(adjacency_matrix, csr_matrix):
            adjacency_matrix = csr_matrix(adjacency_matrix)
        
        unique_labels = np.unique(labels)
        bounded_conductances = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Volume of cluster (sum of degrees)
            cluster_volume = adjacency_matrix[cluster_indices].sum()
            
            if cluster_volume == 0:
                continue
            
            # Edges within cluster
            cluster_subgraph = adjacency_matrix[cluster_indices][:, cluster_mask]
            internal_edges = cluster_subgraph.nnz
            
            # Total edges from cluster
            total_edges = adjacency_matrix[cluster_indices].nnz
            
            # Cut edges (edges leaving cluster)
            cut_edges = total_edges - internal_edges
            
            # Bounded conductance: cut / volume (normalized to [0,1])
            # This is bounded because cut <= volume for connected components
            bounded_cond = cut_edges / cluster_volume
            bounded_conductances.append(bounded_cond)
        
        if len(bounded_conductances) == 0:
            return np.nan
        
        return float(np.mean(bounded_conductances))
        
    except Exception:
        return np.nan


def compute_all_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: Optional[Any] = None,
    reference_labels: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Compute all clustering metrics in one call.
    
    Parameters
    ----------
    embeddings : np.ndarray
        The embedding/feature matrix
    labels : np.ndarray
        Predicted cluster labels
    adjacency_matrix : sparse matrix, optional
        Graph adjacency matrix (required for graph-based metrics)
    reference_labels : dict, optional
        Dictionary of reference labels for supervised metrics
        Example: {'major': major_labels, 'minor': minor_labels}
        
    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Distance-based metrics (unsupervised)
    try:
        if len(np.unique(labels)) >= 2:
            metrics['silhouette'] = silhouette_score(embeddings, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
            metrics['dbcvi'] = compute_dbcvi(embeddings, labels)
        else:
            metrics['silhouette'] = np.nan
            metrics['davies_bouldin'] = np.nan
            metrics['calinski_harabasz'] = np.nan
            metrics['dbcvi'] = np.nan
    except Exception:
        metrics['silhouette'] = np.nan
        metrics['davies_bouldin'] = np.nan
        metrics['calinski_harabasz'] = np.nan
        metrics['dbcvi'] = np.nan
    
    # Graph-based metrics (require adjacency matrix)
    if adjacency_matrix is not None:
        metrics['edge_purity'] = compute_edge_purity(adjacency_matrix, labels)
        metrics['local_entropy'] = compute_local_label_entropy(adjacency_matrix, labels)
        metrics['conductance'] = compute_conductance(adjacency_matrix, labels)
        metrics['bounded_conductance'] = compute_bounded_conductance(adjacency_matrix, labels)
        metrics['modularity'] = compute_modularity(adjacency_matrix, labels)
    else:
        metrics['edge_purity'] = np.nan
        metrics['local_entropy'] = np.nan
        metrics['conductance'] = np.nan
        metrics['bounded_conductance'] = np.nan
        metrics['modularity'] = np.nan
    
    # Reference-based metrics (supervised)
    if reference_labels is not None:
        for ref_name, ref_labels in reference_labels.items():
            try:
                metrics[f'ari_{ref_name}'] = adjusted_rand_score(ref_labels, labels)
                metrics[f'nmi_{ref_name}'] = normalized_mutual_info_score(ref_labels, labels)
            except Exception:
                metrics[f'ari_{ref_name}'] = np.nan
                metrics[f'nmi_{ref_name}'] = np.nan
    
    # Add cluster count
    metrics['n_clusters'] = len(np.unique(labels))
    
    return metrics