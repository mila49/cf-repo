from __future__ import annotations

from typing import Any, Mapping, Optional

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from src.models import GATRefiner


def build_knn_graph_from_embeddings(
    embeddings: np.ndarray,
    k: int,
    include_self_loops: bool = True,
) -> tuple[np.ndarray, csr_matrix]:
    """Build a directed kNN graph from embeddings.

    Args:
        embeddings: Matrix with shape ``[n_cells, emb_dim]``.
        k: Number of nearest neighbors per node.
        include_self_loops: Whether to include ``i -> i`` edges.

    Returns:
        A tuple ``(edge_index, adjacency)`` where:
            - ``edge_index`` has shape ``[2, n_edges]``
            - ``adjacency`` is a CSR matrix with shape ``[n_cells, n_cells]``
    """
    n_cells = embeddings.shape[0]
    if n_cells <= 1:
        raise ValueError("At least 2 cells are required to build a kNN graph.")

    effective_k = max(1, min(k, n_cells - 1))
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(embeddings)
    neighbor_idx = nn.kneighbors(return_distance=False)

    sources = np.repeat(np.arange(n_cells), effective_k)
    targets = neighbor_idx[:, 1 : effective_k + 1].reshape(-1)

    # Symmetrize directed edges for more stable message passing.
    src_all = np.concatenate([sources, targets])
    dst_all = np.concatenate([targets, sources])

    if include_self_loops:
        self_idx = np.arange(n_cells)
        src_all = np.concatenate([src_all, self_idx])
        dst_all = np.concatenate([dst_all, self_idx])

    edge_pairs = np.unique(np.stack([src_all, dst_all], axis=1), axis=0)
    edge_index = edge_pairs.T.astype(np.int64)

    adjacency = csr_matrix(
        (np.ones(edge_pairs.shape[0], dtype=np.float32), (edge_pairs[:, 0], edge_pairs[:, 1])),
        shape=(n_cells, n_cells),
    )

    return edge_index, adjacency


def _safe_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute unsupervised clustering metrics with guards for degenerate partitions."""
    metrics: dict[str, float] = {
        f"{prefix}_n_clusters": float(np.unique(labels).size),
        f"{prefix}_silhouette": np.nan,
        f"{prefix}_davies_bouldin": np.nan,
        f"{prefix}_calinski_harabasz": np.nan,
    }

    n_clusters = int(metrics[f"{prefix}_n_clusters"])
    n_samples = labels.shape[0]
    if n_clusters <= 1 or n_clusters >= n_samples:
        return metrics

    metrics[f"{prefix}_silhouette"] = float(silhouette_score(embeddings, labels))
    metrics[f"{prefix}_davies_bouldin"] = float(davies_bouldin_score(embeddings, labels))
    metrics[f"{prefix}_calinski_harabasz"] = float(calinski_harabasz_score(embeddings, labels))
    return metrics


def _collect_reference_metrics(
    adata: ad.AnnData,
    baseline_labels: np.ndarray,
    refined_labels: np.ndarray,
) -> dict[str, float]:
    """Compute ARI/NMI against available reference labels (post-hoc evaluation only)."""
    metrics: dict[str, float] = {}
    for ref_col in ("major", "minor"):
        if ref_col not in adata.obs.columns:
            continue

        ref_labels = adata.obs[ref_col].astype(str).values
        metrics[f"baseline_ari_{ref_col}"] = float(adjusted_rand_score(ref_labels, baseline_labels))
        metrics[f"baseline_nmi_{ref_col}"] = float(normalized_mutual_info_score(ref_labels, baseline_labels))
        metrics[f"refined_ari_{ref_col}"] = float(adjusted_rand_score(ref_labels, refined_labels))
        metrics[f"refined_nmi_{ref_col}"] = float(normalized_mutual_info_score(ref_labels, refined_labels))
    return metrics


def _train_gat_refiner(
    z0: np.ndarray,
    edge_index: np.ndarray,
    config: Mapping[str, Any],
    device: str,
) -> tuple[np.ndarray, float, Optional[dict[str, np.ndarray]]]:
    """Train GAT refiner with self-supervised reconstruction objective."""
    input_dim = int(z0.shape[1])
    hidden_dim = int(config.get("gat_hidden_dim", max(8, input_dim)))
    out_dim = int(config.get("gat_out_dim", input_dim))
    heads = int(config.get("gat_heads", 4))
    dropout = float(config.get("gat_dropout", 0.2))

    model = GATRefiner(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        heads=heads,
        dropout=dropout,
    ).to(device)

    x = torch.from_numpy(z0).to(device=device, dtype=torch.float32)
    edge_t = torch.from_numpy(edge_index).to(device=device, dtype=torch.long)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("gat_learning_rate", 1e-3)),
        weight_decay=float(config.get("gat_weight_decay", 0.0)),
    )
    loss_fn = torch.nn.MSELoss()

    epochs = int(config.get("gat_epochs", 50))
    model.train()
    loss_value = np.nan
    for _ in range(epochs):
        optimizer.zero_grad()
        _, z0_hat, _ = model(x, edge_t, return_attention=False)
        loss = loss_fn(z0_hat, x)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())

    model.eval()
    with torch.no_grad():
        return_attention = bool(config.get("save_gat_attention_weights", False))
        z1, _, attention = model(x, edge_t, return_attention=return_attention)

    attention_np: Optional[dict[str, np.ndarray]] = None
    if attention is not None:
        attention_np = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in attention.items()
            if tensor is not None
        }

    model_path = config.get("gat_model_output_path")
    if model_path:
        torch.save(model.state_dict(), str(model_path))

    return z1.detach().cpu().numpy(), loss_value, attention_np


def apply_graph_refinement(
    adata: ad.AnnData,
    config: Mapping[str, Any],
    device: str,
) -> Optional[dict[str, float]]:
    """Apply optional graph-based embedding refinement.

    When disabled (`use_graph_refinement=false`), this function is a no-op.
    """
    if not bool(config.get("use_graph_refinement", False)):
        return None

    model_name = str(config.get("graph_refinement_model", "gat")).lower()
    if model_name != "gat":
        raise ValueError(f"Unsupported graph_refinement_model '{model_name}'. Expected 'gat'.")

    z0_key = str(config.get("output_obsm_key", "X_emb"))
    if z0_key not in adata.obsm:
        raise KeyError(f"Embedding key '{z0_key}' not found in adata.obsm.")

    z0 = np.asarray(adata.obsm[z0_key], dtype=np.float32)
    if z0.ndim != 2:
        raise ValueError("Z0 embeddings must be a 2D array [n_cells, embedding_dim].")

    knn_k = int(config.get("knn_k", 15))
    edge_index_z0, adjacency_z0 = build_knn_graph_from_embeddings(z0, k=knn_k)

    z1, gat_loss, attention = _train_gat_refiner(z0, edge_index_z0, config=config, device=device)

    refined_key = str(config.get("refined_output_obsm_key", f"{z0_key}_gat"))
    original_copy_key = str(config.get("original_output_obsm_key", f"{z0_key}_z0"))
    adata.obsm[original_copy_key] = z0.copy()
    adata.obsm[refined_key] = z1

    baseline_graph = ad.AnnData(X=z0)
    refined_graph = ad.AnnData(X=z1)
    leiden_neighbors = int(config.get("n_neighbors", knn_k))
    leiden_resolution = float(config.get("resolution", 0.5))

    sc.pp.neighbors(baseline_graph, n_neighbors=leiden_neighbors, use_rep="X")
    sc.tl.leiden(baseline_graph, resolution=leiden_resolution, key_added="leiden")

    sc.pp.neighbors(refined_graph, n_neighbors=leiden_neighbors, use_rep="X")
    sc.tl.leiden(refined_graph, resolution=leiden_resolution, key_added="leiden")

    baseline_labels = baseline_graph.obs["leiden"].astype(str).values
    refined_labels = refined_graph.obs["leiden"].astype(str).values

    baseline_key = str(config.get("baseline_leiden_key", "leiden_baseline_z0"))
    refined_cluster_key = str(config.get("refined_leiden_key", "leiden_refined_z1"))
    adata.obs[baseline_key] = baseline_labels
    adata.obs[refined_cluster_key] = refined_labels

    adata.obsp["knn_graph_z0"] = adjacency_z0
    adata.obsp["knn_graph_z1"] = refined_graph.obsp["connectivities"].copy()

    adata.uns["graph_refinement_graph_z0_edge_index"] = edge_index_z0
    if attention is not None:
        adata.uns["gat_attention"] = attention

    metrics = {
        "gat_reconstruction_mse": gat_loss,
        "knn_k": float(knn_k),
        "leiden_neighbors": float(leiden_neighbors),
        "leiden_resolution": float(leiden_resolution),
    }
    metrics.update(_safe_clustering_metrics(z0, baseline_labels, prefix="baseline"))
    metrics.update(_safe_clustering_metrics(z1, refined_labels, prefix="refined"))

    for metric_name in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        b_key = f"baseline_{metric_name}"
        r_key = f"refined_{metric_name}"
        metrics[f"delta_{metric_name}"] = float(metrics.get(r_key, np.nan) - metrics.get(b_key, np.nan))

    metrics.update(_collect_reference_metrics(adata, baseline_labels, refined_labels))
    adata.uns["graph_refinement_metrics"] = metrics

    return metrics