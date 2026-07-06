from __future__ import annotations

import numpy as np
import torch


def extract_cluster_attention_matrix(
    gat_model,
    z0: np.ndarray,
    edge_index: np.ndarray,
    cluster_labels: np.ndarray,
    predicted_signatures: np.ndarray | None = None,
    layer: str = "layer2",
) -> tuple[np.ndarray, list[str], dict[str, int], dict[str, str] | None]:
    """Aggregate GAT attention weights into a cluster-level attention matrix.

    Runs a forward pass on the trained GATRefiner with ``return_attention=True``
    and builds a C×C matrix where ``A[i, j]`` is the mean attention weight from
    cells in cluster ``i`` to cells in cluster ``j``.  The diagonal holds
    within-cluster cohesion scores.

    Args:
        gat_model: Trained GATRefiner in eval mode.
        z0: Pre-GAT node embeddings, shape ``[N, D]``.
        edge_index: Directed kNN graph edges, shape ``[2, E]``.
        cluster_labels: Cluster label per cell, shape ``[N]``.
        predicted_signatures: Optional cell-type signature per cell, shape ``[N]``.
            All cells in the same Leiden cluster share the same value.
        layer: Which GAT layer to read attention from.  ``'layer2'`` (default)
            uses the single-head final layer; ``'layer1'`` averages over the
            4-head first layer.

    Returns:
        attention_matrix: ``[C, C]`` float64 array of mean attention weights.
        cluster_ids: Sorted list of unique cluster label strings.
        cluster_sizes: Mapping ``{cluster_id: n_cells}``.
        signature_per_cluster: Mapping ``{cluster_id: signature_name}`` or
            ``None`` when ``predicted_signatures`` is not provided.
    """
    device = next(gat_model.parameters()).device

    x = torch.from_numpy(np.asarray(z0, dtype=np.float32)).to(device)
    edge_t = torch.from_numpy(np.asarray(edge_index, dtype=np.int64)).to(device)

    gat_model.eval()
    with torch.no_grad():
        _, _, attention_dict = gat_model(x, edge_t, return_attention=True)

    if attention_dict is None or layer not in attention_dict:
        available = list(attention_dict or {})
        raise ValueError(
            f"Layer '{layer}' not found in attention output.  "
            f"Available keys: {available}"
        )

    alpha = attention_dict[layer].cpu().numpy()  # [E, heads]
    alpha_mean = alpha.mean(axis=1)              # [E]  — average across heads

    # Build sorted unique cluster IDs and index mapping
    labels_str = np.array([str(lb) for lb in cluster_labels])
    unique_clusters = sorted(
        np.unique(labels_str).tolist(),
        key=lambda c: int(c) if c.isdigit() else c,
    )
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

    src, dst = edge_index[0], edge_index[1]
    src_ci = np.array([cluster_to_idx[labels_str[s]] for s in src], dtype=np.int64)
    dst_ci = np.array([cluster_to_idx[labels_str[d]] for d in dst], dtype=np.int64)

    attn_sum = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    count_mat = np.zeros((n_clusters, n_clusters), dtype=np.float64)

    np.add.at(attn_sum, (src_ci, dst_ci), alpha_mean)
    np.add.at(count_mat, (src_ci, dst_ci), 1.0)

    attention_matrix = np.divide(
        attn_sum,
        count_mat,
        out=np.zeros_like(attn_sum),
        where=count_mat > 0,
    )

    cluster_sizes: dict[str, int] = {
        c: int(np.sum(labels_str == c)) for c in unique_clusters
    }

    signature_per_cluster: dict[str, str] | None = None
    if predicted_signatures is not None:
        sigs_str = np.array([str(s) for s in predicted_signatures])
        signature_per_cluster = {}
        for c in unique_clusters:
            mask = labels_str == c
            cluster_sigs = sigs_str[mask]
            if len(cluster_sigs) > 0:
                # All cells in a cluster share the same signature (assigned per-cluster)
                signature_per_cluster[c] = cluster_sigs[0]

    return attention_matrix, unique_clusters, cluster_sizes, signature_per_cluster


def extract_per_head_attention_matrices(
    gat_model,
    z0: np.ndarray,
    edge_index: np.ndarray,
    cluster_labels: np.ndarray,
    layer: str = "layer1",
) -> tuple[list[np.ndarray], list[str], dict[str, int]]:
    """Build one C×C attention matrix per attention head.

    Unlike ``extract_cluster_attention_matrix`` which averages over heads, this
    keeps each head separate so head-level specialisation is visible.

    Args:
        gat_model: Trained GATRefiner in eval mode.
        z0: Pre-GAT node embeddings, shape ``[N, D]``.
        edge_index: Directed kNN graph edges, shape ``[2, E]``.
        cluster_labels: Cluster label per cell, shape ``[N]``.
        layer: ``'layer1'`` (multi-head, default) or ``'layer2'`` (single head).

    Returns:
        head_matrices: List of ``[C, C]`` float64 arrays, one per head.
        cluster_ids: Sorted list of unique cluster label strings.
        cluster_sizes: Mapping ``{cluster_id: n_cells}``.
    """
    device = next(gat_model.parameters()).device
    x = torch.from_numpy(np.asarray(z0, dtype=np.float32)).to(device)
    edge_t = torch.from_numpy(np.asarray(edge_index, dtype=np.int64)).to(device)

    gat_model.eval()
    with torch.no_grad():
        _, _, attention_dict = gat_model(x, edge_t, return_attention=True)

    if attention_dict is None or layer not in attention_dict:
        raise ValueError(f"Layer '{layer}' not found in attention output.")

    alpha = attention_dict[layer].cpu().numpy()  # [E, heads]
    n_heads = alpha.shape[1]

    labels_str = np.array([str(lb) for lb in cluster_labels])
    unique_clusters = sorted(
        np.unique(labels_str).tolist(),
        key=lambda c: int(c) if c.isdigit() else c,
    )
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

    src, dst = edge_index[0], edge_index[1]
    src_ci = np.array([cluster_to_idx[labels_str[s]] for s in src], dtype=np.int64)
    dst_ci = np.array([cluster_to_idx[labels_str[d]] for d in dst], dtype=np.int64)

    cluster_sizes: dict[str, int] = {
        c: int(np.sum(labels_str == c)) for c in unique_clusters
    }

    head_matrices: list[np.ndarray] = []
    for h in range(n_heads):
        alpha_h = alpha[:, h]
        attn_sum = np.zeros((n_clusters, n_clusters), dtype=np.float64)
        count_mat = np.zeros((n_clusters, n_clusters), dtype=np.float64)
        np.add.at(attn_sum, (src_ci, dst_ci), alpha_h)
        np.add.at(count_mat, (src_ci, dst_ci), 1.0)
        mat = np.divide(
            attn_sum, count_mat, out=np.zeros_like(attn_sum), where=count_mat > 0
        )
        head_matrices.append(mat)

    return head_matrices, unique_clusters, cluster_sizes