from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx


# Fixed palette for the known cell types; unknown types fall back to Set2.
_CELL_TYPE_PALETTE: dict[str, str] = {
    "Basal":     "#E07B54",
    "Secretory": "#4BACC6",
    "Ciliated":  "#70AD47",
    "FOXN4+":    "#9B59B6",
    "NE":        "#F1C40F",
}


def _arc_label_pos(
    p1: tuple[float, float],
    p2: tuple[float, float],
    rad: float = 0.18,
) -> tuple[float, float]:
    """Return the visual midpoint of an arc3 bezier edge for label placement.

    For a quadratic bezier with ``connectionstyle='arc3,rad=rad'``, the visual
    midpoint lies at the straight-line midpoint offset by ``0.5 * rad * d`` in
    the perpendicular direction (left of the edge direction vector).
    """
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = math.sqrt(dx * dx + dy * dy)
    if d > 0:
        # Perpendicular unit vector (90° counter-clockwise from edge direction)
        px, py = -dy / d, dx / d
    else:
        px, py = 0.0, 0.0
    offset = 0.5 * rad * d
    return mx + px * offset, my + py * offset


def _axis_labels(
    cluster_ids: list[str],
    signature_per_cluster: dict[str, str] | None,
) -> list[str]:
    """Return tick labels combining cluster ID and cell type, e.g. '0 (Basal)'."""
    if signature_per_cluster:
        return [f"{c} ({signature_per_cluster.get(c, '?')})" for c in cluster_ids]
    return list(cluster_ids)


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    cluster_ids: list[str],
    signature_per_cluster: dict[str, str] | None = None,
    output_path: str | None = None,
) -> None:
    """Seaborn heatmap of the C×C cluster attention matrix.

    Rows = source cluster (pays attention).
    Columns = destination cluster (receives attention).
    Diagonal = within-cluster cohesion.

    Args:
        attention_matrix: ``[C, C]`` mean attention weights.
        cluster_ids: Ordered list of cluster label strings (matches matrix rows/cols).
        signature_per_cluster: Optional ``{cluster_id: cell_type}`` for axis labels.
        output_path: If given, saves the figure to this path.
    """
    labels = _axis_labels(cluster_ids, signature_per_cluster)
    n = len(cluster_ids)
    size = max(8, n * 0.85)

    fig, ax = plt.subplots(figsize=(size + 2, size))

    annot = n <= 20
    sns.heatmap(
        attention_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="coolwarm",
        vmin=0,
        vmax=0.1,
        annot=annot,
        fmt=".3f" if annot else "",
        linewidths=0.4,
        linecolor="lightgray",
        ax=ax,
        cbar_kws={"label": "Mean Attention Weight", "shrink": 0.8},
    )

    ax.set_xlabel("Destination Cluster  (receives attention)", fontsize=11)
    ax.set_ylabel("Source Cluster  (pays attention)", fontsize=11)
    ax.set_title(
        "GAT Cluster Attention Matrix\n(diagonal = within-cluster cohesion)",
        fontsize=13,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved attention heatmap → {output_path}")
    plt.close(fig)


def plot_attention_graph(
    attention_matrix: np.ndarray,
    cluster_ids: list[str],
    cluster_sizes: dict[str, int],
    signature_per_cluster: dict[str, str] | None = None,
    output_path: str | None = None,
    threshold: float = 0.3,
) -> None:
    """Directed graph where each node is a cluster and edges encode mean attention.

    Edge thickness and opacity scale with attention weight.
    Edges with ``attention < threshold × max(attention)`` are omitted for readability.
    Self-loops (within-cluster attention) are excluded here; see the heatmap diagonal.

    Args:
        attention_matrix: ``[C, C]`` mean attention weights.
        cluster_ids: Ordered list of cluster label strings.
        cluster_sizes: ``{cluster_id: n_cells}`` for sizing nodes.
        signature_per_cluster: Optional ``{cluster_id: cell_type}`` for node colors.
        output_path: If given, saves the figure to this path.
        threshold: Fraction of max attention below which edges are hidden (0–1).
    """
    G = nx.DiGraph()
    G.add_nodes_from(cluster_ids)

    max_attn = float(attention_matrix.max())
    cutoff = threshold * max_attn if max_attn > 0 else 0.0

    for i, ci in enumerate(cluster_ids):
        for j, cj in enumerate(cluster_ids):
            if i == j:
                continue
            val = float(attention_matrix[i, j])
            if val >= cutoff:
                G.add_edge(ci, cj, weight=val)

    # Layout: circular for many clusters, spring for few
    n = len(cluster_ids)
    if n >= 12:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, n ** 0.5))

    # Node colors: use fixed cell-type palette when node names are known types,
    # falling back to signature_per_cluster lookup, then tab10.
    sig_to_color: dict[str, tuple] = {}
    unique_sigs: list[str] = []
    if signature_per_cluster:
        unique_sigs = sorted(set(signature_per_cluster.values()))
        fallback = sns.color_palette("Set2", len(unique_sigs))
        sig_to_color = {
            sig: _CELL_TYPE_PALETTE.get(sig, fallback[i % len(fallback)])
            for i, sig in enumerate(unique_sigs)
        }
        node_colors = [
            sig_to_color.get(signature_per_cluster.get(c, ""), "lightgray")
            for c in G.nodes()
        ]
    else:
        # If cluster IDs are themselves cell-type names, use the fixed palette.
        node_colors = []
        for c in G.nodes():
            if c in _CELL_TYPE_PALETTE:
                node_colors.append(_CELL_TYPE_PALETTE[c])
                if c not in sig_to_color:
                    sig_to_color[c] = _CELL_TYPE_PALETTE[c]
                    unique_sigs.append(c)
            else:
                node_colors.append("steelblue")
        unique_sigs = sorted(unique_sigs)

    # Node sizes proportional to cluster population
    max_sz = max(cluster_sizes.values()) if cluster_sizes else 1
    node_sizes = [
        800 + 2500 * cluster_sizes.get(c, 1) / max_sz for c in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    ax.margins(0.18)  # prevent nodes/labels from being clipped at figure edges

    # Draw each edge individually to allow per-edge width and alpha,
    # then place labels at the visual midpoint of each arc.
    edges = list(G.edges(data="weight"))
    if edges:
        max_w = max(w for _, _, w in edges)
        for u, v, w in edges:
            norm = w / max_w if max_w > 0 else 0.0
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=0.8 + 2.0 * norm,
                alpha=1.0,
                edge_color="black",
                arrows=True,
                arrowsize=22,
                node_size=node_sizes,   # lets networkx land the arrowhead at the node boundary
                connectionstyle="arc3,rad=0.18",
                ax=ax,
            )
            # Label at the visual midpoint of the arc (opposite sides for u→v vs v→u)
            lx, ly = _arc_label_pos(pos[u], pos[v], rad=0.18)
            ax.text(
                lx, ly, f"{w:.3f}",
                fontsize=12, ha="center", va="center",
                color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.9, ec="none"),
            )

    # Cell-type legend
    if sig_to_color and unique_sigs:
        handles = [
            mpatches.Patch(color=sig_to_color[sig], label=sig)
            for sig in unique_sigs
        ]
        ax.legend(
            handles=handles,
            title="Cell Type",
            loc="lower left",
            fontsize=9,
            framealpha=0.8,
        )

    ax.set_title(
        "GAT Cluster Attention Graph\n"
        f"(edges ≥ {threshold:.0%} of max  |  arrow width ∝ attention weight  |  self-loops in heatmap)",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved attention graph     → {output_path}")
    plt.close(fig)


def plot_per_head_heatmaps(
    head_matrices: list[np.ndarray],
    cluster_ids: list[str],
    layer_name: str = "layer1",
    output_path: str | None = None,
    vmax: float = 0.1,
) -> None:
    """Grid of heatmaps — one subplot per attention head.

    Reveals whether individual heads specialise on specific cell-type
    interactions rather than treating all neighbours equally.

    Args:
        head_matrices: List of ``[C, C]`` mean attention arrays (one per head).
        cluster_ids: Ordered list of cluster label strings.
        layer_name: Used in the figure suptitle.
        output_path: If given, saves the figure to this path.
        vmax: Upper bound for the shared colour scale.
    """
    n_heads = len(head_matrices)
    n_cols = min(4, n_heads)
    n_rows = math.ceil(n_heads / n_cols)
    labels = list(cluster_ids)
    n = len(cluster_ids)
    cell_size = max(2.5, n * 0.7)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * (cell_size + 1.5), n_rows * cell_size),
        squeeze=False,
    )

    for h, mat in enumerate(head_matrices):
        row, col = divmod(h, n_cols)
        ax = axes[row][col]
        sns.heatmap(
            mat,
            xticklabels=labels,
            yticklabels=labels,
            cmap="coolwarm",
            vmin=0,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            linewidths=0.3,
            linecolor="lightgray",
            ax=ax,
            cbar=True,
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title(f"Head {h + 1}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    # Hide any unused subplot slots
    for h in range(n_heads, n_rows * n_cols):
        row, col = divmod(h, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Per-Head Attention Matrices  ({layer_name})\n"
        "Rows = source cell type  |  Columns = destination cell type",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved per-head heatmaps  → {output_path}")
    plt.close(fig)