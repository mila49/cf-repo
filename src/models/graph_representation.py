import torch
from torch_geometric.data import Data
from scipy import sparse
import numpy as np
import anndata as ad


def build_gene_correlation_edges(
    adata: ad.AnnData,
    method: str = "topk",
    threshold: float = 0.4,
    top_k: int = 20,
    absolute: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build gene-gene correlation edges from adata.X.

    adata.X should be cells × genes, usually log-normalized HVG expression.

    Returns:
        edge_index: LongTensor with shape [2, num_edges]
        edge_attr: FloatTensor with shape [num_edges, 1]
    """

    X = adata.X

    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    # Correlation between genes, so rowvar=False because columns are genes.
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove self-loops for now.
    np.fill_diagonal(corr, 0.0)

    edge_sources = []
    edge_targets = []
    edge_weights = []

    if method == "threshold":
        score = np.abs(corr) if absolute else corr
        src, dst = np.where(score > threshold)

        edge_sources = src.tolist()
        edge_targets = dst.tolist()
        edge_weights = corr[src, dst].tolist()

    elif method == "topk":
        n_genes = corr.shape[0]

        for i in range(n_genes):
            scores = np.abs(corr[i]) if absolute else corr[i]

            # Get top-k correlated genes for gene i.
            candidate_idx = np.argsort(scores)[-top_k:]

            for j in candidate_idx:
                if i == j:
                    continue

                weight = corr[i, j]

                if weight == 0:
                    continue

                edge_sources.append(i)
                edge_targets.append(j)
                edge_weights.append(weight)

    else:
        raise ValueError("method must be either 'threshold' or 'topk'")

    edge_index = torch.tensor(
        [edge_sources, edge_targets],
        dtype=torch.long,
    )

    edge_attr = torch.tensor(
        edge_weights,
        dtype=torch.float32,
    ).view(-1, 1)

    return edge_index, edge_attr


def build_cell_graphs(
    adata: ad.AnnData,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    label_key: str | None = None,
) -> list[Data]:
    """
    Convert preprocessed AnnData into one PyTorch Geometric graph per cell.

    Each graph:
        nodes = genes
        node feature = expression of each gene in that cell
        edges = gene-gene correlation edges
        edge_attr = correlation coefficient
    """

    X = adata.X

    if sparse.issparse(X):
        X = X.tocsr()

    graphs = []

    labels = None
    if label_key is not None:
        labels = adata.obs[label_key].astype("category").cat.codes.values

    for cell_idx in range(adata.n_obs):
        if sparse.issparse(X):
            expr = X[cell_idx].toarray().ravel()
        else:
            expr = X[cell_idx].ravel()

        x = torch.tensor(expr, dtype=torch.float32).view(-1, 1)

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        graph.cell_id = adata.obs_names[cell_idx]

        if label_key is not None:
            graph.y = torch.tensor(labels[cell_idx], dtype=torch.long)

        graphs.append(graph)

    return graphs