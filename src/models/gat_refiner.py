from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    """Simple multi-head graph attention layer implemented with pure PyTorch.

    This layer expects directed edges in ``edge_index`` where each edge is
    ``source -> target`` and aggregates incoming messages at each target node.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        self.proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))

        if concat:
            self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        else:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply graph attention.

        Args:
            x: Node features with shape ``[num_nodes, in_dim]``.
            edge_index: Directed edge list with shape ``[2, num_edges]``.
            return_attention: Whether to return attention coefficients.

        Returns:
            Tuple of:
                - Updated node features
                - Attention coefficients ``[num_edges, heads]`` if requested
        """
        num_nodes = x.size(0)
        src, dst = edge_index

        h = self.proj(x).view(num_nodes, self.heads, self.out_dim)
        h = self.dropout(h)

        e_src = (h[src] * self.attn_src).sum(dim=-1)
        e_dst = (h[dst] * self.attn_dst).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)

        # Segment-softmax over incoming edges per destination node.
        # Clamp improves numerical stability when graphs are large.
        exp_e = torch.exp(torch.clamp(e, min=-10.0, max=10.0))
        denom = torch.zeros(num_nodes, self.heads, device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-12)
        alpha = self.dropout(alpha)

        messages = alpha.unsqueeze(-1) * h[src]
        out = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, messages)

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_dim) + self.bias
        else:
            out = out.mean(dim=1) + self.bias

        if return_attention:
            return out, alpha
        return out, None


class GATRefiner(nn.Module):
    """Graph-based embedding refiner.

    Input node features are initial embeddings ``Z0`` and output embeddings are
    refined embeddings ``Z1``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.gat1 = GraphAttentionLayer(
            in_dim=input_dim,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.gat2 = GraphAttentionLayer(
            in_dim=hidden_dim * heads,
            out_dim=out_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.reconstruction_head = nn.Linear(out_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Run refinement and reconstruction.

        Returns:
            Tuple of:
                - Refined embeddings ``Z1`` with shape ``[n_cells, out_dim]``
                - Reconstruction of ``Z0`` with shape ``[n_cells, input_dim]``
                - Optional attention dictionary with per-layer attention weights
        """
        z, attn_1 = self.gat1(x, edge_index, return_attention=return_attention)
        z = self.activation(z)
        z = self.dropout(z)

        z1, attn_2 = self.gat2(z, edge_index, return_attention=return_attention)
        z0_hat = self.reconstruction_head(z1)

        if return_attention:
            return z1, z0_hat, {"layer1": attn_1, "layer2": attn_2}
        return z1, z0_hat, None