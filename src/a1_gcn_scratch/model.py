"""
Two-layer GCN and MLP models for Assignment 1.

Both models share the same ``forward`` signature so they can be trained
with the same loop.  The GCN accepts either a dense Â or a sparse
edge-index representation; the MLP ignores graph structure entirely
(equivalent to setting Â = I).

Neither model uses any ready-made GNN layer (PyG / DGL).
"""


import torch
import torch.nn.functional as F
from torch import nn

from src.a1_gcn_scratch.layers import (
    GCNLayer,
    add_self_loops_edge_index,
    compute_a_hat,
    compute_a_hat_coeffs_sparse,
)


class GCNScratch(nn.Module):
    """Two-layer GCN (Kipf & Welling 2017), implemented from scratch.

    Z = softmax( Â · ReLU( Â · X · W0 ) · W1 )

    Args:
        in_features: Input feature dimension ``F``.
        hidden_dim: Hidden layer dimension.
        num_classes: Number of output classes.
        dropout: Dropout probability applied between layers.
        use_bias: Whether layers include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_dim, use_bias=use_bias)
        self.layer2 = GCNLayer(hidden_dim, num_classes, use_bias=use_bias)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        use_sparse: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features, shape ``(N, F)``.
            adj: Dense adjacency matrix (without self-loops), shape ``(N, N)``.
                Used to compute Â for the dense path, or ignored if
                ``use_sparse=True``.
            edge_index: Optional edge index, shape ``(2, E)``.  Required when
                ``use_sparse=True``.
            use_sparse: If True, use the edge-index scatter path.

        Returns:
            Log-probabilities, shape ``(N, C)``.
        """
        num_nodes = x.size(0)

        if use_sparse:
            assert edge_index is not None, "edge_index required for sparse mode"
            ei = add_self_loops_edge_index(edge_index, num_nodes)
            coeffs = compute_a_hat_coeffs_sparse(ei, num_nodes)

            h = self.layer1.forward_sparse(ei, coeffs, x, num_nodes)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layer2.forward_sparse(ei, coeffs, h, num_nodes)
        else:
            a_hat = compute_a_hat(adj)

            h = self.layer1.forward_dense(a_hat, x)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layer2.forward_dense(a_hat, h)

        return F.log_softmax(h, dim=1)

    def get_embeddings(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Return the hidden-layer embeddings (before the final layer).

        Useful for similarity analysis (oversmoothing checks).
        """
        a_hat = compute_a_hat(adj)
        h = self.layer1.forward_dense(a_hat, x)
        h = F.relu(h)
        return h


class MLPBaseline(nn.Module):
    """Two-layer MLP baseline that ignores graph structure.

    Equivalent to a GCN with Â = I (no message passing).

    Args:
        in_features: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        use_sparse: bool = False,
    ) -> torch.Tensor:
        """Forward pass (adj and edge_index are accepted but ignored).

        Args:
            x: Node features, shape ``(N, F)``.
            adj: Ignored (accepted for interface compatibility).
            edge_index: Ignored.
            use_sparse: Ignored.

        Returns:
            Log-probabilities, shape ``(N, C)``.
        """
        h = self.fc1(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)
        return F.log_softmax(h, dim=1)
