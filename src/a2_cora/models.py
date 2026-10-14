"""
Model architectures for Assignment 2 (Cora benchmark).

Four ``nn.Module`` classes with a shared ``forward(x, edge_index)`` interface:
  - **MLP**: ignores graph structure entirely.
  - **GCN**: ``GCNConv`` from PyG.
  - **GraphSAGE**: ``SAGEConv`` from PyG.
  - **GAT**: ``GATConv`` from PyG (multi-head attention).

Using PyG conv layers is allowed for A2 (the from-scratch requirement is A1).
"""


import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

# ---------------------------------------------------------------------------
# MLP (no message passing)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Multi-layer perceptron — no graph structure.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer width.
        out_channels: Number of output classes.
        num_layers: Total linear layers (≥ 2).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass (edge_index is accepted but ignored).

        Returns log-probabilities, shape ``(N, C)``.
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """Graph Convolutional Network using PyG's ``GCNConv``.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer width.
        out_channels: Number of output classes.
        num_layers: Number of GCN layers (≥ 2).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities, shape ``(N, C)``."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return the last hidden-layer embeddings (before the final conv)."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# GraphSAGE
# ---------------------------------------------------------------------------

class GraphSAGE(nn.Module):
    """GraphSAGE using PyG's ``SAGEConv``.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer width.
        out_channels: Number of output classes.
        num_layers: Number of SAGE layers.
        dropout: Dropout probability.
        aggr: Aggregation method (``"mean"``, ``"max"``, ``"sum"``).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(
                    SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
                )
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities, shape ``(N, C)``."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return the last hidden-layer embeddings."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# GAT
# ---------------------------------------------------------------------------

class GAT(nn.Module):
    """Graph Attention Network using PyG's ``GATConv``.

    Hidden layers use concatenated multi-head attention; the output layer
    averages heads.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Per-head hidden dimension.
        out_channels: Number of output classes.
        num_layers: Number of GAT layers.
        dropout: Dropout probability (also used for attention dropout).
        heads: Number of attention heads.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1, concat=False,
                        dropout=dropout)
            )
        else:
            # First layer: multi-head, concatenate
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads,
                        concat=True, dropout=dropout)
            )
            # Middle layers: input dim = hidden_channels * heads
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels,
                            heads=heads, concat=True, dropout=dropout)
                )
            # Output layer: average heads
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels, heads=1,
                        concat=False, dropout=dropout)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities, shape ``(N, C)``."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return the last hidden-layer embeddings."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "MLP": MLP,
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Args:
        name: One of ``"MLP"``, ``"GCN"``, ``"GraphSAGE"``, ``"GAT"``.
        **kwargs: Constructor arguments.

    Returns:
        A model instance.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name!r}. Choose from {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)
