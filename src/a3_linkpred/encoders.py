"""
GNN encoders for link prediction.

Produces node embeddings ``z_i = GNN(X, A_mp)_i`` using only the
message-passing graph (training edges).  The same encoder is used for
training, validation, and test — the graph does not change.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


class LinkPredEncoder(nn.Module):
    """Two-layer GNN encoder for link prediction.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer width.
        out_channels: Embedding dimension.
        encoder_type: ``"GCN"`` or ``"SAGE"``.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        encoder_type: str = "GCN",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        ConvClass = GCNConv if encoder_type == "GCN" else SAGEConv
        self.conv1 = ConvClass(in_channels, hidden_channels)
        self.conv2 = ConvClass(hidden_channels, out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Produce node embeddings.

        Args:
            x: Node features ``(N, F)``.
            edge_index: Message-passing edge index ``(2, E_mp)``.

        Returns:
            Node embeddings ``(N, D)``.
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h
