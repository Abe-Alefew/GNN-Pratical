"""
Decoders for link prediction.

Two decoders are provided:
  1. **Dot-product decoder**: ``score(i,j) = z_i^T z_j``.  Symmetric by
     construction.
  2. **Symmetric MLP decoder**: uses ``[z_i ⊙ z_j || |z_i − z_j|]`` as
     input, which is symmetric under swapping ``i`` and ``j``.
"""

import torch
from torch import nn


class DotProductDecoder(nn.Module):
    """Dot-product decoder: ``σ(z_i^T z_j)``."""

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute link scores.

        Args:
            z_i: Embeddings of source nodes, ``(B, D)``.
            z_j: Embeddings of target nodes, ``(B, D)``.

        Returns:
            Raw scores (logits), shape ``(B,)``.
        """
        return (z_i * z_j).sum(dim=1)


class SymmetricMLPDecoder(nn.Module):
    """Symmetric MLP decoder using element-wise product and absolute difference.

    The input ``[z_i ⊙ z_j || |z_i − z_j|]`` is invariant to swapping
    ``i`` and ``j``, ensuring ``score(i,j) == score(j,i)``.

    Args:
        embedding_dim: Dimension of node embeddings ``D``.
        hidden_dim: Hidden layer width.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # Input: element-wise product (D) + absolute difference (D) = 2D
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute link scores.

        Args:
            z_i: Source embeddings ``(B, D)``.
            z_j: Target embeddings ``(B, D)``.

        Returns:
            Raw scores (logits), shape ``(B,)``.
        """
        product = z_i * z_j               # (B, D) — symmetric
        abs_diff = (z_i - z_j).abs()       # (B, D) — symmetric
        combined = torch.cat([product, abs_diff], dim=1)  # (B, 2D)
        return self.mlp(combined).squeeze(1)
