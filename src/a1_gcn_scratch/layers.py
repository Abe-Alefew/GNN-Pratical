import torch
from torch import nn

# ---------------------------------------------------------------------------
# Graph normalisation utilities
# ---------------------------------------------------------------------------

def add_self_loops(adj: torch.Tensor) -> torch.Tensor:
    """Add self-loops to a dense adjacency matrix.

    Args:
        adj: Dense adjacency matrix, shape ``(N, N)``.

    Returns:
        ``Ã = A + I``, shape ``(N, N)``.
    """
    n = adj.size(0)
    return adj + torch.eye(n, dtype=adj.dtype, device=adj.device)


def compute_degree_vector(adj: torch.Tensor) -> torch.Tensor:
    """Row-wise degree vector from a (possibly self-looped) adjacency.

    Args:
        adj: Dense adjacency, shape ``(N, N)``.

    Returns:
        Degree vector, shape ``(N,)``.
    """
    return adj.sum(dim=1)


def compute_a_hat(adj: torch.Tensor) -> torch.Tensor:
    a_tilde = add_self_loops(adj)
    d_tilde = compute_degree_vector(a_tilde)

    # D̃^{-1/2} — avoid division by zero for isolated nodes
    d_inv_sqrt = torch.zeros_like(d_tilde)
    mask = d_tilde > 0
    d_inv_sqrt[mask] = d_tilde[mask].pow(-0.5)

    # Â = D̃^{-1/2} Ã D̃^{-1/2}  (outer product of d_inv_sqrt)
    a_hat = a_tilde * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)
    return a_hat


# ---------------------------------------------------------------------------
# Edge-index utilities
# ---------------------------------------------------------------------------

def add_self_loops_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Add self-loop edges ``(i, i)`` for all nodes.

    Args:
        edge_index: Shape ``(2, E)``.
        num_nodes: ``N``.

    Returns:
        Extended edge index, shape ``(2, E + N)``.
    """
    self_loops = torch.arange(num_nodes, device=edge_index.device)
    self_loops = self_loops.unsqueeze(0).expand(2, -1)  # (2, N)
    return torch.cat([edge_index, self_loops], dim=1)


def compute_a_hat_coeffs_sparse(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Compute the normalisation coefficient for each edge in edge_index.

    For edge (i, j): coeff = 1 / sqrt(d̃_i * d̃_j)

    Args:
        edge_index: Shape ``(2, E)`` **with self-loops already added**.
        num_nodes: ``N``.

    Returns:
        Coefficients, shape ``(E,)``.
    """
    src, dst = edge_index[0], edge_index[1]

    # Degree from edge index (with self-loops)
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    ones = torch.ones(src.size(0), dtype=torch.float32, device=edge_index.device)
    deg.index_add_(0, dst, ones)

    deg_inv_sqrt = torch.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = deg[mask].pow(-0.5)

    return deg_inv_sqrt[src] * deg_inv_sqrt[dst]


# ---------------------------------------------------------------------------
# GCN Layer
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """A single Graph Convolutional layer (Kipf & Welling 2017).

    Supports two forward paths:
      - **Dense** (``forward_dense``): uses the pre-computed Â matrix.
      - **Sparse** (``forward_sparse``): uses edge_index + scatter.

    The default ``forward`` dispatches based on the input type.

    Args:
        in_features: Input feature dimensionality.
        out_features: Output feature dimensionality.
        use_bias: Whether to add a learnable bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # --- Dense path ---

    def forward_dense(self, a_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Dense forward: Â @ (H @ W) + b.

        Args:
            a_hat: Normalised adjacency, shape ``(N, N)``.
            h: Node features, shape ``(N, F_in)``.

        Returns:
            Output features, shape ``(N, F_out)``.
        """
        support = h @ self.weight  # (N, F_out)
        out = a_hat @ support      # (N, F_out)
        if self.bias is not None:
            out = out + self.bias
        return out

    # --- Sparse (edge-index) path ---

    def forward_sparse(
        self,
        edge_index: torch.Tensor,
        coeffs: torch.Tensor,
        h: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Sparse forward using gather + index_add_.

        Args:
            edge_index: Shape ``(2, E)`` with self-loops.
            coeffs: Per-edge normalisation coefficients, shape ``(E,)``.
            h: Node features, shape ``(N, F_in)``.
            num_nodes: ``N``.

        Returns:
            Output features, shape ``(N, F_out)``.
        """
        src, dst = edge_index[0], edge_index[1]
        support = h @ self.weight  # (N, F_out)

        # Gather messages from source nodes
        messages = support[src] * coeffs.unsqueeze(1)  # (E, F_out)

        # Scatter-add into destination nodes
        out = torch.zeros(num_nodes, support.size(1),
                          dtype=support.dtype, device=support.device)
        out.index_add_(0, dst, messages)

        if self.bias is not None:
            out = out + self.bias
        return out

    # --- Default forward ---

    def forward(
        self,
        a_hat_or_edge_index: torch.Tensor,
        h: torch.Tensor,
        coeffs: torch.Tensor | None = None,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        """Dispatch to dense or sparse forward.

        Dense: pass ``a_hat`` as a 2-D tensor.
        Sparse: pass ``edge_index`` as a ``(2, E)`` tensor, with ``coeffs``
        and ``num_nodes``.
        """
        if a_hat_or_edge_index.dim() == 2 and a_hat_or_edge_index.size(0) != 2:
            # Dense mode (N × N)
            return self.forward_dense(a_hat_or_edge_index, h)
        else:
            # Sparse mode (2 × E)
            assert coeffs is not None and num_nodes is not None
            return self.forward_sparse(
                a_hat_or_edge_index, coeffs, h, num_nodes
            )
