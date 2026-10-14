"""
Edge splitting for link prediction.

Converts a PyG-style directed edge_index (both directions of each undirected
edge) into train / validation / test sets of unique undirected pairs
``(u, v)`` with ``u < v``.

The message-passing graph ``A_mp`` contains *only* training edges (both
directions).  Held-out edges are never in ``A_mp``.
"""


import numpy as np
import torch


def to_undirected_pairs(edge_index: torch.Tensor) -> np.ndarray:
    """Extract unique undirected edges from a directed edge_index.

    Each undirected pair appears once as ``(min, max)``.  Self-loops are
    dropped.

    Args:
        edge_index: Shape ``(2, E)``.

    Returns:
        Array of shape ``(M, 2)`` with ``M`` unique undirected edges.
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    pairs = set()
    for s, d in zip(src, dst):
        if s == d:
            continue  # drop self-loops
        u, v = min(s, d), max(s, d)
        pairs.add((u, v))

    return np.array(sorted(pairs), dtype=np.int64)


def split_edges(
    edge_index: torch.Tensor,
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Split edges into train / val / test.

    Operates on unique undirected pairs.  The message-passing graph contains
    training edges in both directions.

    Args:
        edge_index: Full graph edge index, shape ``(2, E)``.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed for shuffling.

    Returns:
        Dictionary with:
          - ``"train_pos"``: ``(M_train, 2)`` undirected pairs.
          - ``"val_pos"``: ``(M_val, 2)`` undirected pairs.
          - ``"test_pos"``: ``(M_test, 2)`` undirected pairs.
          - ``"mp_edge_index"``: ``(2, 2*M_train)`` directed edges for
            message passing.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    pairs = to_undirected_pairs(edge_index)
    rng = np.random.RandomState(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pos = pairs[:n_train]
    val_pos = pairs[n_train : n_train + n_val]
    test_pos = pairs[n_train + n_val :]

    # Build message-passing edge index (both directions of train edges)
    mp_src = np.concatenate([train_pos[:, 0], train_pos[:, 1]])
    mp_dst = np.concatenate([train_pos[:, 1], train_pos[:, 0]])
    mp_edge_index = torch.tensor(
        np.stack([mp_src, mp_dst], axis=0), dtype=torch.long
    )

    return {
        "train_pos": torch.tensor(train_pos, dtype=torch.long),
        "val_pos": torch.tensor(val_pos, dtype=torch.long),
        "test_pos": torch.tensor(test_pos, dtype=torch.long),
        "mp_edge_index": mp_edge_index,
    }
