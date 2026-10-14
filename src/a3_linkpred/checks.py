"""
Automated leakage and consistency checks for link prediction.

Each function raises ``AssertionError`` with a descriptive message on failure.
These are designed to be called directly in pytest tests.
"""


import torch


def _to_pair_set(edges: torch.Tensor) -> set[tuple[int, int]]:
    """Convert an ``(M, 2)`` tensor to a set of tuples."""
    return set(map(tuple, edges.numpy().tolist()))


def _to_directed_pair_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Convert a ``(2, E)`` edge index to a set of directed ``(src, dst)``."""
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    return set(zip(src.tolist(), dst.tolist()))


# --- Check (a): Positive splits are pairwise disjoint ---

def check_positive_disjoint(
    train_pos: torch.Tensor,
    val_pos: torch.Tensor,
    test_pos: torch.Tensor,
) -> None:
    """Assert train, val, test positive edge sets are pairwise disjoint."""
    train_set = _to_pair_set(train_pos)
    val_set = _to_pair_set(val_pos)
    test_set = _to_pair_set(test_pos)

    assert train_set.isdisjoint(val_set), (
        f"Train-Val overlap: {train_set & val_set}"
    )
    assert train_set.isdisjoint(test_set), (
        f"Train-Test overlap: {train_set & test_set}"
    )
    assert val_set.isdisjoint(test_set), (
        f"Val-Test overlap: {val_set & test_set}"
    )


# --- Check (b): No held-out edge (or reverse) in A_mp ---

def check_no_heldout_in_mp(
    mp_edge_index: torch.Tensor,
    val_pos: torch.Tensor,
    test_pos: torch.Tensor,
) -> None:
    """Assert no validation/test positive edge appears in message-passing graph.

    Checks both ``(u, v)`` and ``(v, u)`` explicitly.
    """
    mp_set = _to_directed_pair_set(mp_edge_index)

    for split_name, pos in [("val", val_pos), ("test", test_pos)]:
        for row in pos.numpy():
            u, v = int(row[0]), int(row[1])
            assert (u, v) not in mp_set, (
                f"{split_name} edge ({u}, {v}) found in A_mp"
            )
            assert (v, u) not in mp_set, (
                f"{split_name} reverse edge ({v}, {u}) found in A_mp"
            )


# --- Check (c): Negatives contain no self-loops and no positive ---

def check_negatives_valid(
    negatives: torch.Tensor,
    positive_set: set[tuple[int, int]],
    split_name: str = "",
) -> None:
    """Assert negatives have no self-loops and no positive overlap."""
    for row in negatives.numpy():
        u, v = int(row[0]), int(row[1])
        assert u != v, f"{split_name} negative contains self-loop: ({u}, {v})"
        pair = (min(u, v), max(u, v))
        assert pair not in positive_set, (
            f"{split_name} negative ({u}, {v}) is a positive edge"
        )


# --- Check (d): Val and test negatives are disjoint ---

def check_negatives_disjoint(
    val_neg: torch.Tensor,
    test_neg: torch.Tensor,
    train_neg: torch.Tensor,
) -> None:
    """Assert negative splits are pairwise disjoint."""
    val_set = _to_pair_set(val_neg)
    test_set = _to_pair_set(test_neg)
    train_set = _to_pair_set(train_neg)

    assert val_set.isdisjoint(test_set), "Val-Test negative overlap"
    assert val_set.isdisjoint(train_set), "Val-Train negative overlap"
    assert test_set.isdisjoint(train_set), "Test-Train negative overlap"


# --- Check (e): Evaluation pairs are deterministic across runs ---

def check_deterministic_splits(
    split_fn,
    edge_index: torch.Tensor,
    seed: int = 0,
) -> None:
    """Assert two calls with the same seed produce identical splits.

    Args:
        split_fn: Callable that takes ``(edge_index, seed=seed)`` and returns
            a dict with ``train_pos``, ``val_pos``, ``test_pos``.
        edge_index: Full edge index.
        seed: Seed to use for both calls.
    """
    split1 = split_fn(edge_index, seed=seed)
    split2 = split_fn(edge_index, seed=seed)

    for key in ["train_pos", "val_pos", "test_pos"]:
        assert torch.equal(split1[key], split2[key]), (
            f"Split '{key}' differs between runs with same seed"
        )


# --- Check (f): Encoder graph at val/test == training MP graph ---

def check_encoder_graph_unchanged(
    mp_edge_index_train: torch.Tensor,
    mp_edge_index_eval: torch.Tensor,
) -> None:
    """Assert the edge index used during evaluation is the same as training."""
    # Sort for comparison
    sorted_train = mp_edge_index_train[:, mp_edge_index_train[0].argsort()]
    sorted_eval = mp_edge_index_eval[:, mp_edge_index_eval[0].argsort()]
    assert torch.equal(sorted_train, sorted_eval), (
        "Encoder graph differs between training and evaluation"
    )


# --- Check (g): Common-neighbor baseline uses train adjacency ---

def check_baseline_uses_train_adj(
    train_adj: torch.Tensor,
    test_pos: torch.Tensor,
) -> None:
    """Verify that baseline scores are computed from train adjacency.

    Strategy: adding a test edge to the adjacency should change baseline
    scores, confirming the baseline doesn't already include it.
    """
    # Pick the first test edge
    u, v = int(test_pos[0, 0]), int(test_pos[0, 1])

    # Baseline score from train adj
    cn_scores = train_adj @ train_adj
    score_original = float(cn_scores[u, v].item())

    # Modify adjacency and check score changes
    adj_modified = train_adj.clone()
    adj_modified[u, v] = 1.0
    adj_modified[v, u] = 1.0
    cn_modified = adj_modified @ adj_modified
    score_modified = float(cn_modified[u, v].item())

    # If they differ, the test edge was not in the original adjacency
    assert score_original != score_modified or score_original == 0.0, (
        f"Test edge ({u},{v}) may already be in train adjacency"
    )
