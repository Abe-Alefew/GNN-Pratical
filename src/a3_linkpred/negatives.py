"""
Negative sampling for link prediction.

Valid negatives: ``u != v``, and the pair ``(min(u,v), max(u,v))`` is not in
the full positive set (train + val + test).  This prevents "negatives" that
are actually real held-out links.

Val/test negatives are sampled once with a fixed seed, stored, and reused
across all experiments.  Training negatives may be resampled each epoch.
"""


import numpy as np
import torch


def _canonicalize(u: int, v: int) -> tuple[int, int]:
    """Return the pair with min first."""
    return (min(u, v), max(u, v))


def build_positive_set(
    *splits: torch.Tensor,
) -> set[tuple[int, int]]:
    """Build the set of all positive pairs across splits.

    Each input tensor has shape ``(M, 2)`` with undirected pairs ``(u, v)``.

    Returns:
        Set of canonicalised ``(min, max)`` tuples.
    """
    pos_set: set[tuple[int, int]] = set()
    for split in splits:
        for row in split.numpy():
            pos_set.add(_canonicalize(int(row[0]), int(row[1])))
    return pos_set


def sample_negatives(
    num_nodes: int,
    num_samples: int,
    positive_set: set[tuple[int, int]],
    exclude_set: set[tuple[int, int]] | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Sample negative (non-edge) pairs.

    Args:
        num_nodes: Total number of nodes ``N``.
        num_samples: Number of negative pairs to sample.
        positive_set: Set of all observed positive pairs (to avoid).
        exclude_set: Additional pairs to exclude (e.g. negatives from
            other splits, to ensure disjointness).
        seed: Random seed.

    Returns:
        Tensor of shape ``(num_samples, 2)`` with canonicalised negative
        pairs.
    """
    rng = np.random.RandomState(seed)
    all_exclude = positive_set.copy()
    if exclude_set is not None:
        all_exclude |= exclude_set

    negatives: list = []
    seen: set[tuple[int, int]] = set()

    max_attempts = num_samples * 20  # safety limit
    attempts = 0

    while len(negatives) < num_samples and attempts < max_attempts:
        u = rng.randint(0, num_nodes)
        v = rng.randint(0, num_nodes)
        if u == v:
            attempts += 1
            continue

        pair = _canonicalize(u, v)
        if pair in all_exclude or pair in seen:
            attempts += 1
            continue

        negatives.append(pair)
        seen.add(pair)
        attempts += 1

    if len(negatives) < num_samples:
        raise RuntimeError(
            f"Could only sample {len(negatives)}/{num_samples} negatives "
            f"after {max_attempts} attempts."
        )

    return torch.tensor(negatives, dtype=torch.long)


def sample_all_negatives(
    num_nodes: int,
    train_pos: torch.Tensor,
    val_pos: torch.Tensor,
    test_pos: torch.Tensor,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample negatives for val and test (fixed) and training (one batch).

    Val/test get 1:1 ratio.  Negatives are disjoint across splits.

    Args:
        num_nodes: Total nodes.
        train_pos, val_pos, test_pos: Positive edge splits ``(M, 2)``.
        seed: Random seed.

    Returns:
        ``(train_neg, val_neg, test_neg)`` each shape ``(M_split, 2)``.
    """
    pos_set = build_positive_set(train_pos, val_pos, test_pos)

    # Val negatives first
    val_neg = sample_negatives(
        num_nodes, len(val_pos), pos_set, seed=seed
    )
    val_neg_set = set(map(tuple, val_neg.numpy().tolist()))

    # Test negatives (exclude val negatives)
    test_neg = sample_negatives(
        num_nodes, len(test_pos), pos_set,
        exclude_set=val_neg_set, seed=seed + 1
    )
    test_neg_set = set(map(tuple, test_neg.numpy().tolist()))

    # Train negatives (exclude val + test negatives)
    train_neg = sample_negatives(
        num_nodes, len(train_pos), pos_set,
        exclude_set=val_neg_set | test_neg_set, seed=seed + 2
    )

    return train_neg, val_neg, test_neg
