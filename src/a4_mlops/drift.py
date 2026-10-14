"""
Drift detection for graph data (Assignment 4, Stage 4.10).

Two families of statistics:
  1. **Graph-structure**: mean degree, degree-distribution KS statistic,
     edge density, fraction of isolated nodes.
  2. **Embedding**: mean cosine similarity to a reference centroid, computed
     with a **fixed** model (never cross-version).

Thresholds are calibrated from a null distribution (bootstrap/resample of
the reference data) so that alerts are empirically grounded.
"""

from typing import Any

import numpy as np
import torch
from scipy import stats as scipy_stats
from torch import nn

# ---------------------------------------------------------------------------
# Graph-structure statistics
# ---------------------------------------------------------------------------


def graph_statistics(edge_index: torch.Tensor, num_nodes: int) -> dict[str, float]:
    """Compute summary statistics of a graph's structure.

    Args:
        edge_index: Shape ``(2, E)`` (directed, both directions).
        num_nodes: Total number of nodes ``N``.

    Returns:
        Dict with ``mean_degree``, ``edge_density``, ``frac_isolated``.
    """
    if edge_index.numel() == 0:
        return {
            "mean_degree": 0.0,
            "edge_density": 0.0,
            "frac_isolated": 1.0,
        }

    # Degree (undirected: count each direction once)
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))

    mean_deg = float(deg.mean().item())
    num_undirected_edges = edge_index.size(1) / 2
    max_edges = num_nodes * (num_nodes - 1) / 2
    density = num_undirected_edges / max_edges if max_edges > 0 else 0.0
    frac_isolated = float((deg == 0).float().mean().item())

    return {
        "mean_degree": mean_deg,
        "edge_density": density,
        "frac_isolated": frac_isolated,
    }


def degree_distribution(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Return the degree vector as a NumPy array."""
    deg = torch.zeros(num_nodes, dtype=torch.float)
    if edge_index.numel() > 0:
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
    return deg.numpy()


def degree_ks_statistic(
    ref_edge_index: torch.Tensor,
    test_edge_index: torch.Tensor,
    num_nodes: int,
) -> float:
    """Two-sample KS statistic between degree distributions.

    Args:
        ref_edge_index: Reference graph edge index.
        test_edge_index: Test graph edge index.
        num_nodes: Total nodes.

    Returns:
        KS statistic (higher = more different).
    """
    ref_deg = degree_distribution(ref_edge_index, num_nodes)
    test_deg = degree_distribution(test_edge_index, num_nodes)
    ks_stat, _ = scipy_stats.ks_2samp(ref_deg, test_deg)
    return float(ks_stat)


# ---------------------------------------------------------------------------
# Embedding statistics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_embeddings(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Run a GNN and return last hidden-layer embeddings.

    The model must have a ``get_embeddings`` method, or we use the full
    forward output.
    """
    model.eval()
    if hasattr(model, "get_embeddings"):
        return model.get_embeddings(x, edge_index)
    else:
        return model(x, edge_index)


def embedding_centroid_cosine(
    ref_embeddings: torch.Tensor,
    test_embeddings: torch.Tensor,
) -> float:
    """Mean cosine similarity of test embeddings to the reference centroid.

    A significant drop indicates drift in the embedding space.

    Args:
        ref_embeddings: Reference embeddings ``(N, D)``.
        test_embeddings: Test embeddings ``(N', D)``.

    Returns:
        Mean cosine similarity (higher = more similar to reference).
    """
    centroid = ref_embeddings.mean(dim=0, keepdim=True)  # (1, D)
    centroid_norm = centroid.norm(dim=1, keepdim=True).clamp(min=1e-8)
    centroid_unit = centroid / centroid_norm

    test_norms = test_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    test_unit = test_embeddings / test_norms

    cos_sims = (test_unit @ centroid_unit.t()).squeeze(1)
    return float(cos_sims.mean().item())


# ---------------------------------------------------------------------------
# Null-distribution calibration
# ---------------------------------------------------------------------------


def calibrate_threshold(
    ref_edge_index: torch.Tensor,
    ref_x: torch.Tensor,
    model: nn.Module,
    num_nodes: int,
    num_null_samples: int = 50,
    alert_quantile: float = 0.99,
    seed: int = 0,
) -> dict[str, float]:
    """Calibrate alert thresholds from a null distribution.

    Creates ``num_null_samples`` bootstrap subgraphs from the reference data,
    computes drift statistics for each, and sets thresholds at the
    ``alert_quantile`` of the null.

    Args:
        ref_edge_index: Reference edge index.
        ref_x: Reference features.
        model: Fixed model for embeddings.
        num_nodes: Total nodes.
        num_null_samples: Number of bootstrap samples.
        alert_quantile: Quantile for the threshold (e.g. 0.99).
        seed: Random seed.

    Returns:
        Dict with ``"ks_threshold"`` and ``"cosine_threshold"`` (lower bound).
    """
    rng = np.random.RandomState(seed)
    ref_emb = compute_embeddings(model, ref_x, ref_edge_index)

    ks_values: list[float] = []
    cos_values: list[float] = []

    for i in range(num_null_samples):
        # Bootstrap: randomly subsample 80% of nodes, keep their edges
        node_subset = rng.choice(num_nodes, size=int(0.8 * num_nodes), replace=False)
        node_set = set(node_subset.tolist())

        # Filter edges to those within the subset
        src = ref_edge_index[0].numpy()
        dst = ref_edge_index[1].numpy()
        mask = np.array([s in node_set and d in node_set for s, d in zip(src, dst)])

        if mask.sum() == 0:
            sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            sub_edge_index = ref_edge_index[:, torch.tensor(mask)]

        # KS stat
        ks = degree_ks_statistic(ref_edge_index, sub_edge_index, num_nodes)
        ks_values.append(ks)

        # Embedding cosine
        sub_emb = compute_embeddings(model, ref_x, sub_edge_index)
        cos = embedding_centroid_cosine(ref_emb, sub_emb)
        cos_values.append(cos)

    ks_threshold = float(np.quantile(ks_values, alert_quantile))
    # For cosine, lower is worse, so threshold is a lower bound
    cos_threshold = float(np.quantile(cos_values, 1 - alert_quantile))

    return {
        "ks_threshold": ks_threshold,
        "cosine_threshold": cos_threshold,
    }


def check_drift(
    ref_edge_index: torch.Tensor,
    test_edge_index: torch.Tensor,
    ref_x: torch.Tensor,
    test_x: torch.Tensor,
    model: nn.Module,
    num_nodes: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Check whether a test snapshot shows drift compared to the reference.

    Args:
        ref_edge_index, test_edge_index: Edge indices for reference/test.
        ref_x, test_x: Features for reference/test.
        model: Fixed model for embedding computation.
        num_nodes: Number of nodes.
        thresholds: Output of ``calibrate_threshold``.

    Returns:
        Dict with statistics, thresholds, and alert flags.
    """
    ks = degree_ks_statistic(ref_edge_index, test_edge_index, num_nodes)

    ref_emb = compute_embeddings(model, ref_x, ref_edge_index)
    test_emb = compute_embeddings(model, test_x, test_edge_index)
    cos = embedding_centroid_cosine(ref_emb, test_emb)

    ks_alert = ks > thresholds["ks_threshold"]
    cos_alert = cos < thresholds["cosine_threshold"]

    return {
        "ks_statistic": ks,
        "ks_threshold": thresholds["ks_threshold"],
        "ks_alert": ks_alert,
        "cosine_similarity": cos,
        "cosine_threshold": thresholds["cosine_threshold"],
        "cosine_alert": cos_alert,
        "any_alert": ks_alert or cos_alert,
    }
