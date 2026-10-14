"""
Experiments for Assignment 2: depth study and edge-removal study.

Each experiment trains models across a parameter sweep and collects
metrics for analysis and plotting.
"""

from typing import Any

import numpy as np
import torch

from src.a2_cora.models import build_model
from src.a2_cora.train import train_one_run
from src.common.seed import set_seed

# ---------------------------------------------------------------------------
# Embedding similarity metrics (oversmoothing analysis)
# ---------------------------------------------------------------------------


def mean_pairwise_cosine(embeddings: torch.Tensor) -> float:
    """Mean pairwise cosine similarity of node embeddings.

    Args:
        embeddings: Shape ``(N, D)``.

    Returns:
        Mean cosine similarity (higher = more similar = more smoothed).
    """
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = embeddings / norms
    sim_matrix = normed @ normed.t()
    n = sim_matrix.size(0)
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    return float(sim_matrix[mask].mean().item())


def dirichlet_energy(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> float:
    """Dirichlet energy: measures smoothness of embeddings over the graph.

    E(H) = (1/N) Σ_{(i,j)∈E} || h_i/√d̃_i − h_j/√d̃_j ||²

    Lower values indicate more oversmoothing.

    Args:
        embeddings: Shape ``(N, D)``.
        edge_index: Shape ``(2, E)`` (with self-loops for d̃).
        num_nodes: ``N``.

    Returns:
        Dirichlet energy (non-negative scalar).
    """
    src, dst = edge_index[0], edge_index[1]

    # Compute degree (including self-loops)
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    deg_inv_sqrt = (deg + 1).pow(-0.5)  # +1 for self-loop approximation

    # Scale embeddings
    scaled = embeddings * deg_inv_sqrt.unsqueeze(1)

    diff = scaled[src] - scaled[dst]
    energy = (diff ** 2).sum(dim=1).mean()
    return float(energy.item())


def class_wise_similarity(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    """Compute mean cosine similarity within-class and between-class.

    Args:
        embeddings: Shape ``(N, D)``.
        labels: Shape ``(N,)``.

    Returns:
        Tuple of (within_class_sim, between_class_sim).
    """
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = embeddings / norms
    sim = normed @ normed.t()
    n = sim.size(0)

    label_match = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)

    within_mask = label_match & diag_mask
    between_mask = ~label_match & diag_mask

    within_sim = float(sim[within_mask].mean().item()) if within_mask.any() else 0.0
    between_sim = float(sim[between_mask].mean().item()) if between_mask.any() else 0.0

    return within_sim, between_sim


# ---------------------------------------------------------------------------
# Depth experiment
# ---------------------------------------------------------------------------


def run_depth_experiment(
    model_name: str,
    depths: list[int],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    dropout: float = 0.5,
    seeds: list[int] = (0, 1, 2),
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 300,
    patience: int = 30,
) -> list[dict[str, Any]]:
    """Vary model depth and record accuracy + embedding similarity.

    Args:
        model_name: ``"GCN"``, ``"GraphSAGE"``, or ``"GAT"``.
        depths: List of layer counts to try.
        (other args): Standard data and training parameters.

    Returns:
        List of result dicts with depth, seed, test metrics, and similarity
        measures.
    """
    results: list[dict[str, Any]] = []

    for depth in depths:
        for seed in seeds:
            model_kwargs = {
                "in_channels": in_channels,
                "hidden_channels": hidden_channels,
                "out_channels": out_channels,
                "num_layers": depth,
                "dropout": dropout,
            }
            if model_name == "GAT":
                model_kwargs["heads"] = 8

            model = build_model(model_name, **model_kwargs)

            run_result = train_one_run(
                model, x, edge_index, labels,
                train_mask, val_mask, test_mask,
                lr=lr, weight_decay=weight_decay,
                epochs=epochs, patience=patience,
                seed=seed,
            )

            # Compute embedding metrics
            model.eval()
            with torch.no_grad():
                if hasattr(model, "get_embeddings"):
                    emb = model.get_embeddings(x, edge_index)
                else:
                    emb = x  # fallback

            cos_sim = mean_pairwise_cosine(emb)
            d_energy = dirichlet_energy(emb, edge_index, x.size(0))
            within_sim, between_sim = class_wise_similarity(emb, labels)

            results.append({
                "model": model_name,
                "depth": depth,
                "seed": seed,
                "test_accuracy": run_result["test_metrics"]["accuracy"],
                "test_macro_f1": run_result["test_metrics"]["macro_f1"],
                "mean_cosine_sim": cos_sim,
                "dirichlet_energy": d_energy,
                "within_class_sim": within_sim,
                "between_class_sim": between_sim,
            })

    return results


# ---------------------------------------------------------------------------
# Edge-removal experiment
# ---------------------------------------------------------------------------


def remove_edges(
    edge_index: torch.Tensor,
    fraction: float,
    seed: int = 0,
) -> torch.Tensor:
    """Remove a fraction of undirected edges (both directions together).

    Args:
        edge_index: Shape ``(2, E)`` (directed, both directions present).
        fraction: Fraction of undirected edges to remove.
        seed: Random seed.

    Returns:
        Reduced edge_index with specified fraction removed.
    """
    if fraction <= 0.0:
        return edge_index.clone()
    if fraction >= 1.0:
        return torch.zeros((2, 0), dtype=edge_index.dtype)

    rng = np.random.RandomState(seed)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    # Build unique undirected edges (u < v)
    undirected = set()
    for s, d in zip(src, dst):
        u, v = min(s, d), max(s, d)
        undirected.add((u, v))

    undirected_list = sorted(undirected)
    num_remove = int(len(undirected_list) * fraction)
    remove_indices = set(
        rng.choice(len(undirected_list), size=num_remove, replace=False)
    )

    # Keep edges not in the remove set
    keep_edges = []
    for idx, (u, v) in enumerate(undirected_list):
        if idx not in remove_indices:
            keep_edges.append([u, v])
            keep_edges.append([v, u])

    if not keep_edges:
        return torch.zeros((2, 0), dtype=edge_index.dtype)

    keep_edges = np.array(keep_edges, dtype=np.int64).T
    return torch.tensor(keep_edges, dtype=torch.long)


def run_edge_removal_experiment(
    model_names: list[str],
    fractions: list[float],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    dropout: float = 0.5,
    seeds: list[int] = (0, 1, 2),
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 300,
    patience: int = 30,
) -> list[dict[str, Any]]:
    """Remove edges at various fractions and evaluate all models.

    The MLP is included as a flat control line.

    Returns:
        List of result dicts.
    """
    results: list[dict[str, Any]] = []

    for frac in fractions:
        for seed in seeds:
            # Perturb edges (remove consistently for train and test)
            ei_perturbed = remove_edges(edge_index, fraction=frac, seed=seed)

            for model_name in model_names:
                set_seed(seed)

                model_kwargs = {
                    "in_channels": in_channels,
                    "hidden_channels": hidden_channels,
                    "out_channels": out_channels,
                    "num_layers": 2,
                    "dropout": dropout,
                }
                if model_name == "GAT":
                    model_kwargs["heads"] = 8

                model = build_model(model_name, **model_kwargs)

                run_result = train_one_run(
                    model, x, ei_perturbed, labels,
                    train_mask, val_mask, test_mask,
                    lr=lr, weight_decay=weight_decay,
                    epochs=epochs, patience=patience,
                    seed=seed,
                )

                results.append({
                    "model": model_name,
                    "edge_removal_fraction": frac,
                    "seed": seed,
                    "test_accuracy": run_result["test_metrics"]["accuracy"],
                    "test_macro_f1": run_result["test_metrics"]["macro_f1"],
                })

    return results


def main() -> None:
    """CLI entrypoint for running Assignment 2 Experiments (Depth & Edge Removal)."""
    from torch_geometric.datasets import Planetoid

    print("=" * 70)
    print("  ASSIGNMENT 2 EXPERIMENTS — DEPTH & EDGE REMOVAL ABLATIONS")
    print("=" * 70)

    dataset = Planetoid(root="data/Cora", name="Cora")
    data = dataset[0]

    # 1. Depth Experiment
    print("\n[1] Running GCN Depth / Oversmoothing Experiment L in [1, 2, 4, 8]...")
    depth_res = run_depth_experiment(
        model_name="GCN",
        depths=[1, 2, 4, 8],
        x=data.x,
        edge_index=data.edge_index,
        labels=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        in_channels=data.num_features,
        out_channels=dataset.num_classes,
        hidden_channels=64,
        seeds=[42],
        epochs=100,
        patience=15,
    )
    print("    Depth | Test Acc | Cosine Sim | Dirichlet Energy")
    print("    ------|----------|------------|-----------------")
    for r in depth_res:
        print(f"    {r['depth']:>5d} | {r['test_accuracy']:>8.2%} | {r['mean_cosine_sim']:>10.4f} | {r['dirichlet_energy']:>16.4f}")

    # 2. Edge Removal Experiment
    print("\n[2] Running Edge Removal Experiment (0%, 25%, 50%, 75%, 100%)...")
    edge_res = run_edge_removal_experiment(
        model_names=["MLP", "GCN"],
        fractions=[0.0, 0.25, 0.50, 0.75, 1.0],
        x=data.x,
        edge_index=data.edge_index,
        labels=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        in_channels=data.num_features,
        out_channels=dataset.num_classes,
        hidden_channels=64,
        seeds=[42],
        epochs=100,
        patience=15,
    )
    print("    Fraction Removed | Model | Test Acc")
    print("    -----------------|-------|---------")
    for r in edge_res:
        print(f"    {r['edge_removal_fraction']:>16.0%} | {r['model']:<5} | {r['test_accuracy']:>8.2%}")
    print("\nExperiments completed successfully!\n")


if __name__ == "__main__":
    main()

