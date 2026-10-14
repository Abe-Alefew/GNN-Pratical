"""
Training loop for Assignment 1 (GCN from scratch on synthetic data).

Handles:
  - Masked cross-entropy loss (only training nodes contribute).
  - Validation-based early stopping with best-checkpoint restore.
  - Per-epoch metric logging.
  - Edge perturbation experiments (randomise a fraction of edges).
"""

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from src.common.metrics import accuracy, macro_f1
from src.common.seed import set_seed


def train_one_epoch(
    model: nn.Module,
    x: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: optim.Optimizer,
) -> float:
    """Run one training epoch.

    Args:
        model: GCN or MLP model.
        x: Node features, ``(N, F)``.
        adj: Adjacency matrix, ``(N, N)``.
        labels: Ground-truth labels, ``(N,)``.
        train_mask: Boolean mask for training nodes.
        optimizer: PyTorch optimiser.

    Returns:
        Training loss (scalar).
    """
    model.train()
    optimizer.zero_grad()
    out = model(x, adj)  # (N, C) log-probs
    loss = F.nll_loss(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Evaluate the model on a given split.

    Args:
        model: Trained model.
        x: Node features.
        adj: Adjacency matrix.
        labels: Ground-truth labels.
        mask: Boolean mask for the split.

    Returns:
        Dictionary with ``"loss"``, ``"accuracy"``, ``"macro_f1"``.
    """
    model.eval()
    out = model(x, adj)
    loss = F.nll_loss(out[mask], labels[mask])

    preds = out[mask].argmax(dim=1).cpu().numpy()
    true = labels[mask].cpu().numpy()

    return {
        "loss": float(loss.item()),
        "accuracy": accuracy(true, preds),
        "macro_f1": macro_f1(true, preds),
    }


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    verbose: bool = False,
) -> dict[str, Any]:
    """Full training loop with early stopping.

    Args:
        model: Model instance.
        x, adj, labels: Graph data.
        train_mask, val_mask, test_mask: Boolean masks.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        epochs: Maximum epochs.
        patience: Early-stopping patience (epochs without improvement).
        verbose: Print per-epoch stats.

    Returns:
        Dictionary containing:
          - ``"train_history"``: list of per-epoch train losses.
          - ``"val_history"``: list of per-epoch val metrics.
          - ``"best_val"``: best validation metrics dict.
          - ``"test_metrics"``: test metrics from the best-val checkpoint.
          - ``"best_epoch"``: epoch of the best validation.
          - ``"model_state"``: state_dict of the best checkpoint.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    train_history: list[float] = []
    val_history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, x, adj, labels, train_mask, optimizer)
        val_metrics = evaluate(model, x, adj, labels, val_mask)

        train_history.append(train_loss)
        val_history.append(val_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # Restore best checkpoint and evaluate on test
    assert best_state is not None, "Training produced no checkpoint"
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, x, adj, labels, test_mask)

    return {
        "train_history": train_history,
        "val_history": val_history,
        "best_val": val_history[best_epoch - 1],
        "test_metrics": test_metrics,
        "best_epoch": best_epoch,
        "model_state": best_state,
    }


def randomize_edges(
    adj: torch.Tensor,
    fraction: float,
    seed: int = 0,
) -> torch.Tensor:
    """Randomise a fraction of edges while preserving edge count and symmetry.

    Args:
        adj: Dense adjacency matrix ``(N, N)`` (symmetric, no self-loops).
        fraction: Fraction of edges to rewire, in [0, 1].
        seed: Random seed.

    Returns:
        New adjacency matrix with ``fraction`` of edges randomly rewired.
    """
    if fraction <= 0.0:
        return adj.clone()

    rng = np.random.RandomState(seed)
    n = adj.size(0)
    adj_np = adj.numpy().copy()

    # Extract upper triangle edges
    rows, cols = np.triu_indices(n, k=1)
    edges = adj_np[rows, cols]
    edge_indices = np.where(edges > 0)[0]
    num_edges = len(edge_indices)
    num_to_randomize = int(num_edges * fraction)

    if num_to_randomize == 0:
        return adj.clone()

    # Remove selected edges
    selected = rng.choice(edge_indices, size=num_to_randomize, replace=False)
    for idx in selected:
        r, c = rows[idx], cols[idx]
        adj_np[r, c] = 0.0
        adj_np[c, r] = 0.0

    # Add same number of random new edges (avoiding duplicates and self-loops)
    non_edge_mask = (adj_np == 0)
    np.fill_diagonal(non_edge_mask, False)
    # Only upper triangle
    for i in range(n):
        for j in range(i):
            non_edge_mask[i, j] = False

    available = np.argwhere(non_edge_mask)
    num_to_randomize = min(num_to_randomize, len(available))

    chosen = rng.choice(len(available), size=num_to_randomize, replace=False)
    for idx in chosen:
        r, c = available[idx]
        adj_np[r, c] = 1.0
        adj_np[c, r] = 1.0

    return torch.tensor(adj_np, dtype=torch.float32)


def run_perturbation_experiment(
    data: dict[str, torch.Tensor],
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    fractions: list[float],
    hidden_dim: int = 16,
    dropout: float = 0.5,
    seeds: list[int] = (0, 1, 2),
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
) -> list[dict[str, Any]]:
    """Run the edge-perturbation sweep.

    For each fraction of randomised edges and each seed, trains a GCN and
    an MLP and records test metrics.

    Args:
        data: Dataset dict from ``generate_sbm_dataset``.
        train_mask, val_mask, test_mask: Split masks.
        fractions: List of edge-randomisation fractions.
        hidden_dim: GCN hidden dimension.
        dropout: Dropout rate.
        seeds: List of random seeds.
        lr: Learning rate.
        weight_decay: Weight decay.
        epochs: Max epochs.
        patience: Early-stopping patience.

    Returns:
        List of result dicts with keys: ``fraction``, ``seed``,
        ``model_type``, ``test_accuracy``, ``test_macro_f1``.
    """
    from src.a1_gcn_scratch.model import GCNScratch, MLPBaseline

    results: list[dict[str, Any]] = []
    x = data["x"]
    labels = data["y"]
    in_features = x.size(1)
    num_classes = int(labels.max().item()) + 1

    for frac in fractions:
        for seed in seeds:
            set_seed(seed)

            # Perturbed adjacency
            adj_perturbed = randomize_edges(data["adj"], fraction=frac, seed=seed)

            # --- GCN ---
            gcn = GCNScratch(in_features, hidden_dim, num_classes, dropout)
            gcn_result = train_model(
                gcn, x, adj_perturbed, labels,
                train_mask, val_mask, test_mask,
                lr=lr, weight_decay=weight_decay,
                epochs=epochs, patience=patience,
            )
            results.append({
                "fraction": frac,
                "seed": seed,
                "model_type": "GCN",
                "test_accuracy": gcn_result["test_metrics"]["accuracy"],
                "test_macro_f1": gcn_result["test_metrics"]["macro_f1"],
            })

            # --- MLP (unaffected by graph, but train for each seed) ---
            mlp = MLPBaseline(in_features, hidden_dim, num_classes, dropout)
            mlp_result = train_model(
                mlp, x, adj_perturbed, labels,
                train_mask, val_mask, test_mask,
                lr=lr, weight_decay=weight_decay,
                epochs=epochs, patience=patience,
            )
            results.append({
                "fraction": frac,
                "seed": seed,
                "model_type": "MLP",
                "test_accuracy": mlp_result["test_metrics"]["accuracy"],
                "test_macro_f1": mlp_result["test_metrics"]["macro_f1"],
            })

    return results


def main() -> None:
    """CLI entrypoint for running Assignment 1."""
    from pathlib import Path
    from src.a1_gcn_scratch.data_synth import (
        compute_edge_homophily,
        create_split_masks,
        generate_sbm_dataset,
    )
    from src.a1_gcn_scratch.model import GCNScratch, MLPBaseline
    from src.common.config import load_config

    config_path = Path("configs/a1_config.yaml")
    if config_path.exists():
        cfg = load_config(config_path)
    else:
        cfg = {
            "dataset": {"num_nodes": 200, "num_classes": 2, "num_features": 3, "p_in": 0.08, "p_out": 0.01, "feature_sigma": 1.5},
            "split": {"train_ratio": 0.3, "val_ratio": 0.2, "test_ratio": 0.5},
            "model": {"hidden_dim": 16, "dropout": 0.5},
            "training": {"lr": 0.01, "weight_decay": 5e-4, "epochs": 200, "patience": 20},
            "seeds": [42],
        }

    seed = cfg.get("seeds", [42])[0]
    set_seed(seed)

    print("=" * 65)
    print("  ASSIGNMENT 1 — GCN FROM SCRATCH ON SYNTHETIC SBM GRAPH")
    print("=" * 65)

    ds_cfg = cfg.get("dataset", {})
    data = generate_sbm_dataset(
        seed=seed,
        num_nodes=ds_cfg.get("num_nodes", 200),
        num_classes=ds_cfg.get("num_classes", 2),
        num_features=ds_cfg.get("num_features", 3),
        p_in=ds_cfg.get("p_in", 0.08),
        p_out=ds_cfg.get("p_out", 0.01),
        feature_sigma=ds_cfg.get("feature_sigma", 1.5),
        class_means=ds_cfg.get("class_means"),
    )

    x, y, adj, edge_index = data["x"], data["y"], data["adj"], data["edge_index"]
    h = compute_edge_homophily(edge_index, y)

    sp_cfg = cfg.get("split", {})
    train_mask, val_mask, test_mask = create_split_masks(
        num_nodes=x.size(0),
        labels=y,
        train_ratio=sp_cfg.get("train_ratio", 0.3),
        val_ratio=sp_cfg.get("val_ratio", 0.2),
        test_ratio=sp_cfg.get("test_ratio", 0.5),
        seed=seed,
    )

    print(f"\n[1] Dataset Synthesized:")
    print(f"    - Nodes: {x.size(0)}, Features: {x.size(1)}, Total Directed Edges: {edge_index.size(1)}")
    print(f"    - Edge Homophily Ratio (h): {h:.4f} (High community clustering)")
    print(f"    - Split: Train={train_mask.sum().item()}, Val={val_mask.sum().item()}, Test={test_mask.sum().item()}")

    m_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})
    hidden_dim = m_cfg.get("hidden_dim", 16)
    dropout = m_cfg.get("dropout", 0.5)
    lr = tr_cfg.get("lr", 0.01)
    weight_decay = tr_cfg.get("weight_decay", 5e-4)
    epochs = tr_cfg.get("epochs", 200)
    patience = tr_cfg.get("patience", 20)

    print(f"\n[2] Training 2-Layer GCN (from scratch)...")
    gcn = GCNScratch(x.size(1), hidden_dim, int(y.max().item()) + 1, dropout)
    gcn_res = train_model(
        gcn, x, adj, y, train_mask, val_mask, test_mask,
        lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience, verbose=False,
    )

    print(f"[3] Training Structure-Agnostic MLP Baseline...")
    mlp = MLPBaseline(x.size(1), hidden_dim, int(y.max().item()) + 1, dropout)
    mlp_res = train_model(
        mlp, x, adj, y, train_mask, val_mask, test_mask,
        lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience, verbose=False,
    )

    print("\n" + "-" * 65)
    print("  BENCHMARK COMPARISON ON SYNTHETIC SBM GRAPH")
    print("-" * 65)
    print(f"  Model            | Test Accuracy | Test Macro F1 | Best Val Loss")
    print(f"  -----------------|---------------|---------------|--------------")
    print(f"  GCN (Scratch)    | {gcn_res['test_metrics']['accuracy']:>12.2%} | {gcn_res['test_metrics']['macro_f1']:>13.4f} | {gcn_res['best_val']['loss']:>12.4f}")
    print(f"  MLP (Baseline)   | {mlp_res['test_metrics']['accuracy']:>12.2%} | {mlp_res['test_metrics']['macro_f1']:>13.4f} | {mlp_res['best_val']['loss']:>12.4f}")
    print("-" * 65)
    advantage = gcn_res['test_metrics']['accuracy'] - mlp_res['test_metrics']['accuracy']
    print(f"  Topological Context Advantage: +{advantage:.2%}\n")


if __name__ == "__main__":
    main()

