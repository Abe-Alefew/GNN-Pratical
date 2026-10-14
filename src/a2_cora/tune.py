"""
Hyperparameter tuning for Assignment 2.

Uses random search with an equal trial budget for each model.  Model
selection is done on **validation** accuracy (or macro F1 — configurable).
Test metrics are computed only for the chosen configuration.
"""

import csv
import random
from pathlib import Path
from typing import Any

import torch

from src.a2_cora.models import build_model
from src.a2_cora.train import train_one_run
from src.common.seed import set_seed


def sample_hyperparams(
    model_name: str,
    search_space: dict[str, list],
) -> dict[str, Any]:
    """Sample one set of hyperparameters from a search space.

    Args:
        model_name: Model name (``"MLP"``, ``"GCN"``, etc.).
        search_space: Dict of param name → list of candidate values.

    Returns:
        Sampled config dict.
    """
    config = {"model": model_name}
    for key, values in search_space.items():
        config[key] = random.choice(values)
    return config


def run_tuning(
    model_name: str,
    search_space: dict[str, list],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    in_channels: int,
    out_channels: int,
    num_trials: int = 20,
    epochs: int = 300,
    patience: int = 30,
    selection_metric: str = "accuracy",
    tuning_seed: int = 0,
    save_csv: str | None = None,
) -> dict[str, Any]:
    """Run random-search hyperparameter tuning for a single model type.

    Args:
        model_name: One of ``"MLP"``, ``"GCN"``, ``"GraphSAGE"``, ``"GAT"``.
        search_space: Search space per hyperparameter.
        x, edge_index, labels: Graph data.
        train_mask, val_mask, test_mask: Boolean masks.
        in_channels: Input feature dimension.
        out_channels: Number of classes.
        num_trials: Number of random trials (equal budget per model).
        epochs: Max training epochs per trial.
        patience: Early-stopping patience.
        selection_metric: ``"accuracy"`` or ``"macro_f1"`` on validation.
        tuning_seed: Seed for the search itself (not for training).
        save_csv: If given, path to save the trial table.

    Returns:
        Dictionary with ``"best_config"``, ``"best_val_score"``,
        ``"all_trials"`` (list of dicts).
    """
    set_seed(tuning_seed)

    all_trials: list[dict[str, Any]] = []
    best_score = -1.0
    best_config = None

    for trial_idx in range(num_trials):
        config = sample_hyperparams(model_name, search_space)

        # Build model kwargs
        model_kwargs = {
            "in_channels": in_channels,
            "hidden_channels": config.get("hidden_dim", 64),
            "out_channels": out_channels,
            "num_layers": config.get("num_layers", 2),
            "dropout": config.get("dropout", 0.5),
        }
        if model_name == "GAT":
            model_kwargs["heads"] = config.get("heads", 8)
        if model_name == "GraphSAGE":
            model_kwargs["aggr"] = config.get("aggr", "mean")

        model = build_model(model_name, **model_kwargs)

        # Train with a fixed seed for this trial
        result = train_one_run(
            model, x, edge_index, labels,
            train_mask, val_mask, test_mask,
            lr=config.get("lr", 0.01),
            weight_decay=config.get("weight_decay", 5e-4),
            epochs=epochs,
            patience=patience,
            seed=trial_idx,
        )

        val_score = result["best_val"][selection_metric]

        trial_record = {
            "trial": trial_idx,
            **config,
            f"val_{selection_metric}": val_score,
            "val_loss": result["best_val"]["loss"],
            "best_epoch": result["best_epoch"],
        }
        all_trials.append(trial_record)

        if val_score > best_score:
            best_score = val_score
            best_config = config.copy()

    # Save trial table
    if save_csv and all_trials:
        path = Path(save_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_trials[0].keys())
            writer.writeheader()
            writer.writerows(all_trials)

    return {
        "best_config": best_config,
        "best_val_score": best_score,
        "all_trials": all_trials,
    }
