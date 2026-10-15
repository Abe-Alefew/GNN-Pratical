"""
MLflow-tracked training script (Assignment 4, Stage 4.3).

Wraps the training loop with full experiment tracking:
  - Parameters: hyperparameters, seed, dataset hash, git SHA, library versions.
  - Metrics: per-epoch loss, final val/test accuracy and macro F1.
  - Artifacts: model package, config YAML, training curve plot.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import torch
from torch_geometric.datasets import Planetoid

from src.a2_cora.models import build_model
from src.a2_cora.train import train_one_run
from src.a4_mlops.package import save_model_package
from src.common.config import load_config
from src.common.data_hash import compute_dataset_hash
from src.common.seed import set_seed


def prepare_inductive_data(
    data: Any,
    holdout_fraction: float,
    holdout_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Remove held-out nodes and incident edges before training.

    The returned graph is compacted so the training tensors contain no rows or
    edges for held-out nodes.  This makes the inductive boundary explicit
    rather than relying only on masks over the original full graph.
    """
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("inductive.holdout_fraction must be in [0.0, 1.0)")

    num_nodes = int(data.num_nodes)
    num_holdout = int(num_nodes * holdout_fraction)
    if holdout_fraction > 0.0:
        num_holdout = max(1, num_holdout)
    if num_holdout >= num_nodes:
        raise ValueError("inductive holdout must leave at least one training node")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(holdout_seed)
    permutation = torch.randperm(num_nodes, generator=generator)
    holdout_node_ids = permutation[:num_holdout]
    heldout_mask = torch.zeros(num_nodes, dtype=torch.bool)
    heldout_mask[holdout_node_ids] = True

    remaining_node_ids = torch.nonzero(~heldout_mask, as_tuple=False).view(-1)
    node_mapping = torch.full((num_nodes,), -1, dtype=torch.long)
    node_mapping[remaining_node_ids] = torch.arange(len(remaining_node_ids))

    src, dst = data.edge_index
    keep_edges = ~heldout_mask[src] & ~heldout_mask[dst]
    filtered_edges = data.edge_index[:, keep_edges]
    training_edge_index = node_mapping[filtered_edges]

    # These checks make the exclusion guarantee executable and auditable.
    assert not heldout_mask[remaining_node_ids].any()
    assert not heldout_mask[filtered_edges].any()
    assert (training_edge_index >= 0).all()

    exclusion_metadata: dict[str, object] = {
        "enabled": num_holdout > 0,
        "holdout_fraction": holdout_fraction,
        "holdout_seed": holdout_seed,
        "holdout_node_ids": holdout_node_ids.tolist(),
        "holdout_count": num_holdout,
        "training_node_count": len(remaining_node_ids),
        "original_edge_count": int(data.edge_index.size(1)),
        "training_edge_count": int(training_edge_index.size(1)),
        "heldout_incident_edge_count": int((~keep_edges).sum().item()),
        "heldout_nodes_in_training_graph": False,
    }
    return (
        data.x[remaining_node_ids],
        training_edge_index,
        data.y[remaining_node_ids],
        data.train_mask[remaining_node_ids],
        data.val_mask[remaining_node_ids],
        data.test_mask[remaining_node_ids],
        exclusion_metadata,
    )


def get_git_sha() -> str:
    """Get the current git commit SHA, or 'unknown' if not in a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_library_versions() -> dict[str, str]:
    """Collect versions of key libraries."""
    import numpy as np
    import sklearn
    import torch_geometric

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "mlflow": mlflow.__version__,
    }


def train_tracked(
    config_path: str = "configs/a4_config.yaml",
    seed: int = 0,
    experiment_name: str | None = None,
    output_dir: str = "./model_packages",
) -> str:
    """Run a fully tracked training experiment.

    When ``inductive.holdout_fraction`` is non-zero, held-out nodes and all
    incident edges are removed before training.  The compact training graph
    is recorded in the run metadata so serving can introduce those nodes
    later with their features and graph context.

    Args:
        config_path: Path to the YAML config.
        seed: Random seed.
        experiment_name: MLflow experiment name (overrides config).
        output_dir: Where to save the model package.

    Returns:
        MLflow run ID.
    """
    cfg = load_config(config_path)
    set_seed(seed)

    # --- Data ---
    dataset = Planetoid(root=cfg.get("dataset", {}).get("root", "./data"), name="Cora")
    data = dataset[0]

    inductive_cfg = cfg.get("inductive", {})
    (
        train_x,
        train_edge_index,
        train_y,
        train_mask,
        val_mask,
        test_mask,
        exclusion_metadata,
    ) = prepare_inductive_data(
        data,
        holdout_fraction=float(inductive_cfg.get("holdout_fraction", 0.0)),
        holdout_seed=int(inductive_cfg.get("holdout_seed", seed)),
    )

    full_dataset_hash = compute_dataset_hash(data.x, data.edge_index, data.y)
    dataset_hash = compute_dataset_hash(train_x, train_edge_index, train_y)
    git_sha = get_git_sha()
    lib_versions = get_library_versions()

    # --- Model ---
    model_cfg = cfg.get("base_model", {})
    model = build_model(
        name=model_cfg.get("type", "GCN"),
        in_channels=dataset.num_features,
        hidden_channels=model_cfg.get("hidden_dim", 64),
        out_channels=dataset.num_classes,
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.5),
    )

    # --- MLflow setup ---
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = experiment_name or cfg.get("mlflow", {}).get(
        "experiment_name", "gnn_node_classification"
    )
    mlflow.set_experiment(exp_name)

    train_cfg = cfg.get("training", {})

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "model_type": model_cfg.get("type", "GCN"),
            "hidden_dim": model_cfg.get("hidden_dim", 64),
            "num_layers": model_cfg.get("num_layers", 2),
            "dropout": model_cfg.get("dropout", 0.5),
            "lr": train_cfg.get("lr", 0.01),
            "weight_decay": train_cfg.get("weight_decay", 5e-4),
            "epochs": train_cfg.get("epochs", 200),
            "patience": train_cfg.get("patience", 30),
            "seed": seed,
            "dataset_hash": dataset_hash,
            "full_dataset_hash": full_dataset_hash,
            "git_sha": git_sha,
            "inductive_holdout_count": exclusion_metadata["holdout_count"],
            "inductive_holdout_seed": exclusion_metadata["holdout_seed"],
            "inductive_training_node_count": exclusion_metadata["training_node_count"],
            "inductive_training_edge_count": exclusion_metadata["training_edge_count"],
        })
        for lib, ver in lib_versions.items():
            mlflow.log_param(f"lib_{lib}", ver)

        # --- Train ---
        result = train_one_run(
            model, train_x, train_edge_index, train_y,
            train_mask, val_mask, test_mask,
            lr=train_cfg.get("lr", 0.01),
            weight_decay=train_cfg.get("weight_decay", 5e-4),
            epochs=train_cfg.get("epochs", 200),
            patience=train_cfg.get("patience", 30),
            seed=seed,
        )

        # Log per-epoch metrics
        for i, train_loss in enumerate(result["train_losses"]):
            mlflow.log_metric("train_loss", train_loss, step=i + 1)
        for i, val_m in enumerate(result["val_metrics_history"]):
            mlflow.log_metric("val_loss", val_m["loss"], step=i + 1)
            mlflow.log_metric("val_accuracy", val_m["accuracy"], step=i + 1)

        # Log final metrics
        mlflow.log_metrics({
            "final_val_accuracy": result["best_val"]["accuracy"],
            "final_val_macro_f1": result["best_val"]["macro_f1"],
            "final_test_accuracy": result["test_metrics"]["accuracy"],
            "final_test_macro_f1": result["test_metrics"]["macro_f1"],
            "best_epoch": result["best_epoch"],
            "wall_time_s": result["wall_time_s"],
            "param_count": result["param_count"],
        })

        # --- Save model package ---
        package_dir = Path(output_dir) / run.info.run_id
        package_dir.mkdir(parents=True, exist_ok=True)

        save_model_package(
            model_state=result["model_state"],
            model_config={
                "type": model_cfg.get("type", "GCN"),
                "in_channels": dataset.num_features,
                "hidden_channels": model_cfg.get("hidden_dim", 64),
                "out_channels": dataset.num_classes,
                "num_layers": model_cfg.get("num_layers", 2),
                "dropout": model_cfg.get("dropout", 0.5),
            },
            feature_schema={
                "num_features": dataset.num_features,
                "dtype": "float32",
            },
            label_mapping={str(i): f"class_{i}" for i in range(dataset.num_classes)},
            metadata={
                "run_id": run.info.run_id,
                "dataset_hash": dataset_hash,
                "full_dataset_hash": full_dataset_hash,
                "git_sha": git_sha,
                "seed": seed,
                "val_macro_f1": result["best_val"]["macro_f1"],
                "inductive": exclusion_metadata,
            },
            output_dir=str(package_dir),
        )

        # Log model package as artifact
        mlflow.log_artifacts(str(package_dir), artifact_path="model_package")

        # --- Training curve plot ---
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["train_losses"], label="Train Loss")
        val_losses = [m["loss"] for m in result["val_metrics_history"]]
        ax.plot(val_losses, label="Val Loss")
        ax.axvline(result["best_epoch"] - 1, color="red", linestyle="--",
                    label=f"Best Epoch ({result['best_epoch']})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Curve")
        ax.legend()
        fig.tight_layout()
        plot_path = package_dir / "training_curve.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))

        # Log config
        mlflow.log_artifact(config_path)

        return run.info.run_id


if __name__ == "__main__":
    run_id = train_tracked()
    print(f"Run ID: {run_id}")
