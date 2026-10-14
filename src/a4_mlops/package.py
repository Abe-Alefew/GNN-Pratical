"""
Model packaging and loading (Assignment 4, Stage 4.4).

A model package is a directory containing everything needed to reproduce
inference in a fresh process:
  - ``config.json``: architecture constructor arguments.
  - ``weights.pt``: ``state_dict``.
  - ``feature_schema.json``: expected input shape, dtype, valid ranges.
  - ``labels.json``: class index → class name mapping.
  - ``metadata.json``: run ID, dataset hash, git SHA, version.
  - ``preprocessing.json``: any preprocessing parameters (optional).
"""

import json
from pathlib import Path
from typing import Any

import torch

from src.a2_cora.models import build_model


def save_model_package(
    model_state: dict[str, Any],
    model_config: dict[str, Any],
    feature_schema: dict[str, Any],
    label_mapping: dict[str, str],
    metadata: dict[str, Any],
    output_dir: str,
    preprocessing: dict[str, Any] | None = None,
) -> Path:
    """Save a complete model package to disk.

    Args:
        model_state: Model ``state_dict``.
        model_config: Constructor arguments for ``build_model``.
        feature_schema: Feature count, dtype, valid ranges.
        label_mapping: ``{class_idx_str: class_name}``.
        metadata: Run ID, dataset hash, git SHA, etc.
        output_dir: Directory to write to.
        preprocessing: Optional preprocessing parameters.

    Returns:
        Path to the package directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Weights
    torch.save(model_state, out / "weights.pt")

    # JSON files
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    with open(out / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    with open(out / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if preprocessing is not None:
        with open(out / "preprocessing.json", "w", encoding="utf-8") as f:
            json.dump(preprocessing, f, indent=2)

    return out


def load_model_package(
    package_dir: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a model package from disk.

    Returns a dictionary with:
      - ``"model"``: instantiated model with loaded weights, in eval mode.
      - ``"config"``: model constructor config.
      - ``"feature_schema"``: feature schema dict.
      - ``"label_mapping"``: class index → class name dict.
      - ``"metadata"``: metadata dict.
      - ``"preprocessing"``: preprocessing dict (or None).

    Raises:
        FileNotFoundError: If required files are missing.
        RuntimeError: If weights cannot be loaded into the model.
    """
    pkg = Path(package_dir)

    required_files = ["config.json", "weights.pt", "feature_schema.json",
                       "labels.json", "metadata.json"]
    for fname in required_files:
        if not (pkg / fname).exists():
            raise FileNotFoundError(
                f"Missing required file in model package: {fname}"
            )

    with open(pkg / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(pkg / "feature_schema.json", "r", encoding="utf-8") as f:
        feature_schema = json.load(f)

    with open(pkg / "labels.json", "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    with open(pkg / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    preprocessing = None
    if (pkg / "preprocessing.json").exists():
        with open(pkg / "preprocessing.json", "r", encoding="utf-8") as f:
            preprocessing = json.load(f)

    # Build and load model
    model = build_model(
        name=config["type"],
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        out_channels=config["out_channels"],
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.5),
    )
    state_dict = torch.load(pkg / "weights.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return {
        "model": model,
        "config": config,
        "feature_schema": feature_schema,
        "label_mapping": label_mapping,
        "metadata": metadata,
        "preprocessing": preprocessing,
    }
