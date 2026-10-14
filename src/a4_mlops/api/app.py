"""
FastAPI service for GNN node classification (Assignment 4, Stage 4.6).

Endpoints:
  - ``POST /predict``: Accepts node features, edges, and target node IDs.
    Returns predicted classes, probabilities, and model version.
  - ``GET /health``: Service health check.
  - ``GET /model-info``: Model metadata.

The service loads a model package on startup.  If loading fails, the
service refuses to start (fail-fast).
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException

from src.a4_mlops.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    NodePrediction,
    PredictRequest,
    PredictResponse,
)
from src.a4_mlops.package import load_model_package

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GNN Node Classification API",
    description="Serves a trained GNN model for node classification on graphs.",
    version="1.0.0",
)

# Global model state — loaded on startup
_model_state: dict | None = None


@app.on_event("startup")
def startup_load_model() -> None:
    """Load model package on startup.  Fail fast if package is missing."""
    global _model_state

    package_dir = os.environ.get("MODEL_PACKAGE_DIR", "./model_packages/latest")
    pkg_path = Path(package_dir)

    # If package_dir contains a run-id subdirectory, find it
    if pkg_path.exists() and not (pkg_path / "config.json").exists():
        subdirs = [d for d in pkg_path.iterdir() if d.is_dir() and (d / "config.json").exists()]
        if subdirs:
            pkg_path = subdirs[0]

    try:
        _model_state = load_model_package(str(pkg_path))
    except (FileNotFoundError, RuntimeError) as e:
        # Log and mark as unavailable — /health will return 503
        print(f"ERROR: Failed to load model package from {package_dir}: {e}")
        _model_state = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    if _model_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=_model_state["metadata"].get("run_id", "unknown"),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model metadata."""
    if _model_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cfg = _model_state["config"]
    meta = _model_state["metadata"]
    return ModelInfoResponse(
        model_type=cfg["type"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg.get("num_layers", 2),
        num_features=cfg["in_channels"],
        num_classes=cfg["out_channels"],
        model_version=meta.get("run_id", "unknown"),
        dataset_hash=meta.get("dataset_hash"),
        run_id=meta.get("run_id"),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
def predict(request: PredictRequest) -> PredictResponse:
    """Run inference on the supplied subgraph.

    Validates inputs, maps node IDs to contiguous indices, runs the GNN
    in eval/no_grad mode, and returns predictions with probabilities.
    """
    if _model_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = _model_state["model"]
    label_mapping = _model_state["label_mapping"]
    feature_schema = _model_state["feature_schema"]
    metadata = _model_state["metadata"]

    expected_dim = feature_schema["num_features"]

    # --- Validation ---

    # Build ID → index mapping
    node_ids = [n.node_id for n in request.nodes]
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # Check for duplicate node IDs — we deduplicate
    if len(node_ids) != len(set(node_ids)):
        # Deduplicate, keeping first occurrence
        seen = set()
        deduped = []
        for n in request.nodes:
            if n.node_id not in seen:
                deduped.append(n)
                seen.add(n.node_id)
        request.nodes = deduped
        node_ids = [n.node_id for n in request.nodes]
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # Check feature dimensions
    for node in request.nodes:
        if len(node.features) != expected_dim:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Node {node.node_id}: expected {expected_dim} features, "
                    f"got {len(node.features)}"
                ),
            )

    # Check target IDs exist in supplied nodes
    unknown_targets = [
        tid for tid in request.target_node_ids if tid not in id_to_idx
    ]
    if unknown_targets:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown target node IDs: {unknown_targets}",
        )

    # Deduplicate target IDs
    target_ids = list(dict.fromkeys(request.target_node_ids))

    # Validate edges
    edge_pairs = []
    for i, edge in enumerate(request.edges):
        if len(edge) != 2:
            raise HTTPException(
                status_code=422,
                detail=f"Edge at index {i} must have exactly 2 elements, got {len(edge)}",
            )
        src, dst = edge
        if not isinstance(src, int) or not isinstance(dst, int):
            raise HTTPException(
                status_code=422,
                detail=f"Edge at index {i}: endpoints must be integers",
            )
        if src < 0 or dst < 0:
            raise HTTPException(
                status_code=422,
                detail=f"Edge at index {i}: negative endpoint ({src}, {dst})",
            )
        if src not in id_to_idx or dst not in id_to_idx:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Edge at index {i}: endpoint not in supplied nodes. "
                    f"src={src} in_nodes={src in id_to_idx}, "
                    f"dst={dst} in_nodes={dst in id_to_idx}"
                ),
            )
        edge_pairs.append([id_to_idx[src], id_to_idx[dst]])

    # --- Build tensors ---
    x = torch.tensor(
        [n.features for n in request.nodes], dtype=torch.float32
    )

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=1)

    # --- Build response ---
    predictions = []
    for tid in target_ids:
        idx = id_to_idx[tid]
        node_probs = probs[idx].tolist()
        pred_class_idx = int(probs[idx].argmax().item())
        pred_class_name = label_mapping.get(str(pred_class_idx), f"class_{pred_class_idx}")

        prob_dict = {
            label_mapping.get(str(c), f"class_{c}"): round(p, 6)
            for c, p in enumerate(node_probs)
        }

        warnings = []
        # Check for isolated node (no edges involving this node)
        node_degree = (
            (edge_index[0] == idx).sum() + (edge_index[1] == idx).sum()
        ).item()
        if node_degree == 0:
            warnings.append("isolated_node")

        predictions.append(NodePrediction(
            node_id=tid,
            predicted_class=pred_class_name,
            predicted_class_index=pred_class_idx,
            probabilities=prob_dict,
            warnings=warnings if warnings else None,
        ))

    return PredictResponse(
        predictions=predictions,
        model_version=metadata.get("run_id", "unknown"),
    )
