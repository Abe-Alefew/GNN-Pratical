"""
Pydantic request/response schemas for the GNN prediction API.
"""


from pydantic import BaseModel, Field


class NodeFeatures(BaseModel):
    """A single node with its ID and feature vector."""
    node_id: int = Field(..., description="Unique node identifier")
    features: list[float] = Field(
        ..., description="Feature vector (must match model's expected dimension)"
    )


class PredictRequest(BaseModel):
    """Request body for the ``/predict`` endpoint.

    The client sends the (sub)graph on which to run inference.

    For a k-layer GCN, the graph must include the k-hop neighbourhood of
    every target node.  Fewer hops silently give different predictions.
    """
    nodes: list[NodeFeatures] = Field(
        ..., description="List of nodes with IDs and features"
    )
    edges: list[list[int]] = Field(
        ..., description="Edge list as [[src, dst], ...] — undirected pairs"
    )
    target_node_ids: list[int] = Field(
        ..., description="IDs of nodes to produce predictions for"
    )


class NodePrediction(BaseModel):
    """Prediction for a single target node."""
    node_id: int
    predicted_class: str
    predicted_class_index: int
    probabilities: dict[str, float] = Field(
        ..., description="Class name → probability"
    )
    warnings: list[str] | None = None


class PredictResponse(BaseModel):
    """Response from the ``/predict`` endpoint."""
    model_config = {"protected_namespaces": ()}
    predictions: list[NodePrediction]
    model_version: str


class HealthResponse(BaseModel):
    """Response from the ``/health`` endpoint."""
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Response from the ``/model-info`` endpoint."""
    model_config = {"protected_namespaces": ()}
    model_type: str
    hidden_channels: int
    num_layers: int
    num_features: int
    num_classes: int
    model_version: str
    dataset_hash: str | None = None
    run_id: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
