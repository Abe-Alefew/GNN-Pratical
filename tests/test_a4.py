"""
Tests for Assignment 4 — MLOps pipeline.

Covers:
  - Model packaging and reload (exact prediction match)
  - Validation gate (all 5 behaviours)
  - API endpoints (predict, health, model-info)
  - API failure handling (wrong features, invalid edges, unknown targets, isolated nodes)
  - Rollback
  - Drift detection
"""

import os
import tempfile

import torch

from src.a2_cora.models import build_model
from src.a4_mlops.gate import evaluate_gate
from src.a4_mlops.package import load_model_package, save_model_package
from src.a4_mlops.rollback import (
    REGISTRY_FILE,
    get_champion,
    promote_version,
    register_version,
    rollback,
)
from src.common.seed import set_seed

# ---------------------------------------------------------------------------
# Model packaging tests
# ---------------------------------------------------------------------------


class TestModelPackaging:
    """Save and reload a model package; predictions must match exactly."""

    def test_reload_exact_match(self):
        set_seed(42)
        model = build_model(
            "GCN", in_channels=10, hidden_channels=16, out_channels=4,
        )
        model.eval()

        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        with torch.no_grad():
            original_out = model(x, edge_index)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_package(
                model_state=model.state_dict(),
                model_config={
                    "type": "GCN",
                    "in_channels": 10,
                    "hidden_channels": 16,
                    "out_channels": 4,
                    "num_layers": 2,
                    "dropout": 0.5,
                },
                feature_schema={"num_features": 10, "dtype": "float32"},
                label_mapping={"0": "A", "1": "B", "2": "C", "3": "D"},
                metadata={"run_id": "test-run", "dataset_hash": "abc123"},
                output_dir=tmpdir,
            )

            loaded = load_model_package(tmpdir)
            loaded_model = loaded["model"]

            with torch.no_grad():
                loaded_out = loaded_model(x, edge_index)

        assert torch.allclose(original_out, loaded_out, atol=1e-6), (
            f"Max diff: {(original_out - loaded_out).abs().max()}"
        )

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                load_model_package(tmpdir)
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Validation gate tests
# ---------------------------------------------------------------------------


class TestValidationGate:
    """Test all 5 gate behaviours."""

    def test_improvement_promotes(self):
        decision = evaluate_gate(
            candidate_metrics={"val_macro_f1": 0.85},
            incumbent_metrics={"val_macro_f1": 0.80},
            threshold_delta=0.0,
        )
        assert decision.promoted is True

    def test_regression_blocks(self):
        decision = evaluate_gate(
            candidate_metrics={"val_macro_f1": 0.75},
            incumbent_metrics={"val_macro_f1": 0.80},
            threshold_delta=0.0,
        )
        assert decision.promoted is False

    def test_tie_keeps_incumbent(self):
        decision = evaluate_gate(
            candidate_metrics={"val_macro_f1": 0.80},
            incumbent_metrics={"val_macro_f1": 0.80},
            threshold_delta=0.0,
            tie_behavior="keep",
        )
        assert decision.promoted is False

    def test_missing_metric_blocks(self):
        decision = evaluate_gate(
            candidate_metrics={"some_other_metric": 0.9},
            incumbent_metrics={"val_macro_f1": 0.80},
        )
        assert decision.promoted is False

    def test_nan_metric_blocks(self):
        decision = evaluate_gate(
            candidate_metrics={"val_macro_f1": float("nan")},
            incumbent_metrics={"val_macro_f1": 0.80},
        )
        assert decision.promoted is False

    def test_first_model_promotes(self):
        decision = evaluate_gate(
            candidate_metrics={"val_macro_f1": 0.50},
            incumbent_metrics=None,
            min_absolute_floor=0.3,
        )
        assert decision.promoted is True

    def test_first_model_below_floor_blocks(self):
        decision = evaluate_gate(
            candidate_metrics={"val_macro_f1": 0.1},
            incumbent_metrics=None,
            min_absolute_floor=0.3,
        )
        assert decision.promoted is False

    def test_threshold_delta(self):
        # Candidate is better but not by enough
        decision = evaluate_gate(
            candidate_metrics={"val_macro_f1": 0.81},
            incumbent_metrics={"val_macro_f1": 0.80},
            threshold_delta=0.05,
        )
        assert decision.promoted is False


# ---------------------------------------------------------------------------
# API tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------


class TestAPI:
    """Test API endpoints with FastAPI's TestClient."""

    @staticmethod
    def _setup_api(tmpdir: str):
        """Create a model package and set env var for the API."""
        set_seed(42)
        model = build_model(
            "GCN", in_channels=3, hidden_channels=8, out_channels=2,
        )
        save_model_package(
            model_state=model.state_dict(),
            model_config={
                "type": "GCN",
                "in_channels": 3,
                "hidden_channels": 8,
                "out_channels": 2,
                "num_layers": 2,
                "dropout": 0.5,
            },
            feature_schema={"num_features": 3, "dtype": "float32"},
            label_mapping={"0": "class_0", "1": "class_1"},
            metadata={"run_id": "test-v1", "dataset_hash": "test-hash"},
            output_dir=tmpdir,
        )
        os.environ["MODEL_PACKAGE_DIR"] = tmpdir

    def test_health(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_api(tmpdir)
            # Re-import to trigger startup
            from src.a4_mlops.api.app import app, startup_load_model
            startup_load_model()

            from fastapi.testclient import TestClient
            client = TestClient(app)

            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"

    def test_predict_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_api(tmpdir)
            from src.a4_mlops.api.app import app, startup_load_model
            startup_load_model()

            from fastapi.testclient import TestClient
            client = TestClient(app)

            resp = client.post("/predict", json={
                "nodes": [
                    {"node_id": 0, "features": [1.0, 2.0, 3.0]},
                    {"node_id": 1, "features": [4.0, 5.0, 6.0]},
                ],
                "edges": [[0, 1], [1, 0]],
                "target_node_ids": [0, 1],
            })
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["predictions"]) == 2

    def test_wrong_feature_dim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_api(tmpdir)
            from src.a4_mlops.api.app import app, startup_load_model
            startup_load_model()

            from fastapi.testclient import TestClient
            client = TestClient(app)

            resp = client.post("/predict", json={
                "nodes": [
                    {"node_id": 0, "features": [1.0, 2.0]},  # wrong dim
                ],
                "edges": [],
                "target_node_ids": [0],
            })
            assert resp.status_code == 422

    def test_unknown_target_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_api(tmpdir)
            from src.a4_mlops.api.app import app, startup_load_model
            startup_load_model()

            from fastapi.testclient import TestClient
            client = TestClient(app)

            resp = client.post("/predict", json={
                "nodes": [
                    {"node_id": 0, "features": [1.0, 2.0, 3.0]},
                ],
                "edges": [],
                "target_node_ids": [99],  # not in nodes
            })
            assert resp.status_code == 422

    def test_invalid_edge_endpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_api(tmpdir)
            from src.a4_mlops.api.app import app, startup_load_model
            startup_load_model()

            from fastapi.testclient import TestClient
            client = TestClient(app)

            resp = client.post("/predict", json={
                "nodes": [
                    {"node_id": 0, "features": [1.0, 2.0, 3.0]},
                ],
                "edges": [[0, 99]],  # 99 not in nodes
                "target_node_ids": [0],
            })
            assert resp.status_code == 422

    def test_isolated_node_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_api(tmpdir)
            from src.a4_mlops.api.app import app, startup_load_model
            startup_load_model()

            from fastapi.testclient import TestClient
            client = TestClient(app)

            resp = client.post("/predict", json={
                "nodes": [
                    {"node_id": 0, "features": [1.0, 2.0, 3.0]},
                ],
                "edges": [],  # no edges = isolated
                "target_node_ids": [0],
            })
            assert resp.status_code == 200
            pred = resp.json()["predictions"][0]
            assert pred.get("warnings") is not None
            assert "isolated_node" in pred["warnings"]


# ---------------------------------------------------------------------------
# Rollback tests
# ---------------------------------------------------------------------------


class TestRollback:
    """Test version registration, promotion, and rollback."""

    def test_register_and_promote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, REGISTRY_FILE)

            register_version("v1", "/path/v1", {"val_macro_f1": 0.8}, reg_path)
            promote_version("v1", "First model", reg_path)

            champion = get_champion(reg_path)
            assert champion is not None
            assert champion["version_id"] == "v1"

    def test_rollback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, REGISTRY_FILE)

            register_version("v1", "/path/v1", {"val_macro_f1": 0.8}, reg_path)
            promote_version("v1", "First model", reg_path)

            register_version("v2", "/path/v2", {"val_macro_f1": 0.85}, reg_path)
            promote_version("v2", "Better model", reg_path)

            # Current champion is v2
            assert get_champion(reg_path)["version_id"] == "v2"

            # Roll back to v1
            result = rollback("Regression detected", reg_path)
            assert result == "v1"
            assert get_champion(reg_path)["version_id"] == "v1"

    def test_rollback_no_previous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, REGISTRY_FILE)

            register_version("v1", "/path/v1", {"val_macro_f1": 0.8}, reg_path)
            promote_version("v1", "First model", reg_path)

            # Only one version, can't roll back
            result = rollback("Test", reg_path)
            assert result is None
