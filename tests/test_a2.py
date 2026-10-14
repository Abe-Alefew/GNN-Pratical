"""
Tests for Assignment 2 — Cora model architectures.

Covers:
  - Model output shapes for all four architectures.
  - Forward pass correctness (log-probabilities sum to 1).
  - Model factory function.
  - Parameter count is positive and finite.
"""

import torch

from src.a2_cora.models import GAT, GCN, MLP, GraphSAGE, build_model
from src.common.seed import set_seed

# Shared test data
NUM_NODES = 50
IN_CHANNELS = 10
HIDDEN = 16
OUT_CHANNELS = 4
NUM_EDGES = 100


def _make_test_data():
    """Create minimal test graph data."""
    set_seed(0)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    return x, edge_index


class TestModelShapes:
    """All models must produce (N, C) log-probabilities."""

    def test_mlp_shape(self):
        x, ei = _make_test_data()
        model = MLP(IN_CHANNELS, HIDDEN, OUT_CHANNELS)
        out = model(x, ei)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)

    def test_gcn_shape(self):
        x, ei = _make_test_data()
        model = GCN(IN_CHANNELS, HIDDEN, OUT_CHANNELS)
        out = model(x, ei)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)

    def test_sage_shape(self):
        x, ei = _make_test_data()
        model = GraphSAGE(IN_CHANNELS, HIDDEN, OUT_CHANNELS)
        out = model(x, ei)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)

    def test_gat_shape(self):
        x, ei = _make_test_data()
        model = GAT(IN_CHANNELS, HIDDEN, OUT_CHANNELS, heads=4)
        out = model(x, ei)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)


class TestLogProbabilities:
    """Output should be valid log-probabilities."""

    def _check_log_probs(self, model):
        x, ei = _make_test_data()
        model.eval()
        with torch.no_grad():
            out = model(x, ei)
        probs = out.exp()
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(NUM_NODES), atol=1e-4), (
            f"Prob sums: min={sums.min():.4f}, max={sums.max():.4f}"
        )

    def test_mlp(self):
        self._check_log_probs(MLP(IN_CHANNELS, HIDDEN, OUT_CHANNELS))

    def test_gcn(self):
        self._check_log_probs(GCN(IN_CHANNELS, HIDDEN, OUT_CHANNELS))

    def test_sage(self):
        self._check_log_probs(GraphSAGE(IN_CHANNELS, HIDDEN, OUT_CHANNELS))

    def test_gat(self):
        self._check_log_probs(GAT(IN_CHANNELS, HIDDEN, OUT_CHANNELS, heads=4))


class TestModelFactory:
    """Test the build_model factory function."""

    def test_build_all(self):
        for name in ["MLP", "GCN", "GraphSAGE", "GAT"]:
            kwargs = {
                "in_channels": IN_CHANNELS,
                "hidden_channels": HIDDEN,
                "out_channels": OUT_CHANNELS,
            }
            if name == "GAT":
                kwargs["heads"] = 4
            model = build_model(name, **kwargs)
            assert model is not None

    def test_invalid_name(self):
        try:
            build_model("InvalidModel", in_channels=10, hidden_channels=16,
                         out_channels=4)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestParameterCount:
    """Models should have a positive, finite number of trainable parameters."""

    def test_param_counts(self):
        for name in ["MLP", "GCN", "GraphSAGE", "GAT"]:
            kwargs = {
                "in_channels": IN_CHANNELS,
                "hidden_channels": HIDDEN,
                "out_channels": OUT_CHANNELS,
            }
            if name == "GAT":
                kwargs["heads"] = 4
            model = build_model(name, **kwargs)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert count > 0, f"{name} has 0 trainable parameters"
            assert count < 1e8, f"{name} has suspiciously many parameters"


class TestEmbeddings:
    """Models with get_embeddings should return valid embeddings."""

    def test_gcn_embeddings(self):
        x, ei = _make_test_data()
        model = GCN(IN_CHANNELS, HIDDEN, OUT_CHANNELS)
        model.eval()
        with torch.no_grad():
            emb = model.get_embeddings(x, ei)
        assert emb.shape == (NUM_NODES, HIDDEN)

    def test_sage_embeddings(self):
        x, ei = _make_test_data()
        model = GraphSAGE(IN_CHANNELS, HIDDEN, OUT_CHANNELS)
        model.eval()
        with torch.no_grad():
            emb = model.get_embeddings(x, ei)
        assert emb.shape == (NUM_NODES, HIDDEN)

    def test_gat_embeddings(self):
        x, ei = _make_test_data()
        model = GAT(IN_CHANNELS, HIDDEN, OUT_CHANNELS, heads=4)
        model.eval()
        with torch.no_grad():
            emb = model.get_embeddings(x, ei)
        # GAT concatenates heads: hidden * heads
        assert emb.shape[0] == NUM_NODES
        assert emb.shape[1] == HIDDEN * 4
