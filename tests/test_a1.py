"""
Tests for Assignment 1 — GCN from scratch on synthetic data.

Covers:
  - Manual node check (hand-computed vs layer output)
  - Permutation equivariance
  - Dense vs sparse path consistency
  - Self-loop and normalisation properties
  - Dataset generation validity
"""

import torch

from src.a1_gcn_scratch.data_synth import (
    compute_edge_homophily,
    create_split_masks,
    generate_sbm_dataset,
)
from src.a1_gcn_scratch.layers import (
    GCNLayer,
    add_self_loops,
    add_self_loops_edge_index,
    compute_a_hat,
    compute_a_hat_coeffs_sparse,
    compute_degree_vector,
)
from src.a1_gcn_scratch.model import GCNScratch, MLPBaseline
from src.common.seed import set_seed

# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestDatasetGeneration:
    """Tests for the stochastic block model generator."""

    def test_shapes(self):
        data = generate_sbm_dataset(seed=42, num_nodes=100)
        assert data["x"].shape == (100, 3)
        assert data["y"].shape == (100,)
        assert data["adj"].shape == (100, 100)
        assert data["edge_index"].shape[0] == 2

    def test_labels_binary(self):
        data = generate_sbm_dataset(seed=42, num_nodes=100, num_classes=2)
        assert set(data["y"].numpy().tolist()) == {0, 1}

    def test_adjacency_symmetric(self):
        data = generate_sbm_dataset(seed=42, num_nodes=50)
        adj = data["adj"]
        assert torch.allclose(adj, adj.t()), "Adjacency should be symmetric"

    def test_no_self_loops_in_adj(self):
        data = generate_sbm_dataset(seed=42, num_nodes=50)
        diag = torch.diag(data["adj"])
        assert (diag == 0).all(), "Adjacency should have no self-loops"

    def test_deterministic(self):
        d1 = generate_sbm_dataset(seed=42, num_nodes=50)
        d2 = generate_sbm_dataset(seed=42, num_nodes=50)
        assert torch.equal(d1["x"], d2["x"])
        assert torch.equal(d1["y"], d2["y"])
        assert torch.equal(d1["adj"], d2["adj"])

    def test_homophily_positive(self):
        data = generate_sbm_dataset(seed=42, num_nodes=200, p_in=0.08, p_out=0.01)
        h = compute_edge_homophily(data["edge_index"], data["y"])
        assert 0.0 <= h <= 1.0
        # With p_in >> p_out, homophily should be well above 0.5
        assert h > 0.5, f"Expected high homophily, got {h:.3f}"


class TestSplitMasks:
    """Tests for train/val/test split masks."""

    def test_disjoint_and_cover(self):
        data = generate_sbm_dataset(seed=42, num_nodes=100)
        train, val, test = create_split_masks(100, data["y"], 0.3, 0.2, 0.5)

        # Disjoint
        assert (train & val).sum() == 0
        assert (train & test).sum() == 0
        assert (val & test).sum() == 0

        # Cover (every node in exactly one split)
        assert (train | val | test).all()


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------


class TestGCNLayer:
    """Tests for the from-scratch GCN layer."""

    def test_a_hat_self_loops(self):
        """Every row of Ã should have a nonzero diagonal."""
        adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        a_tilde = add_self_loops(adj)
        assert (torch.diag(a_tilde) > 0).all()

    def test_a_hat_symmetric(self):
        """Â should be symmetric."""
        adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        a_hat = compute_a_hat(adj)
        assert torch.allclose(a_hat, a_hat.t(), atol=1e-6)

    def test_a_hat_max_eigenvalue(self):
        """Largest eigenvalue of Â should be ≤ 1."""
        adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        a_hat = compute_a_hat(adj)
        eigenvalues = torch.linalg.eigvalsh(a_hat)
        assert eigenvalues.max() <= 1.0 + 1e-6

    def test_manual_node_check(self):
        """Pick node 0, compute the GCN layer output by hand, compare."""
        set_seed(42)
        adj = torch.tensor([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=torch.float32)

        x = torch.randn(4, 3)
        a_hat = compute_a_hat(adj)

        layer = GCNLayer(3, 2, use_bias=False)

        # Hand computation for node 0
        # Neighbors of 0 (with self-loop): {0, 1, 2}
        # A_tilde = A + I
        a_tilde = add_self_loops(adj)
        d_tilde = compute_degree_vector(a_tilde)

        # Coefficient for each neighbor j of node i=0:
        # 1 / sqrt(d_tilde[0] * d_tilde[j])
        i = 0
        neighbors = [j for j in range(4) if a_tilde[i, j] > 0]

        # Compute support = x @ W
        support = x @ layer.weight  # (4, 2)

        # Manual aggregation for node 0
        manual_out = torch.zeros(2)
        for j in neighbors:
            coeff = 1.0 / (d_tilde[i].sqrt() * d_tilde[j].sqrt())
            manual_out += coeff * support[j]

        # Layer output
        layer_out = layer.forward_dense(a_hat, x)

        assert torch.allclose(layer_out[0], manual_out, atol=1e-5), (
            f"Manual: {manual_out}, Layer: {layer_out[0]}"
        )

    def test_dense_equals_sparse(self):
        """Dense and sparse (edge-index) paths must produce identical output."""
        set_seed(42)
        adj = torch.tensor([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=torch.float32)

        x = torch.randn(4, 3)
        num_nodes = 4

        # Edge index from adjacency
        src, dst = torch.nonzero(adj, as_tuple=True)
        edge_index = torch.stack([src, dst])

        layer = GCNLayer(3, 2, use_bias=True)

        a_hat = compute_a_hat(adj)
        dense_out = layer.forward_dense(a_hat, x)

        ei_with_loops = add_self_loops_edge_index(edge_index, num_nodes)
        coeffs = compute_a_hat_coeffs_sparse(ei_with_loops, num_nodes)
        sparse_out = layer.forward_sparse(ei_with_loops, coeffs, x, num_nodes)

        assert torch.allclose(dense_out, sparse_out, atol=1e-5), (
            f"Max diff: {(dense_out - sparse_out).abs().max()}"
        )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestPermutationEquivariance:
    """Permutation equivariance test for the GCN model."""

    def test_gcn_equivariant(self):
        """f(PX, PAP^T) ≈ P f(X, A) for random permutation P."""
        set_seed(42)
        data = generate_sbm_dataset(seed=42, num_nodes=20, p_in=0.3, p_out=0.05)
        x, adj = data["x"], data["adj"]

        model = GCNScratch(3, 8, 2, dropout=0.0)
        model.eval()

        # Original output
        with torch.no_grad():
            out_original = model(x, adj)

        # Random permutation
        perm = torch.randperm(20)
        x_perm = x[perm]
        # PAP^T: reindex rows and columns
        adj_perm = adj[perm][:, perm]

        with torch.no_grad():
            out_perm = model(x_perm, adj_perm)

        # f(PX, PAP^T) should equal P f(X, A)
        expected = out_original[perm]

        assert torch.allclose(out_perm, expected, atol=1e-5), (
            f"Max diff: {(out_perm - expected).abs().max():.6f}"
        )

    def test_gcn_equivariant_trained(self):
        """Same test but on a trained model (not just random init)."""
        set_seed(42)
        data = generate_sbm_dataset(seed=42, num_nodes=20, p_in=0.3, p_out=0.05)
        x, adj, labels = data["x"], data["adj"], data["y"]

        model = GCNScratch(3, 8, 2, dropout=0.0)
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            out = model(x, adj)
            loss = torch.nn.functional.nll_loss(out, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out_original = model(x, adj)

        perm = torch.randperm(20)
        x_perm = x[perm]
        adj_perm = adj[perm][:, perm]

        with torch.no_grad():
            out_perm = model(x_perm, adj_perm)

        expected = out_original[perm]
        assert torch.allclose(out_perm, expected, atol=1e-5)


class TestModelShapes:
    """Output shape and interface tests."""

    def test_gcn_output_shape(self):
        model = GCNScratch(3, 16, 2)
        x = torch.randn(10, 3)
        adj = torch.eye(10)
        out = model(x, adj)
        assert out.shape == (10, 2)

    def test_mlp_output_shape(self):
        model = MLPBaseline(3, 16, 2)
        x = torch.randn(10, 3)
        out = model(x)
        assert out.shape == (10, 2)

    def test_log_softmax_output(self):
        """Output should be valid log-probabilities."""
        model = GCNScratch(3, 8, 2)
        x = torch.randn(5, 3)
        adj = torch.eye(5)
        out = model(x, adj)
        # exp(log_softmax) should sum to 1
        probs = out.exp()
        assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)
