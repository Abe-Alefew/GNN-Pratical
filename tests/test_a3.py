"""
Tests for Assignment 3 — Link prediction and data leakage.

Covers:
  - Edge split correctness and disjointness
  - No held-out edges in message-passing graph
  - Negative validity (no self-loops, no positives)
  - Negative disjointness across splits
  - Decoder symmetry
  - Deterministic splits
"""

import torch

from src.a3_linkpred.checks import (
    check_negatives_disjoint,
    check_negatives_valid,
    check_no_heldout_in_mp,
    check_positive_disjoint,
)
from src.a3_linkpred.decoders import DotProductDecoder, SymmetricMLPDecoder
from src.a3_linkpred.encoders import LinkPredEncoder
from src.a3_linkpred.negatives import build_positive_set, sample_all_negatives
from src.a3_linkpred.split import split_edges, to_undirected_pairs
from src.a3_linkpred.train import (
    build_train_adjacency,
    evaluate_common_neighbor_baseline,
    predict_top_candidates,
    run_leaky_experiment,
)
from src.common.seed import set_seed



def _make_test_graph():
    """Create a small test graph with known structure."""
    # 10 nodes, ~20 undirected edges
    set_seed(42)
    edges = set()
    import random
    random.seed(42)
    while len(edges) < 20:
        u = random.randint(0, 9)
        v = random.randint(0, 9)
        if u != v:
            edges.add((min(u, v), max(u, v)))

    src_list, dst_list = [], []
    for u, v in edges:
        src_list.extend([u, v])
        dst_list.extend([v, u])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index, 10


class TestEdgeSplit:
    """Tests for edge splitting."""

    def test_undirected_pairs_no_duplicates(self):
        ei, _ = _make_test_graph()
        pairs = to_undirected_pairs(ei)
        # Each pair should have u < v
        assert (pairs[:, 0] < pairs[:, 1]).all()
        # No duplicates
        pair_set = set(map(tuple, pairs.tolist()))
        assert len(pair_set) == len(pairs)

    def test_split_covers_all_edges(self):
        ei, _ = _make_test_graph()
        splits = split_edges(ei, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0)
        total = len(splits["train_pos"]) + len(splits["val_pos"]) + len(splits["test_pos"])
        original = len(to_undirected_pairs(ei))
        assert total == original

    def test_positive_disjoint(self):
        ei, _ = _make_test_graph()
        splits = split_edges(ei, seed=0)
        check_positive_disjoint(splits["train_pos"], splits["val_pos"], splits["test_pos"])

    def test_no_heldout_in_mp(self):
        ei, _ = _make_test_graph()
        splits = split_edges(ei, seed=0)
        check_no_heldout_in_mp(
            splits["mp_edge_index"], splits["val_pos"], splits["test_pos"]
        )

    def test_deterministic(self):
        ei, _ = _make_test_graph()
        s1 = split_edges(ei, seed=42)
        s2 = split_edges(ei, seed=42)
        assert torch.equal(s1["train_pos"], s2["train_pos"])
        assert torch.equal(s1["val_pos"], s2["val_pos"])
        assert torch.equal(s1["test_pos"], s2["test_pos"])


class TestNegativeSampling:
    """Tests for negative sampling."""

    def test_negatives_valid(self):
        ei, num_nodes = _make_test_graph()
        splits = split_edges(ei, seed=0)
        pos_set = build_positive_set(
            splits["train_pos"], splits["val_pos"], splits["test_pos"]
        )
        train_neg, val_neg, test_neg = sample_all_negatives(
            num_nodes, splits["train_pos"], splits["val_pos"], splits["test_pos"],
            seed=0,
        )

        check_negatives_valid(val_neg, pos_set, "val")
        check_negatives_valid(test_neg, pos_set, "test")
        check_negatives_valid(train_neg, pos_set, "train")

    def test_negatives_disjoint(self):
        ei, num_nodes = _make_test_graph()
        splits = split_edges(ei, seed=0)
        train_neg, val_neg, test_neg = sample_all_negatives(
            num_nodes, splits["train_pos"], splits["val_pos"], splits["test_pos"],
            seed=0,
        )
        check_negatives_disjoint(val_neg, test_neg, train_neg)

    def test_negatives_count(self):
        ei, num_nodes = _make_test_graph()
        splits = split_edges(ei, seed=0)
        train_neg, val_neg, test_neg = sample_all_negatives(
            num_nodes, splits["train_pos"], splits["val_pos"], splits["test_pos"],
            seed=0,
        )
        # 1:1 ratio
        assert len(val_neg) == len(splits["val_pos"])
        assert len(test_neg) == len(splits["test_pos"])
        assert len(train_neg) == len(splits["train_pos"])


class TestDecoderSymmetry:
    """Decoders must be symmetric: score(i,j) == score(j,i)."""

    def test_dot_product_symmetric(self):
        decoder = DotProductDecoder()
        set_seed(42)
        z_i = torch.randn(10, 16)
        z_j = torch.randn(10, 16)

        s_ij = decoder(z_i, z_j)
        s_ji = decoder(z_j, z_i)

        assert torch.allclose(s_ij, s_ji, atol=1e-6), (
            f"Max diff: {(s_ij - s_ji).abs().max()}"
        )

    def test_mlp_decoder_symmetric(self):
        decoder = SymmetricMLPDecoder(embedding_dim=16, hidden_dim=32)
        decoder.eval()
        set_seed(42)
        z_i = torch.randn(10, 16)
        z_j = torch.randn(10, 16)

        with torch.no_grad():
            s_ij = decoder(z_i, z_j)
            s_ji = decoder(z_j, z_i)

        assert torch.allclose(s_ij, s_ji, atol=1e-6), (
            f"Max diff: {(s_ij - s_ji).abs().max()}"
        )


class TestBaselineAndExperiments:
    """Tests for baseline evaluation and leaky experiment."""

    def test_baseline_evaluation(self):
        ei, num_nodes = _make_test_graph()
        splits = split_edges(ei, seed=0)
        train_pos, val_pos, test_pos = splits["train_pos"], splits["val_pos"], splits["test_pos"]
        _, _, test_neg = sample_all_negatives(num_nodes, train_pos, val_pos, test_pos, seed=0)

        train_adj = build_train_adjacency(num_nodes, train_pos)
        metrics = evaluate_common_neighbor_baseline(train_adj, test_pos, test_neg)
        assert "test_auc" in metrics
        assert "test_ap" in metrics
        assert 0.0 <= metrics["test_auc"] <= 1.0
        assert 0.0 <= metrics["test_ap"] <= 1.0

    def test_leaky_experiment_runs(self):
        ei, num_nodes = _make_test_graph()
        splits = split_edges(ei, seed=0)
        train_pos, val_pos, test_pos = splits["train_pos"], splits["val_pos"], splits["test_pos"]
        train_neg, val_neg, test_neg = sample_all_negatives(num_nodes, train_pos, val_pos, test_pos, seed=0)

        x = torch.randn(num_nodes, 8)
        encoder = LinkPredEncoder(in_channels=8, hidden_channels=16, out_channels=8, encoder_type="GCN")
        decoder = DotProductDecoder()

        res_leaky = run_leaky_experiment(
            encoder, decoder, x,
            train_pos, train_neg, val_pos, val_neg, test_pos, test_neg,
            epochs=5, patience=5, seed=0,
        )
        assert "test_auc" in res_leaky
        assert "test_ap" in res_leaky


class TestCandidateLinks:
    """Tests for candidate link prediction and open-world filtering."""

    def test_candidate_links_exclusion(self):
        ei, num_nodes = _make_test_graph()
        splits = split_edges(ei, seed=0)
        train_pos, val_pos, test_pos = splits["train_pos"], splits["val_pos"], splits["test_pos"]
        pos_set = build_positive_set(train_pos, val_pos, test_pos)
        train_adj = build_train_adjacency(num_nodes, train_pos)

        x = torch.randn(num_nodes, 8)
        encoder = LinkPredEncoder(in_channels=8, hidden_channels=16, out_channels=8, encoder_type="GCN")
        candidates = predict_top_candidates(
            encoder, x, splits["mp_edge_index"], pos_set, train_adj, top_k=5
        )

        assert len(candidates) == 5
        for c in candidates:
            u, v = c["u"], c["v"]
            # No self-loops
            assert u != v
            # Canonical u < v
            assert u < v
            # No observed positive links
            assert (u, v) not in pos_set
            assert 0.0 <= c["prob"] <= 1.0
            assert c["common_neighbors"] >= 0

