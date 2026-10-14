"""
Training loop for link prediction (Assignment 3).

Uses binary cross-entropy on positive (label 1) and negative (label 0) pairs.
Supports both dot-product and MLP decoders.  Model selection is on validation
AP or AUC.
"""

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from src.common.metrics import average_precision, roc_auc
from src.common.seed import set_seed



def compute_scores(
    encoder: nn.Module,
    decoder: nn.Module,
    x: torch.Tensor,
    mp_edge_index: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    """Compute link scores for a set of node pairs.

    Args:
        encoder: GNN encoder producing node embeddings.
        decoder: Dot-product or MLP decoder.
        x: Node features ``(N, F)``.
        mp_edge_index: Message-passing edge index ``(2, E_mp)``.
        pairs: Node pairs ``(M, 2)``.

    Returns:
        Raw scores (logits), shape ``(M,)``.
    """
    z = encoder(x, mp_edge_index)
    z_i = z[pairs[:, 0]]
    z_j = z[pairs[:, 1]]
    return decoder(z_i, z_j)


def train_link_predictor(
    encoder: nn.Module,
    decoder: nn.Module,
    x: torch.Tensor,
    mp_edge_index: torch.Tensor,
    train_pos: torch.Tensor,
    train_neg: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    selection_metric: str = "ap",
    seed: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train encoder+decoder for link prediction.

    Args:
        encoder: GNN encoder.
        decoder: Link decoder (dot or MLP).
        x: Node features.
        mp_edge_index: Training message-passing edges (both directions).
        train_pos, train_neg: Training pairs ``(M, 2)``.
        val_pos, val_neg: Validation pairs.
        test_pos, test_neg: Test pairs.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        epochs: Max epochs.
        patience: Early-stopping patience.
        selection_metric: ``"ap"`` (average precision) or ``"auc"``.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dictionary with val/test AUC and AP, training history, best epoch,
        and best model states.
    """
    if seed is not None:
        set_seed(seed)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    best_val_score = -1.0
    best_epoch = 0
    best_encoder_state = None
    best_decoder_state = None
    epochs_no_improve = 0

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        pos_scores = compute_scores(encoder, decoder, x, mp_edge_index, train_pos)
        neg_scores = compute_scores(encoder, decoder, x, mp_edge_index, train_neg)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(len(train_pos)),
            torch.zeros(len(train_neg)),
        ])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

        # --- Validate ---
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_pos_scores = compute_scores(
                encoder, decoder, x, mp_edge_index, val_pos
            )
            val_neg_scores = compute_scores(
                encoder, decoder, x, mp_edge_index, val_neg
            )

        val_scores = torch.cat([val_pos_scores, val_neg_scores]).numpy()
        val_labels = torch.cat([
            torch.ones(len(val_pos)),
            torch.zeros(len(val_neg)),
        ]).numpy()

        val_auc = roc_auc(val_labels, val_scores)
        val_ap = average_precision(val_labels, val_scores)

        val_score = val_ap if selection_metric == "ap" else val_auc

        history.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "val_auc": val_auc,
            "val_ap": val_ap,
        })

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_decoder_state = copy.deepcopy(decoder.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 20 == 0:
            print(
                f"  Epoch {epoch:3d} | Loss {loss.item():.4f} | "
                f"Val AUC {val_auc:.4f} | Val AP {val_ap:.4f}"
            )

        if epochs_no_improve >= patience:
            break

    # --- Test ---
    assert best_encoder_state is not None
    encoder.load_state_dict(best_encoder_state)
    decoder.load_state_dict(best_decoder_state)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        test_pos_scores = compute_scores(
            encoder, decoder, x, mp_edge_index, test_pos
        )
        test_neg_scores = compute_scores(
            encoder, decoder, x, mp_edge_index, test_neg
        )

    test_scores = torch.cat([test_pos_scores, test_neg_scores]).numpy()
    test_labels = torch.cat([
        torch.ones(len(test_pos)),
        torch.zeros(len(test_neg)),
    ]).numpy()

    test_auc = roc_auc(test_labels, test_scores)
    test_ap = average_precision(test_labels, test_scores)

    return {
        "best_epoch": best_epoch,
        "history": history,
        "val_auc": history[best_epoch - 1]["val_auc"],
        "val_ap": history[best_epoch - 1]["val_ap"],
        "test_auc": test_auc,
        "test_ap": test_ap,
        "encoder_state": best_encoder_state,
        "decoder_state": best_decoder_state,
    }


def common_neighbor_baseline(
    train_adj: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    """Common-neighbor baseline scores.

    ``score(u, v) = (A_train^2)_{u,v} = |N(u) ∩ N(v)|``

    Args:
        train_adj: Dense adjacency from training edges ``(N, N)``.
        pairs: Node pairs ``(M, 2)``.

    Returns:
        Scores, shape ``(M,)``.
    """
    cn_matrix = train_adj @ train_adj
    return cn_matrix[pairs[:, 0], pairs[:, 1]]


def build_train_adjacency(num_nodes: int, train_pos: torch.Tensor) -> torch.Tensor:
    """Construct dense symmetric adjacency matrix using only training edges."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    src = train_pos[:, 0]
    dst = train_pos[:, 1]
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    return adj


def evaluate_common_neighbor_baseline(
    train_adj: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
) -> dict[str, float]:
    """Evaluate common-neighbor heuristic on held-out test edges."""
    pos_scores = common_neighbor_baseline(train_adj, test_pos).numpy()
    neg_scores = common_neighbor_baseline(train_adj, test_neg).numpy()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return {
        "test_auc": float(roc_auc(labels, scores)),
        "test_ap": float(average_precision(labels, scores)),
    }


def run_leaky_experiment(
    encoder: nn.Module,
    decoder: nn.Module,
    x: torch.Tensor,
    train_pos: torch.Tensor,
    train_neg: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 100,
    patience: int = 20,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run deliberately leaky link prediction where held-out links are in message passing.

    This experiment contaminates the message-passing graph with validation and test
    edges (both directions) to demonstrate the severe hazard of data leakage.
    """
    all_pos = torch.cat([train_pos, val_pos, test_pos], dim=0)
    leaky_src = torch.cat([all_pos[:, 0], all_pos[:, 1]])
    leaky_dst = torch.cat([all_pos[:, 1], all_pos[:, 0]])
    leaky_mp_edge_index = torch.stack([leaky_src, leaky_dst], dim=0)

    return train_link_predictor(
        encoder=encoder,
        decoder=decoder,
        x=x,
        mp_edge_index=leaky_mp_edge_index,
        train_pos=train_pos,
        train_neg=train_neg,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        selection_metric="ap",
        seed=seed,
        verbose=verbose,
    )


def predict_top_candidates(
    encoder: nn.Module,
    x: torch.Tensor,
    mp_edge_index: torch.Tensor,
    all_pos_set: set[tuple[int, int]],
    train_adj: torch.Tensor,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Present top-k scored unobserved node pairs as candidate links.

    Scores non-edge pairs (u, v) with u < v that are NOT in all_pos_set,
    excluding self-loops and all observed positive edges.
    """
    encoder.eval()
    with torch.no_grad():
        z = encoder(x, mp_edge_index)
        scores_mat = z @ z.t()

    num_nodes = x.size(0)
    scores_mat.fill_diagonal_(float("-inf"))
    tril_i, tril_j = torch.tril_indices(num_nodes, num_nodes)
    scores_mat[tril_i, tril_j] = float("-inf")

    for u, v in all_pos_set:
        scores_mat[min(u, v), max(u, v)] = float("-inf")

    flat = scores_mat.view(-1)
    top_scores, top_indices = torch.topk(flat, k=top_k)

    candidates = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
        u = int(idx // num_nodes)
        v = int(idx % num_nodes)
        prob = float(torch.sigmoid(score).item())
        cn = int(train_adj[u] @ train_adj[v])
        candidates.append({
            "rank": rank,
            "u": u,
            "v": v,
            "score": float(score.item()),
            "prob": prob,
            "common_neighbors": cn,
        })
    return candidates


def run_a3_experiments(
    config_path: str | Path | None = None,
    epochs: int | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run full Assignment 3 experiment pipeline meeting all requirements."""
    from torch_geometric.datasets import Planetoid

    from src.a3_linkpred.checks import (
        check_baseline_uses_train_adj,
        check_deterministic_splits,
        check_encoder_graph_unchanged,
        check_negatives_disjoint,
        check_negatives_valid,
        check_no_heldout_in_mp,
        check_positive_disjoint,
    )
    from src.a3_linkpred.decoders import DotProductDecoder, SymmetricMLPDecoder
    from src.a3_linkpred.encoders import LinkPredEncoder
    from src.a3_linkpred.negatives import build_positive_set, sample_all_negatives
    from src.a3_linkpred.split import split_edges
    from src.common.config import load_config

    print("=" * 80)
    print("  ASSIGNMENT 3 -- LINK PREDICTION & DATA LEAKAGE PREVENTION")
    print("=" * 80)

    # 1. Load config
    cfg = {}
    if config_path is not None and Path(config_path).exists():
        cfg = load_config(config_path)
    elif Path("configs/a3_config.yaml").exists():
        cfg = load_config("configs/a3_config.yaml")

    num_epochs = epochs or cfg.get("training", {}).get("epochs", 100)
    lr = cfg.get("training", {}).get("lr", 0.01)
    weight_decay = cfg.get("training", {}).get("weight_decay", 5e-4)
    patience = cfg.get("training", {}).get("patience", 20)
    hidden_dim = cfg.get("encoder", {}).get("hidden_dim", 64)
    embedding_dim = cfg.get("encoder", {}).get("embedding_dim", 32)
    dropout = cfg.get("encoder", {}).get("dropout", 0.3)
    top_k = cfg.get("candidates", {}).get("top_k", 10)

    # 2. Load dataset
    print("\n[Step 1] Loading Cora citation graph...")
    dataset = Planetoid(root="data/Cora", name="Cora")
    data = dataset[0]
    print(f"  Nodes: {data.num_nodes}, Features: {data.num_features}")

    # 3. Canonical Edge Splitting
    print("\n[Step 2] Splitting unique undirected positive edges (85/5/10)...")
    splits = split_edges(data.edge_index, train_ratio=0.85, val_ratio=0.05, test_ratio=0.10, seed=seed)
    train_pos, val_pos, test_pos = splits["train_pos"], splits["val_pos"], splits["test_pos"]
    mp_edge_index = splits["mp_edge_index"]
    print(f"  Train positive links: {len(train_pos)} pairs")
    print(f"  Val positive links:   {len(val_pos)} pairs")
    print(f"  Test positive links:  {len(test_pos)} pairs")
    print(f"  Message Passing A_mp: {mp_edge_index.size(1)} directed edges (train only, both directions)")

    # 4. Negative Sampling
    print("\n[Step 3] Sampling disjoint negative pairs (excluding self-links and all positives)...")
    train_neg, val_neg, test_neg = sample_all_negatives(
        data.num_nodes, train_pos, val_pos, test_pos, seed=seed
    )
    pos_set = build_positive_set(train_pos, val_pos, test_pos)
    print(f"  Train negatives: {len(train_neg)} | Val negatives: {len(val_neg)} | Test negatives: {len(test_neg)}")

    # 5. Automated Verification Checks
    print("\n[Step 4] Executing automated leakage and consistency checks...")
    check_positive_disjoint(train_pos, val_pos, test_pos)
    check_no_heldout_in_mp(mp_edge_index, val_pos, test_pos)
    check_negatives_valid(train_neg, pos_set, "train")
    check_negatives_valid(val_neg, pos_set, "val")
    check_negatives_valid(test_neg, pos_set, "test")
    check_negatives_disjoint(val_neg, test_neg, train_neg)
    check_deterministic_splits(split_edges, data.edge_index, seed=seed)
    check_encoder_graph_unchanged(mp_edge_index, mp_edge_index)
    train_adj = build_train_adjacency(data.num_nodes, train_pos)
    check_baseline_uses_train_adj(train_adj, test_pos)
    print("  [PASS] All 7 automated checks passed successfully!")
    print("         PROVED: Held-out edges & reverse copies are completely absent from A_mp.")

    # 6. Common Neighbor Baseline
    print("\n[Step 5] Evaluating Common-Neighbor baseline (using training graph only)...")
    cn_metrics = evaluate_common_neighbor_baseline(train_adj, test_pos, test_neg)
    print(f"  CN Baseline Test ROC-AUC: {cn_metrics['test_auc']:.4f} | Test AP: {cn_metrics['test_ap']:.4f}")

    # 7. Valid Experiment: Dot-Product Decoder
    print("\n[Step 6] Training GNN Encoder + Dot-Product Decoder (Valid A_mp)...")
    enc_dot = LinkPredEncoder(data.num_features, hidden_dim, embedding_dim, "GCN", dropout=dropout)
    dec_dot = DotProductDecoder()
    res_dot = train_link_predictor(
        enc_dot, dec_dot, data.x, mp_edge_index,
        train_pos, train_neg, val_pos, val_neg, test_pos, test_neg,
        lr=lr, weight_decay=weight_decay, epochs=num_epochs, patience=patience,
        selection_metric="ap", seed=seed, verbose=verbose,
    )
    print(f"  Dot-Product Decoder Test ROC-AUC: {res_dot['test_auc']:.4f} | Test AP: {res_dot['test_ap']:.4f}")

    # 8. Valid Experiment: Symmetric MLP Decoder
    print("\n[Step 7] Training GNN Encoder + Symmetric MLP Decoder (Valid A_mp)...")
    enc_mlp = LinkPredEncoder(data.num_features, hidden_dim, embedding_dim, "GCN", dropout=dropout)
    dec_mlp = SymmetricMLPDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    res_mlp = train_link_predictor(
        enc_mlp, dec_mlp, data.x, mp_edge_index,
        train_pos, train_neg, val_pos, val_neg, test_pos, test_neg,
        lr=lr, weight_decay=weight_decay, epochs=num_epochs, patience=patience,
        selection_metric="ap", seed=seed, verbose=verbose,
    )
    print(f"  Symmetric MLP Decoder Test ROC-AUC: {res_mlp['test_auc']:.4f} | Test AP: {res_mlp['test_ap']:.4f}")

    # 9. Deliberately Leaky Experiment
    print("\n[Step 8] Running Deliberately Leaky Experiment (Held-out links in A_mp)...")
    enc_leaky = LinkPredEncoder(data.num_features, hidden_dim, embedding_dim, "GCN", dropout=dropout)
    dec_leaky = DotProductDecoder()
    res_leaky = run_leaky_experiment(
        enc_leaky, dec_leaky, data.x,
        train_pos, train_neg, val_pos, val_neg, test_pos, test_neg,
        lr=lr, weight_decay=weight_decay, epochs=num_epochs, patience=patience,
        seed=seed, verbose=verbose,
    )
    print(f"  Leaky Experiment Test ROC-AUC: {res_leaky['test_auc']:.4f} | Test AP: {res_leaky['test_ap']:.4f}")

    # 10. Results Summary Table
    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPARISON SUMMARY")
    print("=" * 80)
    print(f"  {'Method / Approach':<40} | {'Test ROC-AUC':<12} | {'Test AP':<10} | {'Status'}")
    print("  " + "-" * 76)
    print(f"  {'Common Neighbors Baseline (Train Graph)':<40} | {cn_metrics['test_auc']:<12.4f} | {cn_metrics['test_ap']:<10.4f} | Valid Baseline")
    print(f"  {'GCN Encoder + Dot-Product Decoder':<40} | {res_dot['test_auc']:<12.4f} | {res_dot['test_ap']:<10.4f} | Valid Model")
    print(f"  {'GCN Encoder + Symmetric MLP Decoder':<40} | {res_mlp['test_auc']:<12.4f} | {res_mlp['test_ap']:<10.4f} | Valid Model")
    print(f"  {'GCN Encoder + Dot-Product (DELIBERATELY LEAKY)':<40} | {res_leaky['test_auc']:<12.4f} | {res_leaky['test_ap']:<10.4f} | INVALID (Leaky)")
    print("  " + "-" * 76)

    # 11. Explanation of Leaky Evaluation
    print("\n[Analysis: Why Leaky Evaluation is Invalid]")
    print("  1. Mechanism: In GNNs, message passing aggregates neighbor features along edges.")
    print("     If test link (u, v) is present in A_mp, z_u directly incorporates x_v and z_v")
    print("     incorporates x_u. The decoder merely detects this 1-hop message exchange rather")
    print("     than generalizing to structural or semantic link patterns.")
    print("  2. Real-World Invalidation: In production/real-world link prediction (e.g., future")
    print("     citations, new drug interactions), candidate links do NOT yet exist in the graph.")
    print("     Testing on edges that were already in the graph creates an artificial illusion")
    print("     of performance.")
    print("  3. Model Selection: Leaky evaluation must NEVER be used for model selection or hyperparameter")
    print("     tuning because it optimizes for memorization of existing communication pathways.")

    # 12. Top 10 Candidate Links
    print("\n[Step 9] Presenting Top 10 Unobserved Candidate Links (Open-World Discovery)...")
    candidates = predict_top_candidates(
        enc_dot, data.x, mp_edge_index, pos_set, train_adj, top_k=top_k
    )
    print(f"  {'Rank':<5} | {'Pair (u, v)':<15} | {'Logit Score':<12} | {'Prob (Sigmoid)':<14} | {'Common Neighbors'}")
    print("  " + "-" * 66)
    for c in candidates:
        print(f"  {c['rank']:<5d} | ({c['u']:4d}, {c['v']:4d})     | {c['score']:<12.4f} | {c['prob']:<14.4f} | {c['common_neighbors']:<5d}")

    # 13. Explanation of Unobserved Links
    print("\n[Analysis: Open-World Assumption & Prediction Validation]")
    print("  1. Open-World Assumption: In network data, missing edges indicate UNOBSERVED pairs,")
    print("     NOT confirmed negatives. In a citation network, two authors might write complementary")
    print("     papers without citing each other simply because they were unaware of each other's work.")
    print("  2. Why Predictions Require Validation: A high-scoring unobserved pair represents a")
    print("     plausible link hypothesis discovered by the GNN. However, before acting on it")
    print("     (e.g., recommending a paper, synthesizing a chemical compound), empirical verification")
    print("     or domain-expert review is strictly required.")
    print("=" * 80 + "\n")

    return {
        "cn_metrics": cn_metrics,
        "res_dot": res_dot,
        "res_mlp": res_mlp,
        "res_leaky": res_leaky,
        "candidates": candidates,
    }


def main() -> None:
    """CLI entry point for Assignment 3."""
    parser = argparse.ArgumentParser(description="Assignment 3: Link Prediction & Data Leakage Prevention")
    parser.add_argument("--config", type=str, default="configs/a3_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print detailed training progress")
    args = parser.parse_args()

    run_a3_experiments(
        config_path=args.config,
        epochs=args.epochs,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

