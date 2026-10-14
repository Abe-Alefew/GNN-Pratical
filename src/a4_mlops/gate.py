"""
Validation gate for model promotion (Assignment 4, Stage 4.8).

Implements five documented behaviours:
  1. **Metric**: validation macro F1 (configurable).
  2. **Threshold**: candidate must exceed incumbent by ≥ δ.
  3. **Tie behaviour**: keep incumbent (conservative).
  4. **Missing metric**: block (fail closed).
  5. **First-model**: bootstrap-promote if metric ≥ absolute floor.

Exit code 0 = promote; non-zero = block.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any


class GateDecision:
    """Result of a gate evaluation."""

    def __init__(
        self,
        promoted: bool,
        reason: str,
        candidate_score: float | None = None,
        incumbent_score: float | None = None,
    ) -> None:
        self.promoted = promoted
        self.reason = reason
        self.candidate_score = candidate_score
        self.incumbent_score = incumbent_score

    def __repr__(self) -> str:
        status = "PROMOTED" if self.promoted else "BLOCKED"
        return (
            f"GateDecision({status}, reason={self.reason!r}, "
            f"candidate={self.candidate_score}, incumbent={self.incumbent_score})"
        )


def evaluate_gate(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None,
    metric_name: str = "val_macro_f1",
    threshold_delta: float = 0.0,
    min_absolute_floor: float = 0.3,
    tie_behavior: str = "keep",
) -> GateDecision:
    """Evaluate whether to promote a candidate model.

    Args:
        candidate_metrics: Metrics dict for the candidate model.
            Must contain ``metric_name``.
        incumbent_metrics: Metrics dict for the current approved model.
            ``None`` if no model is currently approved (first-model case).
        metric_name: Key in the metrics dicts to compare.
        threshold_delta: Candidate must exceed incumbent by this amount.
        min_absolute_floor: Minimum metric for first-model bootstrap.
        tie_behavior: ``"keep"`` (block on tie) or ``"promote"``.

    Returns:
        GateDecision with promotion/block and reason.
    """
    # --- Check candidate metric exists and is valid ---
    candidate_score = candidate_metrics.get(metric_name)

    if candidate_score is None or (
        isinstance(candidate_score, float) and math.isnan(candidate_score)
    ):
        return GateDecision(
            promoted=False,
            reason=f"Candidate metric '{metric_name}' is missing or NaN",
            candidate_score=candidate_score,
        )

    # --- First-model case (no incumbent) ---
    if incumbent_metrics is None:
        if candidate_score >= min_absolute_floor:
            return GateDecision(
                promoted=True,
                reason=(
                    f"Bootstrap promotion: first model with "
                    f"{metric_name}={candidate_score:.4f} >= floor={min_absolute_floor}"
                ),
                candidate_score=candidate_score,
            )
        else:
            return GateDecision(
                promoted=False,
                reason=(
                    f"First model blocked: {metric_name}={candidate_score:.4f} "
                    f"< floor={min_absolute_floor}"
                ),
                candidate_score=candidate_score,
            )

    # --- Compare with incumbent ---
    incumbent_score = incumbent_metrics.get(metric_name)

    if incumbent_score is None or (
        isinstance(incumbent_score, float) and math.isnan(incumbent_score)
    ):
        # Incumbent has no valid metric — promote candidate if it has one
        return GateDecision(
            promoted=True,
            reason="Incumbent metric missing/NaN; candidate has valid metric",
            candidate_score=candidate_score,
            incumbent_score=incumbent_score,
        )

    diff = candidate_score - incumbent_score

    if diff > threshold_delta:
        return GateDecision(
            promoted=True,
            reason=(
                f"Improvement: {candidate_score:.4f} - {incumbent_score:.4f} "
                f"= {diff:.4f} > delta={threshold_delta}"
            ),
            candidate_score=candidate_score,
            incumbent_score=incumbent_score,
        )
    elif abs(diff) <= threshold_delta:
        # Tie (within threshold)
        if tie_behavior == "promote":
            return GateDecision(
                promoted=True,
                reason=f"Tie (diff={diff:.4f} <= delta={threshold_delta}), promoting",
                candidate_score=candidate_score,
                incumbent_score=incumbent_score,
            )
        else:
            return GateDecision(
                promoted=False,
                reason=(
                    f"Tie (diff={diff:.4f} <= delta={threshold_delta}), "
                    f"keeping incumbent (conservative)"
                ),
                candidate_score=candidate_score,
                incumbent_score=incumbent_score,
            )
    else:
        # Regression
        return GateDecision(
            promoted=False,
            reason=(
                f"Regression: {candidate_score:.4f} - {incumbent_score:.4f} "
                f"= {diff:.4f} < delta={threshold_delta}"
            ),
            candidate_score=candidate_score,
            incumbent_score=incumbent_score,
        )


def run_gate_cli(
    candidate_metrics_path: str,
    incumbent_metrics_path: str | None = None,
    config_path: str = "configs/a4_config.yaml",
) -> int:
    """CLI entry point for the validation gate.

    Reads metrics from JSON files, evaluates the gate, prints the result,
    and returns the exit code (0 = promote, 1 = block).

    Args:
        candidate_metrics_path: Path to candidate metrics JSON.
        incumbent_metrics_path: Path to incumbent metrics JSON (optional).
        config_path: Path to config for gate parameters.

    Returns:
        Exit code: 0 = promote, 1 = block.
    """
    from src.common.config import load_config

    cfg = load_config(config_path)
    gate_cfg = cfg.get("gate", {})

    with open(candidate_metrics_path, "r", encoding="utf-8") as f:
        candidate_metrics = json.load(f)

    incumbent_metrics = None
    if incumbent_metrics_path and Path(incumbent_metrics_path).exists():
        with open(incumbent_metrics_path, "r", encoding="utf-8") as f:
            incumbent_metrics = json.load(f)

    decision = evaluate_gate(
        candidate_metrics=candidate_metrics,
        incumbent_metrics=incumbent_metrics,
        metric_name=gate_cfg.get("metric", "val_macro_f1"),
        threshold_delta=gate_cfg.get("threshold_delta", 0.0),
        min_absolute_floor=gate_cfg.get("min_absolute_floor", 0.3),
        tie_behavior=gate_cfg.get("tie_behavior", "keep"),
    )

    print(decision)
    return 0 if decision.promoted else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.a4_mlops.gate <candidate.json> [incumbent.json]")
        sys.exit(2)

    candidate_path = sys.argv[1]
    incumbent_path = sys.argv[2] if len(sys.argv) > 2 else None
    exit_code = run_gate_cli(candidate_path, incumbent_path)
    sys.exit(exit_code)
