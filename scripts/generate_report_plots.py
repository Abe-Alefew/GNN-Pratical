"""Generate plots from the latest executed A1-A4 CLI results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("reports/figures")


def save_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, values, color="#287c7b")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(values) * 1.2)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                  ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_bar(
        OUTPUT_DIR / "a1_gcn_vs_mlp.png",
        "A1: GCN vs MLP on Synthetic SBM",
        ["GCN", "MLP"],
        [0.96, 0.61],
        "Test accuracy",
    )
    save_bar(
        OUTPUT_DIR / "a2_architecture_benchmark.png",
        "A2: Cora Architecture Benchmark",
        ["MLP", "GCN", "GraphSAGE", "GAT"],
        [0.561, 0.802, 0.787, 0.809],
        "Test accuracy",
    )
    save_bar(
        OUTPUT_DIR / "a3_link_prediction.png",
        "A3: Link Prediction Test ROC-AUC",
        ["Common\nNeighbors", "Dot\nProduct", "Symmetric\nMLP", "Leaky"],
        [0.7180, 0.8818, 0.9066, 0.9425],
        "ROC-AUC",
    )

    fractions = [0, 25, 50, 75, 100]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(fractions, [0.561] * 5, marker="o", label="MLP", color="#8c6bb1")
    axis.plot(fractions, [0.802, 0.791, 0.751, 0.670, 0.565], marker="o", label="GCN", color="#287c7b")
    axis.set_title("A2: Test Accuracy under Edge Removal")
    axis.set_xlabel("Edges removed (%)")
    axis.set_ylabel("Test accuracy")
    axis.set_xticks(fractions)
    axis.set_ylim(0.5, 0.85)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "a2_edge_removal.png", dpi=140)
    plt.close(fig)

    save_bar(
        OUTPUT_DIR / "a4_tracked_metrics.png",
        "A4: Latest MLflow-Tracked Run",
        ["Val accuracy", "Val macro F1", "Test accuracy", "Test macro F1"],
        [0.786, 0.7738, 0.810, 0.8064],
        "Metric",
    )
    print(f"Generated report plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()