"""
Untracked baseline training script (Assignment 4, Stage 4.1).

This is deliberately bare: hard-coded hyperparameters, no seed control,
no logging, no packaging.  It exists as the "before" example showing
what an untracked script looks like and what its limitations are.

Limitations (to document in report):
  1. Results not comparable — configs are lost between runs.
  2. Randomness not controlled — no seed, so results vary silently.
  3. No artifact versioning — model on disk has no metadata.
  4. No automated testing or gate — a worse model can overwrite a better one.
  5. Deployment is manual and not reproducible.
  6. No monitoring after release.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from src.a2_cora.models import GCN


def train_untracked(epochs: int = 200) -> float:
    """Train a GCN on Cora with no tracking, returning test accuracy."""
    # Hard-coded everything
    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0]

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
        dropout=0.5,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

    return acc


def main() -> None:
    """Run untracked training and save unversioned checkpoint."""
    acc = train_untracked(epochs=200)
    print(f"Test accuracy: {acc:.4f}")

    # Save (no metadata, no version)
    # torch.save(...)
    print("Model trained without metadata tracking.")


if __name__ == "__main__":
    main()

