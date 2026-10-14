"""
Dataset fingerprinting.

Computes a SHA-256 hash over the raw byte contents of features, edge index,
and labels so that the *same* data always produces the *same* identifier.
Any modification to the data (even a single bit-flip) changes the hash.

This is used for:
  - Linking MLflow runs to exact dataset versions.
  - Verifying that two experiments used identical data.
"""

import hashlib

import numpy as np
import torch


def compute_dataset_hash(
    features: torch.Tensor | np.ndarray,
    edge_index: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> str:
    h = hashlib.sha256()

    for tensor in [features, edge_index, labels]:
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
        # Use contiguous C-order bytes for determinism
        h.update(np.ascontiguousarray(arr).tobytes())

    return h.hexdigest()
