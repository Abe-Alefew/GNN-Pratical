import numpy as np
import torch


def generate_sbm_dataset(
    seed: int,
    num_nodes: int = 200,
    num_classes: int = 2,
    num_features: int = 3,
    p_in: float = 0.08,
    p_out: float = 0.01,
    feature_sigma: float = 1.5,
    class_means: list[list[float]] | None = None,
) -> dict[str, torch.Tensor]:
    rng = np.random.RandomState(seed)

    # --- Labels ---
    nodes_per_class = num_nodes // num_classes
    labels = np.repeat(np.arange(num_classes), nodes_per_class)
    # Handle remainder nodes
    remainder = num_nodes - len(labels)
    if remainder > 0:
        labels = np.concatenate([labels, rng.randint(0, num_classes, remainder)])

    # --- Features (class-conditional Gaussians) ---
    if class_means is None:
        # Default: canonical basis vectors (padded/truncated to num_features)
        class_means_arr = np.zeros((num_classes, num_features), dtype=np.float32)
        for c in range(num_classes):
            if c < num_features:
                class_means_arr[c, c] = 1.0
    else:
        class_means_arr = np.array(class_means, dtype=np.float32)
        assert class_means_arr.shape == (num_classes, num_features), (
            f"class_means shape mismatch: expected ({num_classes}, {num_features}), "
            f"got {class_means_arr.shape}"
        )

    features = np.zeros((num_nodes, num_features), dtype=np.float32)
    for i in range(num_nodes):
        features[i] = rng.normal(
            loc=class_means_arr[labels[i]],
            scale=feature_sigma,
            size=num_features,
        )

    # --- Adjacency matrix (SBM) ---
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.rand() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # --- Edge index from adjacency ---
    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst], axis=0)

    return {
        "x": torch.tensor(features, dtype=torch.float32),
        "y": torch.tensor(labels, dtype=torch.long),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "adj": torch.tensor(adj, dtype=torch.float32),
    }


def compute_edge_homophily(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return float(same_label.mean().item())


def create_split_masks(
    num_nodes: int,
    labels: torch.Tensor,
    train_ratio: float = 0.3,
    val_ratio: float = 0.2,
    test_ratio: float = 0.5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1, got {train_ratio + val_ratio + test_ratio}"
    )

    rng = np.random.RandomState(seed)
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    for c in np.unique(labels_np):
        class_indices = np.where(labels_np == c)[0]
        rng.shuffle(class_indices)

        n_class = len(class_indices)
        n_train = max(1, int(n_class * train_ratio))
        n_val = max(1, int(n_class * val_ratio))

        train_mask[class_indices[:n_train]] = True
        val_mask[class_indices[n_train : n_train + n_val]] = True
        test_mask[class_indices[n_train + n_val :]] = True

    return (
        torch.tensor(train_mask),
        torch.tensor(val_mask),
        torch.tensor(test_mask),
    )
