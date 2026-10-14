import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all random number generators for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, also enable PyTorch deterministic algorithms.
            Note: some operations (e.g. scatter/index_add on CUDA) may raise
            errors when deterministic mode is active.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Hash-based seeding for Python dict/set ordering
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        # cuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
