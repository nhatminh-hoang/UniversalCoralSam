"""Utilities for reproducible training across seeds and worker processes."""

from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch

__all__ = ["seed_everything", "make_worker_seed_fn"]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs with optional deterministic guards."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[attr-defined]
    else:
        torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)  # type: ignore[attr-defined]


def make_worker_seed_fn(base_seed: int) -> Callable[[int], None]:
    """Create a DataLoader worker init function seeded from the provided base seed."""

    def _seed_worker(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker
