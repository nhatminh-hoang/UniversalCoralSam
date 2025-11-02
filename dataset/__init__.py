"""Dataset utilities and dataloader registry for segmentation experiments."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from .coral_dataset import CoralSegmentationDataset
from .hk_coral_dataset import HKCoralDataset, build_hk_coral_dataloader


def build_coral_dataloader(
    *,
    image_dir: str,
    mask_dir: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    ids: Optional[Iterable[str]] = None,
    transform=None,
) -> DataLoader:
    """Create a DataLoader for directory-based coral segmentation datasets.

    Example
    -------
    >>> from dataset import build_coral_dataloader
    >>> loader = build_coral_dataloader(
    ...     image_dir="data/images",
    ...     mask_dir="data/masks",
    ...     batch_size=2,
    ... )
    >>> isinstance(next(iter(loader))["image"], torch.Tensor)
    True
    """

    dataset = CoralSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, ids=ids, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


DATASET_BUILDERS: Dict[str, Callable[..., DataLoader]] = {
    "coral": build_coral_dataloader,
    "hkcoral": build_hk_coral_dataloader,
}


def create_dataloader(name: str, **kwargs) -> DataLoader:
    """Instantiate a dataloader from the registered dataset builders."""

    key = name.lower()
    try:
        builder = DATASET_BUILDERS[key]
    except KeyError as exc:
        raise NotImplementedError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_BUILDERS)}") from exc
    return builder(**kwargs)


def build_dataloader(**kwargs) -> DataLoader:
    """Backward-compatible alias for the coral dataloader builder."""

    return build_coral_dataloader(**kwargs)


__all__ = [
    "DATASET_BUILDERS",
    "CoralSegmentationDataset",
    "HKCoralDataset",
    "build_coral_dataloader",
    "build_hk_coral_dataloader",
    "build_dataloader",
    "create_dataloader",
]

