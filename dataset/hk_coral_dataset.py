"""PyTorch dataset implementation for the HKCoral benchmark.

The HKCoral dataset follows a Cityscapes-like layout::

    root/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/

Each image ``foo.jpg`` has a matching label named ``foo_labelTrainIds.png``. The
labels follow TrainIds semantics, where ``255`` denotes ignore regions. This
dataset class keeps the labels untouched so loss functions can leverage the
ignore value directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class HKCoralDataset(Dataset):
    """Loads HKCoral RGB images and their segmentation TrainId masks."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
        ids: Optional[Iterable[str]] = None,
    ) -> None:
        valid_splits = {"train", "val", "test"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Expected one of {sorted(valid_splits)}.")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "labels" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory '{self.image_dir}' does not exist.")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Label directory '{self.mask_dir}' does not exist.")

        if ids is None:
            self.ids = self._discover_ids()
        else:
            self.ids = list(ids)

    def _discover_ids(self) -> List[str]:
        return sorted(path.stem for path in self.image_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        image = self._load_image(self.image_dir / f"{image_id}.jpg")
        mask_path = self.mask_dir / f"{image_id}_labelTrainIds.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing label file '{mask_path}'.")
        mask = self._load_mask(mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if mask is None:
            raise RuntimeError("Mask is None. HKCoralDataset currently expects masks to be available.")

        return {"image": image, "mask": mask, "id": f"{image_id}.jpg"}

    def _load_image(self, path: Path) -> torch.Tensor:
        array = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_mask(self, path: Path) -> torch.Tensor:
        array = np.array(Image.open(path), dtype=np.int64)
        return torch.from_numpy(array)


def build_hk_coral_dataloader(
    *,
    root: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
    ids: Optional[Iterable[str]] = None,
) -> DataLoader:
    """Create a DataLoader for the HKCoral dataset.

    Example
    -------
    >>> from dataset.hk_coral_dataset import build_hk_coral_dataloader
    >>> loader = build_hk_coral_dataloader(
    ...     root="HKCoral",
    ...     split="train",
    ...     batch_size=2,
    ...     shuffle=True,
    ...     num_workers=0,
    ... )
    >>> next(iter(loader))["image"].shape[0]
    2
    """

    dataset = HKCoralDataset(root=root, split=split, transform=transform, ids=ids)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


__all__ = ["HKCoralDataset", "build_hk_coral_dataloader"]
