"""PyTorch Dataset for coral segmentation imagery.

Example
-------
>>> from dataset.coral_dataset import CoralSegmentationDataset
>>> dataset = CoralSegmentationDataset("images", "masks", ids=["sample.png"])
>>> len(dataset)
1
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CoralSegmentationDataset(Dataset):
    """Loads paired RGB images and segmentation masks from disk."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        ids: Optional[Iterable[str]] = None,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], tuple]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.ids: List[str] = list(ids) if ids is not None else self._infer_ids()
        self.transform = transform

    def _infer_ids(self) -> List[str]:
        return sorted(f.name for f in self.image_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        image_id = self.ids[index]
        image = self._load_image(self.image_dir / image_id)
        mask = self._load_mask(self.mask_dir / image_id)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return {"image": image, "mask": mask, "id": image_id}

    def _load_image(self, path: Path) -> torch.Tensor:
        array = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_mask(self, path: Path) -> torch.Tensor:
        array = np.array(Image.open(path), dtype=np.int64)
        return torch.from_numpy(array)


if __name__ == "__main__":
    dataset = CoralSegmentationDataset("images", "masks", ids=["demo.png"])
    print(f"Dataset contains {len(dataset)} sample.")
