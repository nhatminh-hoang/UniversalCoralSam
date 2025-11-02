"""Segmentation metric helpers for computing accuracy and mean IoU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class SegmentationMetricAggregator:
    """Accumulates confusion matrices to compute accuracy and IoU."""

    num_classes: int
    ignore_index: Optional[int] = None

    def __post_init__(self) -> None:
        self._confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)
        self._correct: float = 0.0
        self._total: float = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().to(device="cpu", dtype=torch.int64)
        targets = targets.detach().to(device="cpu", dtype=torch.int64)
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        if preds.numel() == 0:
            return

        preds = preds.view(-1)
        targets = targets.view(-1)

        self._total += targets.numel()
        self._correct += (preds == targets).sum().item()

        indices = targets * self.num_classes + preds
        conf = torch.bincount(indices, minlength=self.num_classes * self.num_classes).reshape(
            self.num_classes, self.num_classes
        )
        self._confusion += conf.double()

    def accuracy(self) -> float:
        if self._total == 0:
            return 0.0
        return float(self._correct / self._total)

    def _intersection_union(self) -> tuple[torch.Tensor, torch.Tensor]:
        intersection = torch.diag(self._confusion)
        union = self._confusion.sum(dim=0) + self._confusion.sum(dim=1) - intersection
        return intersection, union

    def per_class_iou(self) -> torch.Tensor:
        intersection, union = self._intersection_union()
        iou = torch.zeros(self.num_classes, dtype=torch.float64)
        valid = union > 0
        iou[valid] = intersection[valid] / union[valid]
        return iou

    def mean_iou(self) -> float:
        intersection, union = self._intersection_union()
        valid = union > 0
        if valid.sum() == 0:
            return 0.0
        iou = torch.zeros(self.num_classes, dtype=torch.float64)
        iou[valid] = intersection[valid] / union[valid]
        return float(iou[valid].mean().item())

    def summary(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy(),
            "miou": self.mean_iou(),
        }


__all__ = ["SegmentationMetricAggregator"]
