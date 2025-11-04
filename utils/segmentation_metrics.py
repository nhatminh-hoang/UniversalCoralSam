"""Segmentation metric helpers for computing accuracy and mean IoU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import torch


@dataclass
class SegmentationMetricAggregator:
    """Accumulates confusion matrices to compute accuracy and IoU."""

    num_classes: int
    ignore_index: Optional[int] = None
    device: Optional[str | torch.device] = None
    ignore_labels: Optional[Iterable[int]] = None
    _ignored_values: set[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        device = torch.device(self.device) if self.device is not None else None
        dtype = torch.float64
        self._confusion = torch.zeros((self.num_classes, self.num_classes), dtype=dtype, device=device)
        self._correct = torch.zeros((), dtype=dtype, device=device)
        self._total = torch.zeros((), dtype=dtype, device=device)
        ignored: set[int] = set()
        if self.ignore_index is not None:
            ignored.add(int(self.ignore_index))
        if self.ignore_labels is not None:
            ignored.update(int(value) for value in self.ignore_labels)
        self._ignored_values = ignored

    def _ensure_device(self, tensor_device: torch.device) -> None:
        if self._confusion.device != tensor_device:
            self._confusion = self._confusion.to(tensor_device)
            self._correct = self._correct.to(tensor_device)
            self._total = self._total.to(tensor_device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach()
        targets = targets.detach()
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")

        self._ensure_device(preds.device)

        if self._ignored_values:
            ignore_mask = torch.zeros_like(targets, dtype=torch.bool)
            for value in self._ignored_values:
                ignore_mask |= targets == value
            mask = ~ignore_mask
            preds = preds[mask]
            targets = targets[mask]

        if preds.numel() == 0:
            return

        preds = preds.view(-1).to(dtype=torch.int64)
        targets = targets.view(-1).to(dtype=torch.int64)

        self._total += torch.tensor(targets.numel(), dtype=self._total.dtype, device=self._total.device)
        correct = (preds == targets).sum().to(dtype=self._correct.dtype)
        self._correct += correct

        indices = targets * self.num_classes + preds
        conf = torch.bincount(indices, minlength=self.num_classes * self.num_classes).reshape(
            self.num_classes, self.num_classes
        )
        self._confusion += conf.to(dtype=self._confusion.dtype)

    def _intersection_union(self) -> tuple[torch.Tensor, torch.Tensor]:
        confusion = self._confusion.detach().cpu()
        intersection = torch.diag(confusion)
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
        return intersection, union

    def accuracy(self) -> float:
        total = float(self._total.detach().cpu().item())
        if total == 0.0:
            return 0.0
        correct = float(self._correct.detach().cpu().item())
        return correct / total

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
