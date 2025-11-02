"""Evaluation utilities for semantic segmentation models.

Example
-------
>>> import torch
>>> from torch.utils.data import DataLoader
>>> from training.evaluation import evaluate
>>> model = torch.nn.Conv2d(3, 2, kernel_size=1)
>>> loss_fn = torch.nn.CrossEntropyLoss()
>>> batch = {"image": torch.randn(1, 3, 4, 4), "mask": torch.zeros(1, 4, 4, dtype=torch.long)}
>>> dataloader = DataLoader([batch], batch_size=None)
>>> metrics = evaluate(model, dataloader, loss_fn, device="cpu")
>>> "loss" in metrics
True
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor

from utils.metrics import MetricLogger
from utils.segmentation_metrics import SegmentationMetricAggregator


def _autocast_context(device: str | torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return nullcontext()
    dev = torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=dtype)


def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    loss_fn: torch.nn.Module,
    device: str | torch.device,
    *,
    amp_dtype: Optional[torch.dtype] = None,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    logger = MetricLogger()
    metric_aggregator = (
        SegmentationMetricAggregator(num_classes=num_classes, ignore_index=ignore_index)
        if num_classes is not None
        else None
    )
    device_str = str(device)
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device_str)
            targets = batch["mask"].to(device_str)
            with _autocast_context(device, amp_dtype):
                if getattr(model, "requires_targets", False):
                    outputs = model(images, targets)
                else:
                    outputs = model(images)

                logits: Tensor | None = None
                loss: Tensor | None = None

                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                    loss = outputs.get("loss")
                elif torch.is_tensor(outputs):
                    logits = outputs
                elif isinstance(outputs, (list, tuple)) and outputs:
                    logits = outputs[0]
                else:
                    raise TypeError(f"Unexpected model output type: {type(outputs)}")

                if loss is None:
                    if logits is None:
                        raise RuntimeError("Model evaluation produced neither logits nor loss.")
                    loss = loss_fn(logits, targets)
            logger.update(loss=float(loss.detach()))
            if metric_aggregator is not None and logits is not None:
                preds = torch.argmax(logits.detach(), dim=1)
                metric_aggregator.update(preds, targets.detach())
    summary = logger.summary()
    if metric_aggregator is not None:
        summary.update(metric_aggregator.summary())
    return summary


