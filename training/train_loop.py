"""Training loop helpers built around PyTorch standard APIs.

Example
-------
>>> import torch
>>> from torch.utils.data import DataLoader
>>> from training.train_loop import train_one_epoch
>>> model = torch.nn.Conv2d(3, 2, kernel_size=1)
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
>>> loss_fn = torch.nn.CrossEntropyLoss()
>>> dummy_batch = {"image": torch.randn(2, 3, 4, 4), "mask": torch.zeros(2, 4, 4, dtype=torch.long)}
>>> dataloader = DataLoader([dummy_batch], batch_size=None)
>>> metrics = train_one_epoch(model, dataloader, optimizer, loss_fn, device="cpu")
>>> "loss" in metrics
True
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.curves import TrainingCurveWriter
from utils.metrics import MetricLogger
from utils.segmentation_metrics import SegmentationMetricAggregator


def _autocast_context(device: str | torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return nullcontext()
    dev = torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=dtype)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str | torch.device,
    *,
    metric_logger: Optional[MetricLogger] = None,
    amp_dtype: Optional[torch.dtype] = None,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    metric_logger = metric_logger or MetricLogger()
    device_str = str(device)
    metric_aggregator = (
        SegmentationMetricAggregator(num_classes=num_classes, ignore_index=ignore_index)
        if num_classes is not None
        else None
    )
    iterator = tqdm(dataloader, desc="train", leave=False)
    for batch in iterator:
        images = batch["image"].to(device_str)
        targets = batch["mask"].to(device_str)
        optimizer.zero_grad()

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
                    raise RuntimeError(
                        "Model forward pass did not return logits or an explicit loss, unable to continue training."
                    )
                loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()
        metric_logger.update(loss=float(loss.detach()))
        if metric_aggregator is not None and logits is not None:
            preds = torch.argmax(logits.detach(), dim=1)
            metric_aggregator.update(preds, targets.detach())

    summary = metric_logger.summary()
    if metric_aggregator is not None:
        summary.update(metric_aggregator.summary())
    return summary


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str | torch.device,
    epochs: int,
    *,
    curve_writer: Optional[TrainingCurveWriter] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    amp_dtype: Optional[torch.dtype] = None,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    """Run a multi-epoch training loop and optionally log curves."""

    summary: Dict[str, float] = {}
    epoch_iterator = tqdm(range(epochs), desc="epochs")
    for epoch in epoch_iterator:
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            metric_logger=None,
            amp_dtype=amp_dtype,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        if curve_writer is not None:
            curve_writer.log_step(epoch, metrics)
        if scheduler is not None:
            scheduler.step()
        summary = metrics
    if curve_writer is not None:
        curve_writer.save()
    return summary
