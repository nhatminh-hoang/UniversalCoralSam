"""Training loop helpers built around PyTorch standard APIs."""

from __future__ import annotations

import time
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
    ignore_labels: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    model.train()
    metric_logger = metric_logger or MetricLogger()
    device_obj = torch.device(device)
    device_str = str(device_obj)

    def _sync_device() -> None:
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)

    metric_aggregator = None
    if num_classes is not None:
        metric_aggregator = SegmentationMetricAggregator(
            num_classes=num_classes,
            ignore_index=ignore_index,
            device=device_obj,
            ignore_labels=tuple(ignore_labels) if ignore_labels is not None else None,
        )
    batch_iter = tqdm(dataloader, desc="train", leave=False)
    data_time_total = 0.0
    forward_time_total = 0.0
    backward_time_total = 0.0
    optim_time_total = 0.0
    log_time_total = 0.0
    total_samples = 0
    batch_count = 0
    epoch_start = time.perf_counter()
    load_start = epoch_start

    for batch in batch_iter:
        batch_count += 1
        batch_size = batch["image"].shape[0] if isinstance(batch, dict) and "image" in batch else 0
        total_samples += batch_size

        load_end = time.perf_counter()
        data_time = load_end - load_start

        images = batch["image"].to(device_str)
        targets = batch["mask"].to(device_str)
        optimizer.zero_grad()

        forward_start = time.perf_counter()
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

        _sync_device()
        forward_end = time.perf_counter()

        loss.backward()
        _sync_device()
        backward_end = time.perf_counter()

        optimizer.step()
        _sync_device()
        optim_end = time.perf_counter()

        log_start = optim_end
        loss_value = float(loss.detach())

        metric_logger.update(loss=loss_value)
        if metric_aggregator is not None and logits is not None:
            preds = torch.argmax(logits.detach(), dim=1)
            metric_aggregator.update(preds, targets.detach())
        log_end = time.perf_counter()

        data_time_total += data_time
        forward_time_total += forward_end - forward_start
        backward_time_total += backward_end - forward_end
        optim_time_total += optim_end - backward_end
        log_time_total += log_end - log_start

        batch_iter.set_postfix(
            {
                "loss": f"{loss_value:.3f}",
                "data_ms": f"{data_time * 1e3:.1f}",
                "fwd_ms": f"{(forward_end - forward_start) * 1e3:.1f}",
                "bwd_ms": f"{(backward_end - forward_end) * 1e3:.1f}",
                "opt_ms": f"{(optim_end - backward_end) * 1e3:.1f}",
                "log_ms": f"{(log_end - log_start) * 1e3:.1f}",
            }
        )

        load_start = time.perf_counter()

    summary = metric_logger.summary()
    if metric_aggregator is not None:
        summary.update(metric_aggregator.summary())

    epoch_time = time.perf_counter() - epoch_start
    if batch_count > 0:
        summary.update(
            {
                "avg_data_time_s": data_time_total / batch_count,
                "avg_forward_time_s": forward_time_total / batch_count,
                "avg_backward_time_s": backward_time_total / batch_count,
                "avg_optimizer_time_s": optim_time_total / batch_count,
                "avg_logging_time_s": log_time_total / batch_count,
            }
        )
    summary["epoch_time_s"] = epoch_time
    if epoch_time > 0 and total_samples > 0:
        summary["samples_per_sec"] = total_samples / epoch_time
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
    ignore_labels: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """Run a multi-epoch training loop and optionally log curves."""

    summary: Dict[str, float] = {}
    epoch_iter = tqdm(range(epochs), desc="epochs")
    for epoch in epoch_iter:
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
            ignore_labels=ignore_labels,
        )
        if curve_writer is not None:
            curve_writer.log_step(epoch, metrics)
        if scheduler is not None:
            scheduler.step()
        summary = metrics
        epoch_iter.set_postfix(loss=metrics.get("loss", 0.0), miou=metrics.get("miou", 0.0))
    if curve_writer is not None:
        curve_writer.save()
    return summary
