"""Inference helpers for running trained models on new data.

Example
-------
>>> import torch
>>> from torch.utils.data import DataLoader
>>> from training.inference import predict
>>> model = torch.nn.Conv2d(3, 2, kernel_size=1)
>>> batch = {"image": torch.randn(1, 3, 4, 4), "mask": torch.zeros(1, 4, 4, dtype=torch.long), "id": "demo"}
>>> dataloader = DataLoader([batch], batch_size=None)
>>> outputs = list(predict(model, dataloader, device="cpu"))
>>> outputs[0]["pred"].shape
torch.Size([1, 4, 4])
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List

import torch
from torch import Tensor


def _infer_model_float_dtype(model: torch.nn.Module) -> torch.dtype:
    for tensor in model.parameters():
        if torch.is_floating_point(tensor):
            return tensor.dtype
    for tensor in model.buffers():
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return torch.float32


def predict(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: str,
) -> Iterator[Dict[str, torch.Tensor]]:
    model.eval()
    model_dtype = _infer_model_float_dtype(model)
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            if torch.is_tensor(images) and torch.is_floating_point(images) and images.dtype != model_dtype:
                images = images.to(model_dtype)
            if getattr(model, "requires_targets", False):
                outputs = model(images, None)
            else:
                outputs = model(images)

            logits: Tensor | None = None
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
            elif torch.is_tensor(outputs):
                logits = outputs
            elif isinstance(outputs, (list, tuple)) and outputs:
                logits = outputs[0]

            if logits is None:
                raise RuntimeError("Model inference did not yield logits; cannot compute predictions.")

            preds = torch.argmax(logits, dim=1, keepdim=True)
            identifier = batch.get("id")

            if isinstance(identifier, (list, tuple)):
                ids: List[object] = list(identifier)
            elif identifier is None:
                ids = [None] * preds.shape[0]
            else:
                ids = [identifier] if preds.shape[0] == 1 else [identifier for _ in range(preds.shape[0])]

            masks = batch.get("mask")
            if torch.is_tensor(masks):
                masks = masks.cpu()
            elif isinstance(masks, (list, tuple)):
                masks = [m.cpu() if torch.is_tensor(m) else m for m in masks]
            else:
                masks = [None] * preds.shape[0]

            for idx in range(preds.shape[0]):
                pred_slice = preds[idx].squeeze(0).cpu()
                image_slice = images[idx].cpu()
                sample = {
                    "id": ids[idx],
                    "pred": pred_slice,
                    "image": image_slice,
                }
                if isinstance(masks, torch.Tensor):
                    sample["mask"] = masks[idx]
                elif isinstance(masks, list):
                    sample["mask"] = masks[idx]
                yield sample
