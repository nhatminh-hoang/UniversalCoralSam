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


def predict(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: str,
) -> Iterator[Dict[str, torch.Tensor]]:
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
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
                sample = {
                    "id": ids[idx],
                    "pred": preds[idx : idx + 1].cpu(),
                    "image": images[idx].cpu(),
                }
                if isinstance(masks, torch.Tensor):
                    sample["mask"] = masks[idx]
                elif isinstance(masks, list):
                    sample["mask"] = masks[idx]
                yield sample
