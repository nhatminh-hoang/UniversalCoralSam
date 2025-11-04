"""Reusable transforms for the HKCoral dataset."""

from __future__ import annotations

import torch
import torch.nn.functional as F

TARGET_SIZE = (512, 1024)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def resize_to_target(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize HKCoral samples to the training resolution and apply ImageNet normalization."""

    image = F.interpolate(image.unsqueeze(0), size=TARGET_SIZE, mode="bilinear", align_corners=False).squeeze(0)
    mask = (
        F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=TARGET_SIZE,
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .long()
    )
    image = (image - IMAGENET_MEAN.to(image.device)) / IMAGENET_STD.to(image.device)
    return image, mask


__all__ = ["resize_to_target", "TARGET_SIZE", "IMAGENET_MEAN", "IMAGENET_STD"]

