"""Common image tensor helpers used across training and evaluation scripts."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW or HW tensor in [0, 1] or [0, 255] to a uint8 HWC array."""

    tensor_cpu = tensor.detach().cpu()
    if tensor_cpu.dtype == torch.bfloat16:
        tensor_cpu = tensor_cpu.to(torch.float32)
    array = tensor_cpu.numpy()
    if array.ndim == 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0) * 255.0
        array = array.astype(np.uint8)
    return array


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    """Extract a centered square crop of the requested size."""

    if size <= 0:
        raise ValueError("Crop size must be positive.")

    height, width = array.shape[:2]
    crop = min(size, height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    bottom = top + crop
    right = left + crop
    return array[top:bottom, left:right].copy()


def ensure_square(array: np.ndarray) -> np.ndarray:
    """Crop the largest centered square available from the array."""

    height, width = array.shape[:2]
    size = min(height, width)
    return center_crop(array, size)


__all__ = ["center_crop", "ensure_square", "tensor_to_image"]
