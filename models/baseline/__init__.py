"""Baseline segmentation architectures and builders."""

from __future__ import annotations

from .small_unet import SmallUNet
from .segmenters import (
    Mask2FormerWrapper,
    SegformerWrapper,
    TorchvisionSegmentationWrapper,
    build_deeplabv3_resnet101,
    build_deeplabv3_resnet50,
    build_mask2former,
    build_segformer,
)


def build_baseline_model(num_classes: int, in_channels: int = 3) -> SmallUNet:
    """Construct the default baseline model.

    Parameters
    ----------
    num_classes:
        Number of segmentation classes the model should predict.
    in_channels:
        Number of input channels; defaults to RGB images.
    """

    model = SmallUNet(in_channels=in_channels, num_classes=num_classes)
    setattr(model, "requires_targets", False)
    setattr(model, "model_description", "SmallUNet Baseline")
    return model


__all__ = [
    "Mask2FormerWrapper",
    "SegformerWrapper",
    "SmallUNet",
    "TorchvisionSegmentationWrapper",
    "build_baseline_model",
    "build_deeplabv3_resnet50",
    "build_deeplabv3_resnet101",
    "build_mask2former",
    "build_segformer",
]
