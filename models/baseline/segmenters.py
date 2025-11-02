"""Baseline segmentation backbones built on Torchvision and HuggingFace."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
)

_TRANSFORMERS_AVAILABLE = True
try:  # pragma: no cover - optional dependency
    from transformers import (
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
        SegformerConfig,
        SegformerForSemanticSegmentation,
    )
except ImportError:  # pragma: no cover - optional dependency
    _TRANSFORMERS_AVAILABLE = False


class TorchvisionSegmentationWrapper(nn.Module):
    """Wrap torchvision segmentation models to expose logits directly."""

    requires_targets: bool = False

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model_description = model.__class__.__name__

    def forward(self, images: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.model(images)
        if isinstance(outputs, dict):
            logits = outputs.get("out")
        else:
            logits = outputs
        if logits is None:
            raise RuntimeError("Torchvision model did not return segmentation logits.")
        return logits


class SegformerWrapper(nn.Module):
    """Ensure SegFormer logits are upsampled to the input spatial dimensions."""

    requires_targets: bool = False

    def __init__(self, model: "SegformerForSemanticSegmentation") -> None:
        super().__init__()
        self.model = model
        self.model_description = "SegFormer"

    def forward(self, images: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.model(pixel_values=images)
        logits = outputs.logits
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class Mask2FormerWrapper(nn.Module):
    """Use the HuggingFace Mask2Former implementation with image processor helpers."""

    requires_targets: bool = True

    def __init__(self, model: "Mask2FormerForUniversalSegmentation", processor: "Mask2FormerImageProcessor") -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.model_description = "Mask2Former"

    def forward(self, images: torch.Tensor, targets: torch.Tensor | None = None) -> Dict[str, torch.Tensor] | torch.Tensor:
        pixel_list = [img.detach().cpu() for img in images]

        if targets is not None:
            mask_list = [mask.detach().cpu() for mask in targets]
            encoded = self.processor.encode_inputs(
                pixel_values_list=pixel_list,
                segmentation_maps=mask_list,
                return_tensors="pt",
                input_data_format="channels_first",
            )

            pixel_values = encoded["pixel_values"].to(images.device)
            pixel_mask = encoded.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(images.device)

            mask_labels = [m.to(images.device) for m in encoded["mask_labels"]]
            class_labels = [c.to(images.device) for c in encoded["class_labels"]]

            outputs = self.model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
                pixel_mask=pixel_mask,
            )
            logits = outputs.logits
            if logits is not None and logits.shape[-2:] != images.shape[-2:]:
                logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
            return {"loss": outputs.loss, "logits": logits}

        encoded = self.processor(
            images=pixel_list,
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            input_data_format="channels_first",
        )
        pixel_values = encoded["pixel_values"].to(images.device)
        pixel_mask = encoded.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(images.device)

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = outputs.logits
        if logits is not None and logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def build_deeplabv3_resnet50(num_classes: int, pretrained: bool = False, **kwargs) -> nn.Module:
    """DeepLabV3 with a ResNet-50 backbone."""

    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, num_classes=num_classes, **kwargs)
    return TorchvisionSegmentationWrapper(model)


def build_deeplabv3_resnet101(num_classes: int, pretrained: bool = False, **kwargs) -> nn.Module:
    """DeepLabV3 with a ResNet-101 backbone."""

    weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet101(weights=weights, num_classes=num_classes, **kwargs)
    return TorchvisionSegmentationWrapper(model)


def _build_segformer_config(variant: str, num_classes: int) -> "SegformerConfig":
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("SegFormer requires the 'transformers' package. Install via `pip install transformers scipy`.")
    config = SegformerConfig(num_labels=num_classes)
    config.id2label = {i: f"class_{i}" for i in range(num_classes)}
    config.label2id = {name: idx for idx, name in config.id2label.items()}
    config.semantic_loss_ignore_index = 255

    variant = variant.lower()
    if variant == "b0":
        pass  # Default configuration corresponds to SegFormer B0.
    elif variant == "b1":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 256
        config.depths = [2, 2, 2, 2]
        config.num_attention_heads = [1, 2, 5, 8]
    else:
        raise ValueError(f"Unknown SegFormer variant '{variant}'.")
    return config


def build_segformer(variant: str, num_classes: int) -> nn.Module:
    config = _build_segformer_config(variant, num_classes)
    model = SegformerForSemanticSegmentation(config)
    return SegformerWrapper(model)


def _build_mask2former_config(variant: str, num_classes: int) -> "Mask2FormerConfig":
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Mask2Former requires the 'transformers' and 'scipy' packages. Install via `pip install transformers scipy`."
        )

    config = Mask2FormerConfig(num_labels=num_classes)
    config.id2label = {i: f"class_{i}" for i in range(num_classes)}
    config.label2id = {name: idx for idx, name in config.id2label.items()}
    config.ignore_value = 255
    config.semantic_loss_ignore_index = 255

    variant = variant.lower()
    backbone_config = dict(config.backbone_config)

    if variant == "swin_t":
        backbone_config.update({"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24], "drop_path_rate": 0.2})
    elif variant == "swin_s":
        backbone_config.update({"embed_dim": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24], "drop_path_rate": 0.3})
    else:
        raise ValueError(f"Unknown Mask2Former variant '{variant}'.")

    config.backbone_config = backbone_config
    return config


def build_mask2former(variant: str, num_classes: int) -> nn.Module:
    config = _build_mask2former_config(variant, num_classes)
    model = Mask2FormerForUniversalSegmentation(config)
    processor = Mask2FormerImageProcessor(
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        ignore_index=255,
    )
    return Mask2FormerWrapper(model, processor)


__all__ = [
    "Mask2FormerWrapper",
    "SegformerWrapper",
    "TorchvisionSegmentationWrapper",
    "build_deeplabv3_resnet50",
    "build_deeplabv3_resnet101",
    "build_mask2former",
    "build_segformer",
]
