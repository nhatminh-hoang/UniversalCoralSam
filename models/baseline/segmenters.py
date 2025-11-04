"""Baseline segmentation backbones built on Torchvision and HuggingFace."""

from __future__ import annotations

import os
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

from .._huggingface import (
    can_use_bfloat16 as _can_use_bfloat16,
    maybe_raise_torch_upgrade as _maybe_raise_torch_upgrade,
    resolve_pretrained_identifier as _resolve_pretrained_identifier,
    snapshot_has_safetensors as _snapshot_has_safetensors,
)

_TRANSFORMERS_AVAILABLE = True
try:  # pragma: no cover - optional dependency
    from transformers import (
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
        SegformerConfig,
        SegformerForSemanticSegmentation,
    )
except ImportError:  # pragma: no cover - optional dependency
    _TRANSFORMERS_AVAILABLE = False


_MASK2FORMER_PRETRAINED_VARIANTS: Dict[str, str] = {
    "mask2former_swin_base": "facebook/mask2former-swin-base-ade-semantic",
    "mask2former_swin_large": "facebook/mask2former-swin-large-ade-semantic",
}

_SEGFORMER_PRETRAINED_VARIANTS: Dict[str, str] = {
    "segformer_b2_cityscapes": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "segformer_b5_cityscapes": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


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

    def __init__(self, model: "SegformerForSemanticSegmentation", variant: str | None = None) -> None:
        super().__init__()
        self.model = model
        self.model_description = f"SegFormer ({variant})" if variant else "SegFormer"

    def forward(self, images: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.model(pixel_values=images)
        logits = outputs.logits
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class Mask2FormerWrapper(nn.Module):
    """Use the HuggingFace Mask2Former implementation."""

    requires_targets: bool = True

    def __init__(
        self,
        model: "Mask2FormerForUniversalSegmentation",
        ignore_index: int | None = None,
        *,
        variant: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_description = f"Mask2Former ({variant})" if variant else "Mask2Former"
        self._ignore_index = (
            ignore_index if ignore_index is not None else getattr(model.config, "semantic_loss_ignore_index", None)
        )

    @staticmethod
    def _build_semantic_logits(
        outputs: "Mask2FormerForUniversalSegmentationOutput", target_size: tuple[int, int]
    ) -> torch.Tensor | None:
        masks_queries_logits = getattr(outputs, "masks_queries_logits", None)
        class_queries_logits = getattr(outputs, "class_queries_logits", None)
        if masks_queries_logits is None or class_queries_logits is None:
            return None

        masks_queries_logits = F.interpolate(
            masks_queries_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        masks_probs = masks_queries_logits.sigmoid()

        # Remove the null class (last index) and compute weighted per-class logits.
        class_probs = class_queries_logits.softmax(dim=-1)[..., :-1]
        if class_probs.shape[-1] == 0:
            return None
        logits = torch.einsum("bqc,bqhw->bchw", class_probs, masks_probs)
        return logits

    def _prepare_training_targets(
        self, targets: torch.Tensor, device: torch.device
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        ignore_index = (
            self._ignore_index if self._ignore_index is not None else getattr(self.model.config, "ignore_value", None)
        )
        mask_labels: list[torch.Tensor] = []
        class_labels: list[torch.Tensor] = []
        mask_dtype = torch.float32

        for target in targets:
            unique_classes = torch.unique(target)
            if ignore_index is not None:
                unique_classes = unique_classes[unique_classes != ignore_index]

            if unique_classes.numel() == 0:
                mask_labels.append(torch.zeros((0, *target.shape[-2:]), dtype=mask_dtype, device=device))
                class_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
                continue

            masks = (target.unsqueeze(0) == unique_classes.view(-1, 1, 1)).to(mask_dtype)
            mask_labels.append(masks.to(device))
            class_labels.append(unique_classes.to(device=device, dtype=torch.long))

        return mask_labels, class_labels

    def forward(self, images: torch.Tensor, targets: torch.Tensor | None = None) -> Dict[str, torch.Tensor] | torch.Tensor:
        target_size = (images.shape[-2], images.shape[-1])
        pixel_values = images
        pixel_mask = None

        if targets is not None:
            mask_labels, class_labels = self._prepare_training_targets(targets, images.device)

            outputs = self.model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
                pixel_mask=pixel_mask,
            )
            logits = self._build_semantic_logits(outputs, target_size)
            return {"loss": outputs.loss, "logits": logits}

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = self._build_semantic_logits(outputs, target_size)
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
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("SegFormer requires the 'transformers' package. Install via `pip install transformers scipy`.")

    variant_key = variant.lower()
    pretrained_repo = _SEGFORMER_PRETRAINED_VARIANTS.get(variant_key)
    variant_name = variant_key if pretrained_repo is not None else variant
    load_identifier = variant
    local_only = False
    load_kwargs: Dict[str, object] = {}
    use_safetensors = False

    if pretrained_repo is not None:
        load_identifier, local_only = _resolve_pretrained_identifier(pretrained_repo)
        load_kwargs["local_files_only"] = local_only
        cache_dir = os.environ.get("HF_HOME")
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        if _can_use_bfloat16():
            load_kwargs["dtype"] = torch.bfloat16
        if local_only and _snapshot_has_safetensors(load_identifier):
            use_safetensors = True
        if not local_only:
            use_safetensors = True

    if use_safetensors:
        load_kwargs["use_safetensors"] = True

    try:
        if "/" in load_identifier:
            model = SegformerForSemanticSegmentation.from_pretrained(
                load_identifier,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                **load_kwargs,
            )
        else:
            config = _build_segformer_config(load_identifier, num_classes)
            model = SegformerForSemanticSegmentation(config)
            if _can_use_bfloat16():
                model = model.to(torch.bfloat16)
    except ValueError as exc:  # pragma: no cover - guard against torch<2.6 restrictions
        _maybe_raise_torch_upgrade(exc, pretrained_repo or load_identifier)
        raise

    config = model.config
    config.num_labels = num_classes
    config.id2label = {i: f"class_{i}" for i in range(num_classes)}
    config.label2id = {name: idx for idx, name in config.id2label.items()}
    config.semantic_loss_ignore_index = 255
    if _can_use_bfloat16():
        config.dtype = torch.bfloat16

    return SegformerWrapper(model, variant=variant_name)


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
    backbone_config = config.backbone_config.to_dict()

    if variant == "swin_t":
        backbone_config.update({"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24], "drop_path_rate": 0.2})
    elif variant == "swin_s":
        backbone_config.update({"embed_dim": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24], "drop_path_rate": 0.3})
    else:
        raise ValueError(f"Unknown Mask2Former variant '{variant}'.")

    config.backbone_config = type(config.backbone_config)(**backbone_config)
    return config


def build_mask2former(variant: str, num_classes: int) -> nn.Module:
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Mask2Former requires the 'transformers' and 'scipy' packages. Install via `pip install transformers scipy`."
        )

    variant_key = variant.lower()
    pretrained_repo = _MASK2FORMER_PRETRAINED_VARIANTS.get(variant_key)
    variant_name = variant_key if pretrained_repo is not None else variant
    load_identifier = variant
    local_only = False
    load_kwargs: Dict[str, object] = {}
    use_safetensors = False

    if pretrained_repo is not None:
        load_identifier, local_only = _resolve_pretrained_identifier(pretrained_repo)
        load_kwargs["local_files_only"] = local_only
        cache_dir = os.environ.get("HF_HOME")
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        if _can_use_bfloat16():
            load_kwargs["dtype"] = torch.bfloat16
        if local_only and _snapshot_has_safetensors(load_identifier):
            use_safetensors = True
        if not local_only:
            use_safetensors = True

    if use_safetensors:
        load_kwargs["use_safetensors"] = True

    try:
        if "/" in load_identifier:
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                load_identifier,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                **load_kwargs,
            )
        else:
            config = _build_mask2former_config(load_identifier, num_classes)
            model = Mask2FormerForUniversalSegmentation(config)
            if _can_use_bfloat16():
                model = model.to(torch.bfloat16)
    except ValueError as exc:  # pragma: no cover - guard against torch<2.6 restrictions
        _maybe_raise_torch_upgrade(exc, pretrained_repo or load_identifier)
        raise

    config = model.config
    config.num_labels = num_classes
    config.id2label = {i: f"class_{i}" for i in range(num_classes)}
    config.label2id = {name: idx for idx, name in config.id2label.items()}
    config.ignore_value = 255
    config.semantic_loss_ignore_index = 255
    if _can_use_bfloat16():
        config.dtype = torch.bfloat16

    return Mask2FormerWrapper(
        model,
        ignore_index=config.semantic_loss_ignore_index,
        variant=variant_name,
    )


__all__ = [
    "Mask2FormerWrapper",
    "SegformerWrapper",
    "TorchvisionSegmentationWrapper",
    "build_deeplabv3_resnet50",
    "build_deeplabv3_resnet101",
    "build_mask2former",
    "build_segformer",
]
