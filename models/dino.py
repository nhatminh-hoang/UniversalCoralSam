"""Wrappers for loading DINOv2 and DINOv3 encoders from HuggingFace."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Type

import torch
from torch import nn

from ._huggingface import (
    can_use_bfloat16,
    maybe_raise_torch_upgrade,
    resolve_pretrained_identifier,
    snapshot_has_safetensors,
)

_TRANSFORMERS_AVAILABLE = True
try:  # pragma: no cover - optional dependency
    from transformers import AutoImageProcessor, Dinov2Model
except ImportError:  # pragma: no cover - optional dependency
    _TRANSFORMERS_AVAILABLE = False

_DINOV3_CLASS_AVAILABLE = False
if _TRANSFORMERS_AVAILABLE:
    try:  # pragma: no cover - optional dependency
        from transformers import Dinov3Model  # type: ignore

        _DINOV3_CLASS_AVAILABLE = True
    except (ImportError, AttributeError):  # pragma: no cover - optional dependency
        Dinov3Model = None  # type: ignore


@dataclass(frozen=True)
class DinoVariant:
    """Descriptor for a specific DINO checkpoint."""

    name: str
    repo_id: str


_DINOV2_PRETRAINED_VARIANTS: Dict[str, DinoVariant] = {
    "dinov2_small": DinoVariant("dinov2_small", "facebook/dinov2-small"),
    "dinov2_base": DinoVariant("dinov2_base", "facebook/dinov2-base"),
    "dinov2_large": DinoVariant("dinov2_large", "facebook/dinov2-large"),
    "dinov2_giant": DinoVariant("dinov2_giant", "facebook/dinov2-giant"),
}

_DINOV3_PRETRAINED_VARIANTS: Dict[str, DinoVariant] = {
    "dinov3_vits16_lvd1689m": DinoVariant("dinov3_vits16_lvd1689m", "facebook/dinov3-vits16-pretrain-lvd1689m"),
    "dinov3_vith16plus_lvd1689m": DinoVariant("dinov3_vith16plus_lvd1689m", "facebook/dinov3-vith16plus-pretrain-lvd1689m"),
    "dinov3_vit7b16_lvd1689m": DinoVariant("dinov3_vit7b16_lvd1689m", "facebook/dinov3-vit7b16-pretrain-lvd1689m"),
    "dinov3_vit7b16_sat493m": DinoVariant("dinov3_vit7b16_sat493m", "facebook/dinov3-vit7b16-pretrain-sat493m"),
}


class DinoBackbone(nn.Module):
    """Thin wrapper that exposes the raw transformer forward method."""

    requires_targets: bool = False

    def __init__(
        self,
        model: nn.Module,
        *,
        variant: str,
        image_processor: Optional["AutoImageProcessor"] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        self.model_description = variant

    def forward(
        self,
        pixel_values: torch.Tensor | dict[str, torch.Tensor],
        **kwargs,
    ):
        if isinstance(pixel_values, dict):
            kwargs = {**pixel_values, **kwargs}
            return self.model(**kwargs)
        return self.model(pixel_values=pixel_values, **kwargs)


def _build_common_load_kwargs(local_only: bool) -> Dict[str, object]:
    load_kwargs: Dict[str, object] = {"local_files_only": local_only}
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if can_use_bfloat16():
        load_kwargs["torch_dtype"] = torch.bfloat16
    return load_kwargs


def _maybe_enable_safetensors(load_kwargs: Dict[str, object], load_identifier: str, local_only: bool) -> None:
    use_safetensors = False
    if not local_only:
        use_safetensors = True
    elif snapshot_has_safetensors(load_identifier):
        use_safetensors = True
    if use_safetensors:
        load_kwargs["use_safetensors"] = True


def _load_image_processor(identifier: str, load_kwargs: Dict[str, object], local_only: bool):
    if not _TRANSFORMERS_AVAILABLE:
        return None
    processor_kwargs = {"local_files_only": local_only}
    if "cache_dir" in load_kwargs:
        processor_kwargs["cache_dir"] = load_kwargs["cache_dir"]
    try:
        return AutoImageProcessor.from_pretrained(identifier, **processor_kwargs)
    except Exception:
        return None


def _load_dino_model(
    variant: DinoVariant,
    model_cls: Type[nn.Module],
) -> DinoBackbone:
    identifier, local_only = resolve_pretrained_identifier(variant.repo_id)
    load_kwargs = _build_common_load_kwargs(local_only)
    _maybe_enable_safetensors(load_kwargs, identifier, local_only)

    try:
        model = model_cls.from_pretrained(identifier, **load_kwargs)
    except ValueError as exc:  # pragma: no cover - guard against torch<2.6 restrictions
        maybe_raise_torch_upgrade(exc, variant.repo_id)
        raise

    image_processor = _load_image_processor(identifier, load_kwargs, local_only)
    backbone = DinoBackbone(model, variant=variant.name, image_processor=image_processor)
    return backbone


def build_dinov2(variant: str = "dinov2_base") -> DinoBackbone:
    """Load a DINOv2 backbone."""

    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("DINO models require the 'transformers' package. Install via `pip install transformers scipy`.")

    key = variant.lower()
    if key not in _DINOV2_PRETRAINED_VARIANTS:
        raise KeyError(f"Unknown DINOv2 variant '{variant}'. Available: {sorted(_DINOV2_PRETRAINED_VARIANTS)}")
    return _load_dino_model(_DINOV2_PRETRAINED_VARIANTS[key], Dinov2Model)


def build_dinov3(variant: str = "dinov3_vits16_lvd1689m") -> DinoBackbone:
    """Load a DINOv3 backbone."""

    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("DINO models require the 'transformers' package. Install via `pip install transformers scipy`.")
    if not _DINOV3_CLASS_AVAILABLE:
        raise ImportError(
            "Your installed 'transformers' version does not provide `Dinov3Model`. "
            "Upgrade to a recent release (>=4.45) to load DINOv3 checkpoints."
        )

    key = variant.lower()
    if key not in _DINOV3_PRETRAINED_VARIANTS:
        raise KeyError(f"Unknown DINOv3 variant '{variant}'. Available: {sorted(_DINOV3_PRETRAINED_VARIANTS)}")
    return _load_dino_model(_DINOV3_PRETRAINED_VARIANTS[key], Dinov3Model)  # type: ignore[arg-type]


__all__ = ["DinoBackbone", "build_dinov2", "build_dinov3"]
