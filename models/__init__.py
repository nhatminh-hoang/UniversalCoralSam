"""Model registry for UniversalCoralSam segmentation experiments.

Example
-------
>>> from models import get_model, available_models
>>> "baseline_small_unet" in available_models()
True
>>> model = get_model("baseline_small_unet", num_classes=2)
>>> model.__class__.__name__
'SmallUNet'
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterable, List

from torch import nn

from .baseline import (
    build_baseline_model,
    build_deeplabv3_resnet101,
    build_deeplabv3_resnet50,
    build_mask2former,
    build_segformer,
)
from .dino import build_dinov2, build_dinov3


MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "baseline_small_unet": build_baseline_model,
    "deeplabv3_resnet50": build_deeplabv3_resnet50,
    "deeplabv3_resnet101": build_deeplabv3_resnet101,
    "dinov2": build_dinov2,
    "dinov3": build_dinov3,
    "mask2former_swin_t": partial(build_mask2former, "swin_t"),
    "mask2former_swin_s": partial(build_mask2former, "swin_s"),
    "mask2former_swin_base": partial(build_mask2former, "mask2former_swin_base"),
    "mask2former_swin_large": partial(build_mask2former, "mask2former_swin_large"),
    "segformer_b0": partial(build_segformer, "b0"),
    "segformer_b1": partial(build_segformer, "b1"),
    "segformer_b2_cityscapes": partial(build_segformer, "segformer_b2_cityscapes"),
    "segformer_b5_cityscapes": partial(build_segformer, "segformer_b5_cityscapes"),
}

MODEL_ALIASES = {
    "baseline": "baseline_small_unet",
}


def resolve_model_name(name: str) -> str:
    key = name.lower()
    return MODEL_ALIASES.get(key, key)


def get_model(name: str, **kwargs) -> nn.Module:
    resolved = resolve_model_name(name)
    try:
        builder = MODEL_BUILDERS[resolved]
    except KeyError as exc:
        raise NotImplementedError(f"Unknown model '{name}'. Available: {sorted(MODEL_BUILDERS)}") from exc
    return builder(**kwargs)


def available_models() -> List[str]:
    return sorted(MODEL_BUILDERS)


def register_model(name: str, builder: Callable[..., nn.Module]) -> None:
    if name in MODEL_BUILDERS:
        raise ValueError(f"Model '{name}' is already registered.")
    MODEL_BUILDERS[name] = builder


def iter_registered_models() -> Iterable[str]:
    return MODEL_BUILDERS.keys()


__all__ = ["MODEL_BUILDERS", "available_models", "get_model", "iter_registered_models", "register_model", "resolve_model_name"]
