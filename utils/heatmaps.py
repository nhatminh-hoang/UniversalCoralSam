"""Produce feature heatmaps tailored to convolutional and transformer segmentation models.

This module generalizes the vanilla hook-based activation capture to support architectures
such as SegFormer and Mask2Former whose encoder features behave more like token grids.
For convolutional networks we retain the traditional forward-hook averaging, while for
transformer-based models we follow the attention-map style popularized by DINO by
aggregating the last encoder representations returned by the HuggingFace implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:  # pragma: no cover - optional dependency
    from models.dino import DinoBackbone  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DinoBackbone = None  # type: ignore

_TRANSFORMERS_AVAILABLE = True
try:  # pragma: no cover - optional dependency
    from transformers import Dinov2Model
except Exception:  # pragma: no cover - optional dependency
    Dinov2Model = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from transformers import Dinov3Model  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Dinov3Model = None  # type: ignore


class _ActivationExtractor:
    """Convert nested hook outputs to a tensor suitable for heatmap generation."""

    @staticmethod
    def extract(activation) -> Tensor:
        visited: Set[int] = set()

        def _inner(obj) -> Optional[Tensor]:
            if obj is None:
                return None
            obj_id = id(obj)
            if obj_id in visited:
                return None
            visited.add(obj_id)

            if torch.is_tensor(obj):
                return obj.detach()

            if isinstance(obj, dict):
                for key in sorted(obj.keys()):
                    tensor = _inner(obj[key])
                    if tensor is not None:
                        return tensor
                return None

            if isinstance(obj, (list, tuple)):
                for item in reversed(obj):
                    tensor = _inner(item)
                    if tensor is not None:
                        return tensor
                return None

            candidate_attrs = (
                "last_hidden_state",
                "hidden_states",
                "feature_maps",
                "encoder_last_hidden_state",
                "encoder_hidden_states",
                "pixel_decoder_last_hidden_state",
                "pixel_decoder_hidden_states",
                "masks_queries_logits",
                "mask_features",
            )
            for attr in candidate_attrs:
                if hasattr(obj, attr):
                    tensor = _inner(getattr(obj, attr))
                    if tensor is not None:
                        return tensor
            return None

        tensor = _inner(activation)
        if tensor is None:
            raise RuntimeError("Target layer hook did not yield a tensor output.")
        if tensor.ndim < 3:
            raise ValueError(f"Expected activation with spatial structure, got shape {tuple(tensor.shape)}.")
        return tensor


def _normalize_heatmap(activation: Tensor, target_size: Optional[Tuple[int, int]] = None) -> Tensor:
    """Average channels, normalize to [0, 1], and optionally upsample to the input size."""

    tensor = activation.to(dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise ValueError(f"Activation has unsupported ndim={tensor.ndim}; expected 3 or 4.")

    heatmap = tensor.mean(dim=1, keepdim=True)
    heatmap = heatmap - heatmap.amin(dim=(-2, -1), keepdim=True)
    denom = heatmap.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    heatmap = heatmap / denom

    if target_size is not None and tuple(target_size) != tuple(heatmap.shape[-2:]):
        heatmap = F.interpolate(heatmap, size=target_size, mode="bilinear", align_corners=False)
    return heatmap


def _infer_device_and_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
    device = torch.device("cpu")
    dtype = torch.float32

    for param in module.parameters():
        if torch.is_floating_point(param):
            return param.device, param.dtype
        device = param.device

    for buffer in module.buffers():
        if torch.is_floating_point(buffer):
            return buffer.device, buffer.dtype
        device = buffer.device

    return device, dtype


def _infer_token_grid(length: int, spatial_size: Tuple[int, int]) -> Tuple[int, int]:
    if length <= 0:
        raise ValueError("Token sequence length must be positive.")

    height_ref, width_ref = spatial_size
    if height_ref <= 0 or width_ref <= 0:
        size = int(round(length**0.5))
        if size * size != length:
            raise ValueError(f"Cannot infer spatial grid from token length {length}. Provide valid spatial_size.")
        return size, size

    best_hw: Optional[Tuple[int, int]] = None
    best_score = float("inf")

    limit = int(math.sqrt(length)) + 1
    for h in range(1, limit):
        if length % h != 0:
            continue
        w = length // h

        for cand_h, cand_w in ((h, w), (w, h)):
            if cand_h > height_ref or cand_w > width_ref:
                continue
            stride_h = height_ref / cand_h
            stride_w = width_ref / cand_w
            score = abs(stride_h - stride_w)
            if score < best_score:
                best_score = score
                best_hw = (cand_h, cand_w)

    if best_hw is None:
        size = int(round(length**0.5))
        if size * size != length:
            raise ValueError(
                f"Unable to determine spatial grid for token length {length} with reference size {spatial_size}."
            )
        return size, size
    return best_hw


def _reshape_tokens_to_grid(feature: Tensor, spatial_size: Tuple[int, int]) -> Tensor:
    if feature.ndim != 3:
        return feature
    batch, length, channels = feature.shape
    height, width = _infer_token_grid(length, spatial_size)
    return feature.permute(0, 2, 1).reshape(batch, channels, height, width)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Follow `.model` chains to find the underlying HuggingFace module."""

    current = model
    visited: Set[int] = set()

    while isinstance(current, nn.Module) and hasattr(current, "model"):
        model_attr = getattr(current, "model")
        if not isinstance(model_attr, nn.Module):
            break
        obj_id = id(model_attr)
        if obj_id in visited:
            break
        visited.add(obj_id)
        current = model_attr
    return current


class _HeatmapStrategy:
    def generate(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _HookedActivationStrategy(_HeatmapStrategy):
    """Default strategy that hooks a module and aggregates the captured activation."""

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.layer = layer
        self._activation: Optional[Tensor] = None
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = layer.register_forward_hook(self._store_activation)
        self.model.eval()
        self.device, self.dtype = _infer_device_and_dtype(self.model)

    def _store_activation(self, _module: nn.Module, _inputs: Tensor, output) -> None:
        self._activation = output

    def _prepare_inputs(self, inputs: Tensor) -> Tensor:
        if not torch.is_tensor(inputs):
            raise TypeError("Hooked heatmap strategy currently supports tensor inputs only.")
        return inputs.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def generate(self, inputs: Tensor) -> Tensor:
        self._activation = None
        pixel_values = self._prepare_inputs(inputs)
        with torch.no_grad():
            _ = self.model(pixel_values)
        if self._activation is None:
            raise RuntimeError("Hooked layer did not run; ensure inputs reach the target layer.")
        activation = _ActivationExtractor.extract(self._activation)
        return _normalize_heatmap(activation, target_size=tuple(inputs.shape[-2:]))

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class _DinoClsAttentionHeatmapStrategy(_HeatmapStrategy):
    """Project CLS self-attention over patch tokens into a spatial heatmap."""

    def __init__(self, wrapper: nn.Module) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "DINO attention visualization requires the 'transformers' package. Install via `pip install transformers`."
            )
        self.wrapper = wrapper
        self.model = _unwrap_model(wrapper)
        self.wrapper_model = getattr(wrapper, "model", None) if hasattr(wrapper, "model") else None
        self._image_processor = getattr(wrapper, "image_processor", None)
        self.model.eval()
        set_attn_impl = getattr(self.model, "set_attn_implementation", None)
        if callable(set_attn_impl):
            try:  # pragma: no cover - depends on transformers version
                set_attn_impl("eager")
            except Exception:
                pass
        self.device, self.dtype = _infer_device_and_dtype(self.model)
        config = getattr(self.model, "config", None)
        self._num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        self._has_cls_token = bool(getattr(config, "use_cls_token", True))

    def _prepare_inputs(self, inputs: Tensor) -> Tensor:
        if not torch.is_tensor(inputs):
            raise TypeError("DINO attention visualization expects tensor inputs.")
        if self._image_processor is not None and hasattr(self._image_processor, "image_mean"):
            # Assume caller already normalized; avoid double preprocessing.
            return inputs.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return inputs.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def _extract_cls_attention(self, attention: Tensor) -> Tensor:
        if attention.ndim != 4:
            raise ValueError(f"Expected attention map with shape [B, heads, tokens, tokens], got {tuple(attention.shape)}")
        if not self._has_cls_token:
            raise RuntimeError("DINO configuration indicates the CLS token is disabled; cannot compute attention map.")

        total_tokens = attention.shape[-1]
        patch_tokens = total_tokens - 1 - self._num_register_tokens
        if patch_tokens <= 0:
            raise RuntimeError(
                f"Invalid attention layout: total_tokens={total_tokens}, register_tokens={self._num_register_tokens}."
            )

        cls_to_tokens = attention[:, :, 0, 1 : 1 + patch_tokens]
        return cls_to_tokens

    def generate(self, inputs: Tensor) -> Tensor:
        pixel_values = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True,
            )

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            raise RuntimeError("DINO model did not return attention weights; ensure `output_attentions=True` is supported.")

        last_attention = attentions[-1]
        cls_to_patch = self._extract_cls_attention(last_attention)
        heatmap = cls_to_patch.mean(dim=1, keepdim=True)  # Average heads

        batch, _, num_tokens = heatmap.shape
        spatial_h, spatial_w = _infer_token_grid(num_tokens, spatial_size=pixel_values.shape[-2:])
        heatmap = heatmap.view(batch, 1, spatial_h, spatial_w)
        return _normalize_heatmap(heatmap, target_size=tuple(inputs.shape[-2:]))


class _SegformerHeatmapStrategy(_HeatmapStrategy):
    """Use SegFormer encoder hidden states (akin to DINO attention maps)."""

    def __init__(self, wrapper: nn.Module) -> None:
        self.wrapper = wrapper
        self.model = getattr(wrapper, "model", wrapper)
        self.model.eval()
        self.device, self.dtype = _infer_device_and_dtype(self.model)

    def _prepare_inputs(self, inputs: Tensor) -> Tensor:
        return inputs.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def generate(self, inputs: Tensor) -> Tensor:
        pixel_values = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                raise RuntimeError("SegFormer model did not expose hidden states for heatmap extraction.")
            feature = last_hidden
        else:
            feature = hidden_states[-1]

        feature = _reshape_tokens_to_grid(feature, spatial_size=pixel_values.shape[-2:])
        return _normalize_heatmap(feature, target_size=tuple(inputs.shape[-2:]))


class _Mask2FormerHeatmapStrategy(_HeatmapStrategy):
    """Aggregate pixel decoder feature maps from Mask2Former."""

    def __init__(self, wrapper: nn.Module) -> None:
        self.wrapper = wrapper
        self.model = getattr(wrapper, "model", wrapper)
        self.model.eval()
        self.device, self.dtype = _infer_device_and_dtype(self.model)

    def _prepare_inputs(self, inputs: Tensor) -> Tensor:
        return inputs.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def generate(self, inputs: Tensor) -> Tensor:
        pixel_values = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        feature = None

        pixel_decoder_states = getattr(outputs, "pixel_decoder_hidden_states", None)
        if pixel_decoder_states:
            feature = pixel_decoder_states[-1]

        if feature is None:
            encoder_hidden = getattr(outputs, "encoder_hidden_states", None)
            if encoder_hidden:
                feature = encoder_hidden[-1]

        if feature is None:
            raise RuntimeError("Mask2Former model did not return decoder or encoder hidden states.")

        feature = _reshape_tokens_to_grid(feature, spatial_size=pixel_values.shape[-2:])
        return _normalize_heatmap(feature, target_size=tuple(inputs.shape[-2:]))


def _is_segformer_model(model: nn.Module, model_identifier: Optional[str]) -> bool:
    name = model_identifier.lower() if model_identifier else model.__class__.__name__.lower()
    return "segformer" in name


def _is_mask2former_model(model: nn.Module, model_identifier: Optional[str]) -> bool:
    name = model_identifier.lower() if model_identifier else model.__class__.__name__.lower()
    return "mask2former" in name


def _is_dino_model(model: nn.Module, model_identifier: Optional[str]) -> bool:
    if model_identifier:
        ident_lower = model_identifier.lower()
        if any(token in ident_lower for token in ("dinov2", "dinov3", "dinov", "dino")):
            return True

    inner = _unwrap_model(model)

    if DinoBackbone is not None and isinstance(model, DinoBackbone):
        return True
    if DinoBackbone is not None and isinstance(inner, DinoBackbone):
        return True

    if Dinov2Model is not None and isinstance(model, Dinov2Model):
        return True
    if Dinov2Model is not None and isinstance(inner, Dinov2Model):
        return True

    if Dinov3Model is not None and isinstance(model, Dinov3Model):
        return True
    if Dinov3Model is not None and isinstance(inner, Dinov3Model):
        return True

    name = model.__class__.__name__.lower()
    if any(token in name for token in ("dinov2", "dinov3", "dinov", "dino")):
        return True

    inner_name = inner.__class__.__name__.lower()
    if any(token in inner_name for token in ("dinov2", "dinov3", "dinov", "dino")):
        return True

    return False


@dataclass
class FeatureHeatmapGenerator:
    """Capture activations and convert them to normalized heatmaps.

    Supports:
    - HuggingFace ViT backbones (e.g., DINOv2/v3) via CLS self-attention visualization.
    - SegFormer and Mask2Former via encoder/pixel-decoder feature aggregation.
    - Arbitrary modules when an explicit ``target_layer`` is supplied (standard forward hook).
    """

    model: nn.Module
    target_layer: Optional[nn.Module] = None
    model_identifier: Optional[str] = None
    _strategy: _HeatmapStrategy = field(init=False)

    def __post_init__(self) -> None:
        self._strategy = self._build_strategy()

    def _build_strategy(self) -> _HeatmapStrategy:
        if _is_dino_model(self.model, self.model_identifier):
            if self.target_layer is not None:
                return _HookedActivationStrategy(self.model, self.target_layer)
            return _DinoClsAttentionHeatmapStrategy(self.model)

        if _is_segformer_model(self.model, self.model_identifier):
            # Allow manual override to fall back to hook-based strategy.
            if self.target_layer is not None:
                return _HookedActivationStrategy(self.model, self.target_layer)
            return _SegformerHeatmapStrategy(self.model)

        if _is_mask2former_model(self.model, self.model_identifier):
            if self.target_layer is not None:
                return _HookedActivationStrategy(self.model, self.target_layer)
            return _Mask2FormerHeatmapStrategy(self.model)

        if self.target_layer is not None:
            return _HookedActivationStrategy(self.model, self.target_layer)

        raise ValueError(
            "Feature heatmaps are currently supported for DINO, SegFormer, and Mask2Former models. "
            "Provide `target_layer` explicitly to hook a custom module."
        )

    def generate(self, inputs: Tensor) -> Tensor:
        return self._strategy.generate(inputs)

    def close(self) -> None:
        self._strategy.close()

if __name__ == "__main__":
    demo_model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 2, kernel_size=1),
    )
    generator = FeatureHeatmapGenerator(demo_model, target_layer=demo_model[0])
    demo_input = torch.randn(1, 3, 16, 16)
    demo_heatmap = generator.generate(demo_input)
    print(demo_heatmap.mean().item())
    generator.close()
