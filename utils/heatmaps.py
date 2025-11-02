"""Produce coarse feature heatmaps from intermediate model activations.

Example
-------
>>> import torch
>>> from torch import nn
>>> from utils.heatmaps import FeatureHeatmapGenerator
>>> model = nn.Sequential(nn.Conv2d(3, 2, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(2, 2, kernel_size=1))
>>> generator = FeatureHeatmapGenerator(model, target_layer=model[0])
>>> inputs = torch.randn(1, 3, 8, 8)
>>> heatmap = generator.generate(inputs)
>>> heatmap.shape
torch.Size([1, 1, 8, 8])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class FeatureHeatmapGenerator:
    """Capture activations from a chosen layer and convert them to heatmaps."""

    model: nn.Module
    target_layer: nn.Module
    _activation: Optional[Tensor] = field(init=False, default=None)
    _hook: Optional[torch.utils.hooks.RemovableHandle] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.model.eval()
        self._hook = self.target_layer.register_forward_hook(self._store_activation)

    def _store_activation(self, _module: nn.Module, _inputs: Tensor, output: Tensor) -> None:
        self._activation = output.detach()

    def generate(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            _ = self.model(inputs)
        if self._activation is None:
            raise RuntimeError("Target layer hook did not run; ensure inputs reach the target layer.")
        heatmap = self._activation.mean(dim=1, keepdim=True)
        heatmap = heatmap - heatmap.amin(dim=(-2, -1), keepdim=True)
        denom = heatmap.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return heatmap / denom

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


if __name__ == "__main__":
    import torch
    from torch import nn

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
