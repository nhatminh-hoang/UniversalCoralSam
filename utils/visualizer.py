"""Simple segmentation visualizer built on top of Pillow and NumPy.

Example
-------
>>> import numpy as np
>>> from utils.visualizer import SegmentationVisualizer
>>> image = np.zeros((4, 4, 3), dtype=np.uint8)
>>> mask = np.array([[0, 0, 1, 1]] * 4, dtype=np.int64)
>>> viz = SegmentationVisualizer(class_colors={0: (0, 0, 0), 1: (255, 0, 0)})
>>> overlay = viz.overlay_mask(image, mask)
>>> overlay.shape
(4, 4, 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

Color = Tuple[int, int, int]


@dataclass
class SegmentationVisualizer:
    """Utility for overlaying segmentation masks on RGB images."""

    class_colors: Dict[int, Color] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.class_colors:
            self.class_colors = {0: (0, 0, 0), 1: (0, 255, 0)}

    def _to_rgb(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return np.stack([data] * 3, axis=-1)
        if data.ndim == 3 and data.shape[-1] == 3:
            return data
        raise ValueError("Input must be grayscale or RGB array.")

    def palette(self, num_classes: int) -> Iterable[Color]:
        for idx in range(num_classes):
            yield self.class_colors.get(idx, (idx * 50 % 255, 128, 255 - idx * 50 % 255))

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.int32)
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in self.class_colors.items():
            color_mask[mask == class_idx] = color
        return color_mask

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        rgb_image = self._to_rgb(image.astype(np.uint8))
        color_mask = self.colorize_mask(mask)
        return (alpha * color_mask + (1 - alpha) * rgb_image).astype(np.uint8)

    def save_overlay(self, image: np.ndarray, mask: np.ndarray, out_path: str, alpha: float = 0.6) -> None:
        overlay = self.overlay_mask(image, mask, alpha=alpha)
        Image.fromarray(overlay).save(out_path)


if __name__ == "__main__":
    import numpy as np

    demo_image = np.zeros((8, 8, 3), dtype=np.uint8)
    demo_image[:, 4:] = 255
    demo_mask = np.array([[0] * 4 + [1] * 4] * 8, dtype=np.int64)
    visualizer = SegmentationVisualizer(class_colors={0: (0, 0, 255), 1: (255, 0, 0)})
    demo_overlay = visualizer.overlay_mask(demo_image, demo_mask)
    Image.fromarray(demo_overlay).save("visualizer_demo.png")
