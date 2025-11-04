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
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Color = Tuple[int, int, int]

_DEFAULT_COLOR_CYCLE: List[Color] = [
    (0, 0, 0),
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]
_IGNORE_COLOR: Color = (64, 64, 64)


@dataclass
class SegmentationVisualizer:
    """Utility for overlaying segmentation masks on RGB images."""

    class_colors: Dict[int, Color] = field(default_factory=dict)
    class_names: Dict[int, str] | Sequence[str] | None = None
    ignore_index: int | None = None

    def __post_init__(self) -> None:
        if not self.class_colors:
            self.class_colors = {idx: color for idx, color in enumerate(_DEFAULT_COLOR_CYCLE[:2])}

    def _color_for_class(self, class_idx: int) -> Color:
        if self.ignore_index is not None and class_idx == self.ignore_index:
            return _IGNORE_COLOR
        if class_idx in self.class_colors:
            return self.class_colors[class_idx]
        cycle_len = len(_DEFAULT_COLOR_CYCLE)
        if cycle_len == 0:
            return (0, 0, 0)
        color = _DEFAULT_COLOR_CYCLE[class_idx % cycle_len]
        self.class_colors[class_idx] = color
        return color

    def _class_name(self, class_idx: int) -> str:
        if isinstance(self.class_names, dict):
            return self.class_names.get(class_idx, f"class_{class_idx}")
        if isinstance(self.class_names, Sequence):
            if 0 <= class_idx < len(self.class_names):
                return str(self.class_names[class_idx])
        if self.ignore_index is not None and class_idx == self.ignore_index:
            return "ignore"
        return f"class_{class_idx}"

    def _to_rgb(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return np.stack([data] * 3, axis=-1)
        if data.ndim == 3 and data.shape[-1] == 3:
            return data
        raise ValueError("Input must be grayscale or RGB array.")

    def palette(self, num_classes: int) -> Iterable[Color]:
        for idx in range(num_classes):
            yield self._color_for_class(idx)

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.int32)
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        unique_classes = np.unique(mask)
        for class_idx in unique_classes:
            color = self._color_for_class(int(class_idx))
            color_mask[mask == class_idx] = color
        return color_mask

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        rgb_image = self._to_rgb(image.astype(np.uint8))
        color_mask = self.colorize_mask(mask)
        return (alpha * color_mask + (1 - alpha) * rgb_image).astype(np.uint8)

    def save_overlay(self, image: np.ndarray, mask: np.ndarray, out_path: str, alpha: float = 0.6) -> None:
        overlay = self.overlay_mask(image, mask, alpha=alpha)
        Image.fromarray(overlay).save(out_path)

    def build_legend(
        self,
        num_classes: int,
        *,
        include_ignore: bool = False,
        tile_size: int = 18,
        spacing: int = 6,
        padding: int = 8,
    ) -> Image.Image:
        entries: List[int] = list(range(num_classes))
        if include_ignore and self.ignore_index is not None and self.ignore_index not in entries:
            entries.append(self.ignore_index)
        elif not include_ignore and self.ignore_index is not None:
            entries = [idx for idx in entries if idx != self.ignore_index]
        if not entries:
            raise ValueError("Legend requires at least one class entry.")

        font = ImageFont.load_default()
        text_heights: List[int] = []
        text_widths: List[int] = []
        labels: List[str] = []
        for idx in entries:
            label = self._class_name(idx)
            bbox = font.getbbox(label)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            labels.append(label)
            text_widths.append(text_width)
            text_heights.append(text_height)

        legend_width = padding * 2 + tile_size + spacing + max(text_widths)
        total_height = sum(max(tile_size, th) for th in text_heights)
        total_height += spacing * (len(entries) - 1)
        legend_height = padding * 2 + total_height

        legend = Image.new("RGB", (legend_width, legend_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(legend)

        y_offset = padding
        for idx, label, text_height in zip(entries, labels, text_heights):
            color = self._color_for_class(idx)
            box_top = y_offset
            box_bottom = y_offset + tile_size
            draw.rectangle(
                [padding, box_top, padding + tile_size, box_bottom],
                fill=color,
                outline=(0, 0, 0),
            )
            text_y = y_offset + max(0, (tile_size - text_height) // 2)
            draw.text(
                (padding + tile_size + spacing, text_y),
                label,
                fill=(0, 0, 0),
                font=font,
            )
            y_offset += max(tile_size, text_height) + spacing
        return legend


if __name__ == "__main__":
    import numpy as np

    demo_image = np.zeros((8, 8, 3), dtype=np.uint8)
    demo_image[:, 4:] = 255
    demo_mask = np.array([[0] * 4 + [1] * 4] * 8, dtype=np.int64)
    visualizer = SegmentationVisualizer(class_colors={0: (0, 0, 255), 1: (255, 0, 0)})
    demo_overlay = visualizer.overlay_mask(demo_image, demo_mask)
    Image.fromarray(demo_overlay).save("visualizer_demo.png")
