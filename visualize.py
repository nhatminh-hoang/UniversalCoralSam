"""Generate visual overlays from model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from dataset import DATASET_BUILDERS, create_dataloader
from models import get_model
from training import predict
from utils import SegmentationVisualizer, tensor_to_image


def _coerce_scalar(text: str):
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _parse_dataset_args(pairs: List[str]) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Dataset argument '{item}' must be in key=value format.")
        key, value = item.split("=", 1)
        kwargs[key.replace("-", "_")] = _coerce_scalar(value)
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions as RGB overlays.")
    parser.add_argument(
        "--dataset",
        default="coral",
        help=f"Dataset identifier. Available: {sorted(DATASET_BUILDERS)}",
    )
    parser.add_argument(
        "--dataset-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keyword arguments forwarded to the dataset builder.",
    )
    parser.add_argument("--weights", required=True, help="Path to model weights produced by torch.save.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of model output classes.")
    parser.add_argument("--model", default="baseline_small_unet", help="Model identifier to load.")
    parser.add_argument("--output-dir", default="overlays", help="Where to write visualization PNGs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model = get_model(args.model, num_classes=args.num_classes)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    dataset_kwargs = _parse_dataset_args(args.dataset_arg)
    dataloader = create_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataset_kwargs,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = SegmentationVisualizer()
    for index, output in enumerate(predict(model, dataloader, device=device)):
        image = tensor_to_image(output["image"])
        mask = output["pred"].squeeze(0).numpy().astype(np.int64)
        overlay = visualizer.overlay_mask(image, mask)
        identifier = output["id"] or f"sample_{index:04d}"
        Image.fromarray(overlay).save(output_dir / f"{Path(str(identifier)).stem}_overlay.png")


if __name__ == "__main__":
    main()
