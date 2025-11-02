"""Evaluate checkpoints and generate comparative visuals across training scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from dataset import DATASET_BUILDERS, create_dataloader
from models import get_model, resolve_model_name
from training import evaluate as evaluate_loop
from training import predict
from utils import FeatureHeatmapGenerator, SegmentationVisualizer, center_crop, tensor_to_image


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


def _parse_dataset_args(pairs: Sequence[str]) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Dataset argument '{item}' must be in key=value format.")
        key, value = item.split("=", 1)
        kwargs[key.replace("-", "_")] = _coerce_scalar(value)
    return kwargs


def _collect_checkpoints(paths: Sequence[str], directory: str | None, pattern: str) -> List[Path]:
    collected: List[Path] = []
    collected.extend(Path(p) for p in paths)
    if directory is not None:
        collected.extend(sorted(Path(directory).glob(pattern)))
    if not collected:
        raise FileNotFoundError("No checkpoints provided. Use --checkpoint or --checkpoint-dir.")
    unique = []
    seen = set()
    for path in collected:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _resolve_layer(model: torch.nn.Module, layer_path: str | None) -> torch.nn.Module:
    if layer_path is None:
        if hasattr(model, "encoder1"):
            layer_path = "encoder1.0"
        else:
            raise ValueError("heatmap layer must be specified via --heatmap-layer for this model.")

    module: torch.nn.Module = model
    for segment in layer_path.split("."):
        if segment.isdigit():
            module = module[int(segment)]  # type: ignore[index]
        else:
            module = getattr(module, segment)
    return module


def _heatmap_to_rgb(heatmap: torch.Tensor) -> np.ndarray:
    array = heatmap.squeeze().detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    colored = cm.get_cmap("magma")(array)[..., :3]
    return (colored * 255.0).astype(np.uint8)


def _prepare_panels(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    heatmap: np.ndarray,
    crop_size: int,
    visualizer: SegmentationVisualizer,
) -> Image.Image:
    h, w = image.shape[:2]
    size = min(crop_size, h, w)
    image_crop = center_crop(image, size)
    gt_crop = center_crop(gt_mask, size)
    pred_crop = center_crop(pred_mask, size)
    heatmap_crop = center_crop(heatmap, size)

    gt_overlay = visualizer.overlay_mask(image_crop, gt_crop)
    pred_overlay = visualizer.overlay_mask(image_crop, pred_crop)

    panels = [
        image_crop,
        gt_overlay,
        pred_overlay,
        heatmap_crop,
    ]

    canvas = Image.new("RGB", (size * 2, size * 2))
    for idx, panel in enumerate(panels):
        row, col = divmod(idx, 2)
        canvas.paste(Image.fromarray(panel), (col * size, row * size))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and generate heatmap comparisons.")
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
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Path to a checkpoint to evaluate. May be specified multiple times.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing checkpoints to evaluate (recursively matched using --checkpoint-pattern).",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="*.pth",
        help="Glob pattern for checkpoints inside --checkpoint-dir.",
    )
    parser.add_argument("--model", required=True, help="Model identifier used during training.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of segmentation classes.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation and inference.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--ignore-index", type=int, default=None, help="Optional ignore index for CrossEntropyLoss.")
    parser.add_argument("--curve-dir", default=None, help="Optional directory to store metric snapshots.")
    parser.add_argument("--metrics-out", default=None, help="Path to write aggregate metrics as JSON.")
    parser.add_argument("--output-dir", default="evaluation_outputs", help="Directory for visualizations.")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to visualize per checkpoint.")
    parser.add_argument("--crop-size", type=int, default=512, help="Square crop size for comparison panels.")
    parser.add_argument("--heatmap-layer", default=None, help="Dotted path to the layer used for feature heatmaps.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    dataset_kwargs = _parse_dataset_args(args.dataset_arg)
    dataloader = create_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataset_kwargs,
    )

    checkpoints = _collect_checkpoints(args.checkpoint, args.checkpoint_dir, args.checkpoint_pattern)
    visualizer = SegmentationVisualizer()

    loss_kwargs = {}
    if args.ignore_index is not None:
        loss_kwargs["ignore_index"] = args.ignore_index
    loss_fn = torch.nn.CrossEntropyLoss(**loss_kwargs)

    results: List[Dict[str, object]] = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint in checkpoints:
        resolved_model_name = resolve_model_name(args.model)
        model = get_model(resolved_model_name, num_classes=args.num_classes)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        metrics = evaluate_loop(
            model,
            dataloader,
            loss_fn,
            device=str(device),
            amp_dtype=torch.bfloat16 if device.type in {"cuda", "cpu"} else None,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        results.append({"checkpoint": str(checkpoint), "metrics": metrics})
        print(f"[Evaluation] {checkpoint.name}: {metrics}")

        target_layer = _resolve_layer(model, args.heatmap_layer)
        heatmap_generator = FeatureHeatmapGenerator(model, target_layer=target_layer)

        sample_dir = output_dir / checkpoint.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        for idx, output in enumerate(predict(model, dataloader, device=str(device))):
            if idx >= args.num_samples:
                break

            image = tensor_to_image(output["image"])
            pred_mask = output["pred"].squeeze(0).numpy().astype(np.int64)
            mask_tensor = output.get("mask")
            if mask_tensor is None:
                continue
            gt_mask = mask_tensor.numpy().astype(np.int64)

            with torch.no_grad():
                sample_input = output["image"].unsqueeze(0).to(device)
                heatmap_tensor = heatmap_generator.generate(sample_input)
            heatmap_rgb = _heatmap_to_rgb(heatmap_tensor)

            panel = _prepare_panels(image, gt_mask, pred_mask, heatmap_rgb, args.crop_size, visualizer)
            identifier = output["id"] or f"sample_{idx:04d}"
            panel.save(sample_dir / f"{Path(str(identifier)).stem}_comparison.png")

        heatmap_generator.close()

    metrics_path = Path(args.metrics_out) if args.metrics_out else output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[Evaluation] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
