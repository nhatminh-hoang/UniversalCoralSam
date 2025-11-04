"""Evaluate checkpoints and generate comparative visuals across training scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from dataset import DATASET_BUILDERS, create_dataloader
from models import get_model, resolve_model_name
from training import evaluate as evaluate_loop
from training import predict
from utils import FeatureHeatmapGenerator, SegmentationVisualizer, configure_huggingface_environment, tensor_to_image


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


def _parse_class_names(entries: Sequence[str], num_classes: int) -> Dict[int, str]:
    names: Dict[int, str] = {idx: f"class_{idx}" for idx in range(num_classes)}
    next_auto = 0
    for entry in entries:
        if "=" in entry:
            idx_text, label = entry.split("=", 1)
            try:
                idx = int(idx_text)
            except ValueError:
                continue
        else:
            label = entry
            while next_auto < num_classes and names[next_auto] != f"class_{next_auto}":
                next_auto += 1
            if next_auto >= num_classes:
                continue
            idx = next_auto
            next_auto += 1
        if not label:
            continue
        names[idx] = label
    return names

def _resolve_attribute_path(obj: object, path: str) -> object | None:
    segments = [segment for segment in path.split(".") if segment]
    current = obj
    for segment in segments:
        index: Optional[int] = None
        attr_name = segment
        if "[" in segment and segment.endswith("]"):
            attr_name, index_text = segment[:-1].split("[", 1)
            try:
                index = int(index_text)
            except ValueError:
                return None
        if not hasattr(current, attr_name):
            return None
        current = getattr(current, attr_name)
        if index is not None:
            if isinstance(current, (list, tuple)):
                current_len = len(current)
                if not (-current_len <= index < current_len):
                    return None
                current = current[index]
            elif isinstance(current, (torch.nn.ModuleList, torch.nn.Sequential)):
                current_len = len(current)
                if not (-current_len <= index < current_len):
                    return None
                current = current[index]
            else:
                try:
                    current = current[index]
                except (IndexError, KeyError, TypeError):
                    return None
    return current


def _resolve_layer(model: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    resolved = _resolve_attribute_path(model, layer_path)
    if isinstance(resolved, torch.nn.Module):
        return resolved

    raise ValueError(f"Failed to resolve heatmap layer '{layer_path}'.")


def _is_transformer_identifier(model_name: str) -> bool:
    lower = model_name.lower()
    return lower.startswith("segformer") or lower.startswith("mask2former")


def _select_heatmap_layer(
    model: torch.nn.Module,
    model_name: str,
    override_path: str | None,
) -> torch.nn.Module | None:
    if override_path:
        return _resolve_layer(model, override_path)
    if _is_transformer_identifier(model_name):
        return None  # Delegate to transformer-aware strategies.
    return None


def _heatmap_to_rgb(heatmap: torch.Tensor) -> np.ndarray:
    array = heatmap.squeeze().detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    colored = matplotlib.colormaps["magma"](array)[..., :3]
    return (colored * 255.0).astype(np.uint8)


def _compose_comparison_panels(
    image: np.ndarray,
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    legend: Image.Image | None,
) -> Image.Image:
    height, width = image.shape[:2]
    panels = [
        ("Input", image),
        ("Ground Truth", gt_rgb),
        ("Prediction", pred_rgb),
        ("Encoder Heatmap", heatmap_rgb),
    ]

    grid = Image.new("RGB", (width * 2, height * 2))
    font = ImageFont.load_default()

    for idx, (title, panel_array) in enumerate(panels):
        panel_image = Image.fromarray(panel_array)
        row, col = divmod(idx, 2)
        x_offset = col * width
        y_offset = row * height
        grid.paste(panel_image, (x_offset, y_offset))
        draw = ImageDraw.Draw(grid)
        bbox = font.getbbox(title)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = 4
        rect = [
            x_offset + padding,
            y_offset + padding,
            x_offset + padding + text_width + padding * 2,
            y_offset + padding + text_height + padding * 2,
        ]
        draw.rectangle(rect, fill=(255, 255, 255))
        draw.text(
            (x_offset + padding * 2, y_offset + padding * 2),
            title,
            fill=(0, 0, 0),
            font=font,
        )

    if legend is None:
        return grid

    spacer = 16
    final_width = max(grid.width, legend.width)
    final_height = grid.height + spacer + legend.height
    canvas = Image.new("RGB", (final_width, final_height), color=(255, 255, 255))
    grid_x = (final_width - grid.width) // 2
    canvas.paste(grid, (grid_x, 0))
    legend_x = (final_width - legend.width) // 2
    canvas.paste(legend, (legend_x, grid.height + spacer))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and generate heatmap comparisons.")
    parser.set_defaults(exclude_label0_metrics=True)
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
    parser.add_argument(
        "--class-name",
        action="append",
        default=[],
        metavar="[IDX=]NAME",
        help="Override class display names. Provide 'index=name' or list names in order (may repeat).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available.")
    parser.add_argument(
        "--exclude-label0-metrics",
        dest="exclude_label0_metrics",
        action="store_true",
        help="Exclude label 0 when computing accuracy and mIoU metrics (default).",
    )
    parser.add_argument(
        "--include-label0-metrics",
        dest="exclude_label0_metrics",
        action="store_false",
        help="Include label 0 when computing accuracy and mIoU metrics.",
    )
    parser.add_argument("--hf-cache-dir", default=None, help="Optional directory to use as Hugging Face cache.")
    parser.set_defaults(hf_offline=None)
    parser.add_argument(
        "--hf-offline",
        dest="hf_offline",
        action="store_true",
        help="Enable Hugging Face offline mode (sets HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, etc.).",
    )
    parser.add_argument(
        "--hf-online",
        dest="hf_offline",
        action="store_false",
        help="Disable Hugging Face offline mode (removes related environment flags).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    cache_path = configure_huggingface_environment(args.hf_cache_dir, offline=args.hf_offline)
    if cache_path is not None:
        print(f"[evaluate] Using Hugging Face cache directory: {cache_path}")
    if args.hf_offline is not None:
        state = "enabled" if args.hf_offline else "disabled"
        print(f"[evaluate] Hugging Face offline mode {state}.")

    dataset_kwargs = _parse_dataset_args(args.dataset_arg)
    dataset_key = args.dataset.lower()
    denorm_stats: tuple[torch.Tensor, torch.Tensor] | None = None

    if dataset_key == "hkcoral":
        from dataset.hk_transforms import IMAGENET_MEAN, IMAGENET_STD, resize_to_target

        dataset_kwargs.setdefault("transform", resize_to_target)
        denorm_stats = (IMAGENET_MEAN, IMAGENET_STD)
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

    class_names_map = _parse_class_names(args.class_name, args.num_classes)
    if args.ignore_index is not None and args.ignore_index not in class_names_map:
        class_names_map[args.ignore_index] = "ignore"
    visualizer = SegmentationVisualizer(class_names=class_names_map, ignore_index=args.ignore_index)
    resolved_model_name = resolve_model_name(args.model)

    for checkpoint in checkpoints:
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
            ignore_labels=(0,) if args.exclude_label0_metrics else None,
        )
        results.append({"checkpoint": str(checkpoint), "metrics": metrics})
        print(f"[Evaluation] {checkpoint.name}: {metrics}")

        target_layer = _select_heatmap_layer(model, resolved_model_name, args.heatmap_layer)
        heatmap_generator = FeatureHeatmapGenerator(
            model,
            target_layer=target_layer,
            model_identifier=resolved_model_name,
        )

        sample_dir = output_dir / checkpoint.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        legend_image = visualizer.build_legend(
            args.num_classes,
            include_ignore=args.ignore_index is not None,
        )
        legend_path = sample_dir / "legend.png"
        legend_image.save(legend_path)

        for idx, output in enumerate(predict(model, dataloader, device=str(device))):
            if idx >= args.num_samples:
                break

            image_tensor = output["image"]
            if denorm_stats is not None:
                mean, std = denorm_stats
                mean = mean.to(device=image_tensor.device, dtype=image_tensor.dtype)
                std = std.to(device=image_tensor.device, dtype=image_tensor.dtype)
                image_tensor = image_tensor * std + mean
            image = tensor_to_image(image_tensor)
            pred_mask = np.asarray(output["pred"]).squeeze().astype(np.int64)
            if pred_mask.ndim != 2:
                raise ValueError(f"Expected 2D prediction mask, got shape {pred_mask.shape}.")
            mask_tensor = output.get("mask")
            if mask_tensor is None:
                continue
            gt_mask = np.asarray(mask_tensor).squeeze().astype(np.int64)
            if gt_mask.ndim != 2:
                raise ValueError(f"Expected 2D ground-truth mask, got shape {gt_mask.shape}.")

            with torch.no_grad():
                sample_input = output["image"].unsqueeze(0).to(device)
                heatmap_tensor = heatmap_generator.generate(sample_input)
                heatmap_tensor = F.interpolate(
                    heatmap_tensor,
                    size=pred_mask.shape,
                    mode="bilinear",
                    align_corners=False,
                )
            heatmap_rgb = _heatmap_to_rgb(heatmap_tensor)

            identifier = output["id"] or f"sample_{idx:04d}"
            stem = Path(str(identifier)).stem
            pred_rgb = visualizer.colorize_mask(pred_mask)
            gt_rgb = visualizer.colorize_mask(gt_mask)

            target_size = (image.shape[1], image.shape[0])
            if (pred_rgb.shape[0], pred_rgb.shape[1]) != image.shape[:2]:
                pred_rgb = np.array(
                    Image.fromarray(pred_rgb).resize(target_size, Image.NEAREST),
                    dtype=np.uint8,
                )
            if (gt_rgb.shape[0], gt_rgb.shape[1]) != image.shape[:2]:
                gt_rgb = np.array(
                    Image.fromarray(gt_rgb).resize(target_size, Image.NEAREST),
                    dtype=np.uint8,
                )
            if (heatmap_rgb.shape[0], heatmap_rgb.shape[1]) != image.shape[:2]:
                heatmap_rgb = np.array(
                    Image.fromarray(heatmap_rgb).resize(target_size, Image.BILINEAR),
                    dtype=np.uint8,
                )

            combined = _compose_comparison_panels(image, gt_rgb, pred_rgb, heatmap_rgb, legend_image)

            Image.fromarray(pred_rgb).save(sample_dir / f"{stem}_pred_mask.png")
            Image.fromarray(heatmap_rgb).save(sample_dir / f"{stem}_heatmap.png")
            combined.save(sample_dir / f"{stem}_comparison.png")

        heatmap_generator.close()

    metrics_path = Path(args.metrics_out) if args.metrics_out else output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[Evaluation] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
