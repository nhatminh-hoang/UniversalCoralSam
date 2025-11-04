#!/usr/bin/env python3
"""Visualize DINO CLS attention maps as heatmap overlays for input images.

When ``--dataset-root`` is supplied the script mirrors the evaluation pipeline by
grabbing the first ``--num-samples`` HKCoral test images (or the IDs passed via
``--sample-id``), ensuring heatmaps line up with the standard visualization set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")  # Ensure headless environments work without display servers.
from matplotlib import cm  # noqa: E402

from models.dino import build_dinov2, build_dinov3
from utils import configure_huggingface_environment
from utils.heatmaps import FeatureHeatmapGenerator
from dataset.hk_coral_dataset import HKCoralDataset
from dataset.hk_transforms import TARGET_SIZE

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DINO CLS attention heatmaps for images.")
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Path(s) to input image files (PNG/JPG/etc.).",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional HKCoral dataset root to pull evaluation samples from.",
    )
    parser.add_argument(
        "--dataset-split",
        default="test",
        help="Split to use when --dataset-root is supplied (default: test).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to visualize when pulling from the dataset.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Explicit sample IDs (without extension) to visualize when using --dataset-root.",
    )
    parser.add_argument(
        "--model-family",
        choices=("dinov2", "dinov3"),
        default="dinov2",
        help="Which DINO family to load.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Specific pre-trained variant (e.g. dinov2_base, dinov3_vits16_lvd1689m). Defaults per family.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run on (e.g. cuda, cuda:1, cpu). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/dino_attention",
        help="Directory where heatmaps and overlays will be saved.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay strength for blending heatmap onto the original image.",
    )
    parser.add_argument(
        "--match-eval-resolution",
        action="store_true",
        help="Resize images to the HKCoral evaluation resolution (512x1024) before processing.",
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
    return parser.parse_args(argv)


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(model_family: str, variant: str | None) -> tuple[torch.nn.Module, str]:
    if model_family == "dinov3":
        chosen_variant = variant or "dinov3_vits16_lvd1689m"
        model = build_dinov3(chosen_variant)
    else:
        chosen_variant = variant or "dinov2_base"
        model = build_dinov2(chosen_variant)
    return model, chosen_variant


def _to_colormap(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    colored = cm.get_cmap("magma")(array)[..., :3]
    return (colored * 255.0).astype(np.uint8)


def _sanitize_variant_name(text: str) -> str:
    return text.replace("/", "_").replace(":", "_")


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _resolve_dataset_images(
    root: str,
    split: str,
    sample_ids: List[str] | None,
    limit: int,
) -> List[Path]:
    dataset = HKCoralDataset(root=root, split=split, transform=None, ids=sample_ids)
    ids = dataset.ids if sample_ids is None else sample_ids
    selected_ids = ids[:limit] if sample_ids is None else ids
    image_paths = [dataset.image_dir / f"{sample_id}.jpg" for sample_id in selected_ids]
    return image_paths


def _resolve_image_paths(args: argparse.Namespace) -> List[Path]:
    if args.images:
        return [Path(path).expanduser().resolve() for path in args.images]
    if args.dataset_root:
        sample_ids = None
        if args.sample_id:
            sample_ids = [Path(sid).stem for sid in args.sample_id]
        return _resolve_dataset_images(args.dataset_root, args.dataset_split, sample_ids, args.num_samples)
    raise ValueError("Provide either --images or --dataset-root.")


def _compose_comparison(input_rgb: np.ndarray, heatmap_rgb: np.ndarray, overlay_rgb: np.ndarray) -> Image.Image:
    panels = [
        ("Input", input_rgb),
        ("Heatmap", heatmap_rgb),
        ("Overlay", overlay_rgb),
    ]
    base_height, base_width = panels[0][1].shape[:2]
    font = ImageFont.load_default()

    processed = []
    for title, array in panels:
        if array.shape[:2] != (base_height, base_width):
            pil_img = Image.fromarray(array)
            pil_img = pil_img.resize((base_width, base_height), Image.BILINEAR)
            array = np.array(pil_img, dtype=np.uint8)
        processed.append((title, array))

    padding = 12
    text_height = font.getbbox("Input")[3] - font.getbbox("Input")[1]
    canvas_width = padding + len(processed) * base_width + (len(processed) - 1) * padding
    canvas_width += padding
    canvas_height = padding + text_height + padding + base_height + padding

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, (title, array) in enumerate(processed):
        x = padding + idx * (base_width + padding)
        y = padding + text_height + padding
        panel_image = Image.fromarray(array)
        canvas.paste(panel_image, (x, y))

        text_x = x + (base_width - (font.getbbox(title)[2] - font.getbbox(title)[0])) / 2
        text_y = padding
        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)

    return canvas


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else None)
    device = _select_device(args.device)

    cache_path = configure_huggingface_environment(args.hf_cache_dir, offline=args.hf_offline)
    if cache_path is not None:
        print(f"[demo] Using Hugging Face cache directory: {cache_path}")
    if args.hf_offline is not None:
        state = "enabled" if args.hf_offline else "disabled"
        print(f"[demo] Hugging Face offline mode {state}.")

    try:
        image_paths = _resolve_image_paths(args)
    except ValueError as exc:
        print(f"[demo] {exc}", file=sys.stderr)
        return 1

    match_eval_resolution = args.match_eval_resolution or bool(args.dataset_root)

    model, variant = _build_model(args.model_family, args.variant)
    model.to(device)
    model.eval()

    processor = getattr(model, "image_processor", None)
    if processor is None:
        raise RuntimeError("DINO backbone did not expose a HuggingFace image processor; cannot preprocess inputs.")

    heatmap_generator = FeatureHeatmapGenerator(model, model_identifier=variant)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_suffix = _sanitize_variant_name(variant)

    for image_path in image_paths:
        if not image_path.exists():
            print(f"[demo] Skipping missing image '{image_path}'.", file=sys.stderr)
            continue

        image = _load_image(image_path)
        if match_eval_resolution:
            target_size = (TARGET_SIZE[1], TARGET_SIZE[0])  # PIL expects (width, height)
            image = image.resize(target_size, Image.BILINEAR)
        original_np = np.array(image)
        processor_outputs = processor(images=image, return_tensors="pt")
        if "pixel_values" not in processor_outputs:
            raise RuntimeError("Image processor did not return 'pixel_values'.")

        pixel_values = processor_outputs["pixel_values"].to(device)
        with torch.no_grad():
            heatmap_tensor = heatmap_generator.generate(pixel_values)

        heatmap_tensor = heatmap_tensor.to(dtype=torch.float32, device="cpu")
        heatmap_tensor = F.interpolate(
            heatmap_tensor,
            size=(original_np.shape[0], original_np.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        heatmap_np = heatmap_tensor.squeeze().numpy()
        heatmap_rgb = _to_colormap(heatmap_np)

        alpha = float(np.clip(args.alpha, 0.0, 1.0))
        overlay = (alpha * heatmap_rgb + (1.0 - alpha) * original_np).astype(np.uint8)

        stem = image_path.stem
        heatmap_path = output_dir / f"{stem}_{variant_suffix}_heatmap.png"
        overlay_path = output_dir / f"{stem}_{variant_suffix}_overlay.png"
        comparison_path = output_dir / f"{stem}_{variant_suffix}_comparison.png"

        Image.fromarray(heatmap_rgb).save(heatmap_path)
        Image.fromarray(overlay).save(overlay_path)
        comparison_image = _compose_comparison(original_np, heatmap_rgb, overlay)
        comparison_image.save(comparison_path)

        print(f"[demo] Saved heatmap: {heatmap_path}")
        print(f"[demo] Saved overlay: {overlay_path}")
        print(f"[demo] Saved comparison: {comparison_path}")

    heatmap_generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
