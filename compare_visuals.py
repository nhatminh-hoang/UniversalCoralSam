"""Aggregate predicted masks and encoder heatmaps across multiple evaluation runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


PRED_SUFFIX = "_pred_mask.png"
HEATMAP_SUFFIX = "_heatmap.png"
ROW_SPACING = 14
COLUMN_SPACING = 18
PADDING = 18
ROW_LABEL_GAP = 12


@dataclass
class SampleVisuals:
    predicted_mask: Optional[Path] = None
    heatmap: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create comparison sheets for predicted masks and encoder heatmaps across models/checkpoints."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="[NAME=]PATH",
        help="Evaluation output directory, optionally prefixed with a run name (e.g. segformer=outputs/segformer_b1).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where combined comparison images will be written.",
    )
    parser.add_argument(
        "--legend",
        default=None,
        help="Optional path to a legend PNG. Defaults to the first legend.png discovered in the provided runs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples combined (after filtering).",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Restrict comparisons to specific sample identifiers (matching the file stem used in evaluation outputs).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples where any run is missing either the predicted mask or heatmap instead of showing placeholders.",
    )
    return parser.parse_args()


def _parse_run_spec(entry: str) -> Tuple[str, Path]:
    if "=" in entry:
        name, path_text = entry.split("=", 1)
        name = name.strip()
        path = Path(path_text.strip())
    else:
        path = Path(entry.strip())
        name = path.name
    return name, path.expanduser().resolve()


def _collect_sample_visuals(run_dir: Path) -> Dict[str, SampleVisuals]:
    visuals: Dict[str, SampleVisuals] = {}

    for pred_file in run_dir.glob(f"*{PRED_SUFFIX}"):
        sample_id = pred_file.name[: -len(PRED_SUFFIX)]
        entry = visuals.setdefault(sample_id, SampleVisuals())
        entry.predicted_mask = pred_file

    for heatmap_file in run_dir.glob(f"*{HEATMAP_SUFFIX}"):
        sample_id = heatmap_file.name[: -len(HEATMAP_SUFFIX)]
        entry = visuals.setdefault(sample_id, SampleVisuals())
        entry.heatmap = heatmap_file

    return visuals


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _resolve_legend(explicit: Optional[str], runs: Sequence[Tuple[str, Path, Dict[str, SampleVisuals]]]) -> Optional[Image.Image]:
    if explicit:
        legend_path = Path(explicit).expanduser().resolve()
        if not legend_path.exists():
            raise FileNotFoundError(f"Legend file '{legend_path}' does not exist.")
        return _load_image(legend_path)

    for _, run_path, _ in runs:
        candidate = run_path / "legend.png"
        if candidate.exists():
            return _load_image(candidate)
    return None


def _font_metrics(font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = font.getbbox("Ag")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _render_sample(
    sample_id: str,
    columns: Sequence[Tuple[str, SampleVisuals]],
    legend: Optional[Image.Image],
    *,
    skip_missing: bool,
) -> Optional[Image.Image]:
    font = ImageFont.load_default()
    _, font_height = _font_metrics(font)

    pred_images: List[Optional[Image.Image]] = []
    heat_images: List[Optional[Image.Image]] = []

    for _, visuals in columns:
        pred_img = _load_image(visuals.predicted_mask) if visuals.predicted_mask else None
        heat_img = _load_image(visuals.heatmap) if visuals.heatmap else None
        pred_images.append(pred_img)
        heat_images.append(heat_img)

    if skip_missing and any(img is None or h_img is None for img, h_img in zip(pred_images, heat_images)):
        return None

    pred_row_height = max((img.height for img in pred_images if img), default=0)
    heat_row_height = max((img.height for img in heat_images if img), default=0)
    if pred_row_height == 0 and heat_row_height == 0:
        return None

    col_width = max(
        (
            max(img.width if img else 0, heat.width if heat else 0)
            for img, heat in zip(pred_images, heat_images)
        ),
        default=0,
    )
    if col_width == 0:
        return None

    row_specs: List[Tuple[str, List[Optional[Image.Image]], int]] = []
    if pred_row_height > 0:
        row_specs.append(("Predicted Mask", pred_images, pred_row_height))
    if heat_row_height > 0:
        row_specs.append(("Encoder Heatmap", heat_images, heat_row_height))

    row_label_width = 0
    if row_specs:
        row_label_width = max(font.getbbox(label)[2] - font.getbbox(label)[0] for label, _, _ in row_specs)
        row_label_width += ROW_LABEL_GAP

    content_width = len(columns) * col_width + max(len(columns) - 1, 0) * COLUMN_SPACING
    canvas_width = PADDING * 2 + row_label_width + content_width

    sample_text_width = font.getbbox(sample_id)[2] - font.getbbox(sample_id)[0]
    run_label_height = font_height

    total_height = PADDING + font_height  # sample id
    total_height += ROW_SPACING + run_label_height

    for _, _, row_height in row_specs:
        total_height += ROW_SPACING + row_height

    if legend is not None:
        total_height += ROW_SPACING + legend.height

    total_height += PADDING

    canvas = Image.new("RGB", (canvas_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    sample_x = (canvas_width - sample_text_width) // 2
    sample_y = PADDING
    draw.text((sample_x, sample_y), sample_id, fill=(0, 0, 0), font=font)

    cursor_y = sample_y + font_height + ROW_SPACING
    for column_idx, (run_name, _) in enumerate(columns):
        text_width = font.getbbox(run_name)[2] - font.getbbox(run_name)[0]
        col_x = PADDING + row_label_width + column_idx * (col_width + COLUMN_SPACING)
        run_x = col_x + (col_width - text_width) / 2
        draw.text((run_x, cursor_y), run_name, fill=(0, 0, 0), font=font)

    cursor_y += run_label_height

    for row_idx, (label, images, row_height) in enumerate(row_specs):
        cursor_y += ROW_SPACING
        row_top = cursor_y
        if row_label_width > 0:
            label_width = font.getbbox(label)[2] - font.getbbox(label)[0]
            label_x = PADDING + row_label_width - ROW_LABEL_GAP - label_width
            label_y = row_top + (row_height - font_height) / 2
            draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

        for column_idx, image in enumerate(images):
            col_x = PADDING + row_label_width + column_idx * (col_width + COLUMN_SPACING)
            box_top = int(row_top)
            box_left = int(col_x)

            if image is None:
                placeholder = Image.new("RGB", (col_width, row_height), color=(230, 230, 230))
                canvas.paste(placeholder, (box_left, box_top))
                draw.rectangle(
                    [
                        box_left,
                        box_top,
                        box_left + col_width - 1,
                        box_top + row_height - 1,
                    ],
                    outline=(160, 160, 160),
                    width=1,
                )
                missing_text = "N/A"
                text_width = font.getbbox(missing_text)[2] - font.getbbox(missing_text)[0]
                text_height = font.getbbox(missing_text)[3] - font.getbbox(missing_text)[1]
                draw.text(
                    (
                        box_left + (col_width - text_width) / 2,
                        box_top + (row_height - text_height) / 2,
                    ),
                    missing_text,
                    fill=(80, 80, 80),
                    font=font,
                )
            else:
                offset_x = box_left + (col_width - image.width) // 2
                offset_y = box_top + (row_height - image.height) // 2
                canvas.paste(image, (offset_x, offset_y))

        cursor_y = row_top + row_height

    if legend is not None:
        cursor_y += ROW_SPACING
        legend_x = (canvas_width - legend.width) // 2
        legend_y = int(cursor_y)
        canvas.paste(legend, (legend_x, legend_y))

    return canvas


def main() -> None:
    args = parse_args()
    runs: List[Tuple[str, Path, Dict[str, SampleVisuals]]] = []

    for spec in args.run:
        name, run_path = _parse_run_spec(spec)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory '{run_path}' does not exist.")
        visuals = _collect_sample_visuals(run_path)
        runs.append((name, run_path, visuals))

    if not runs:
        raise RuntimeError("At least one run directory must be provided via --run.")

    all_sample_ids = sorted({sample_id for _, _, items in runs for sample_id in items})

    if args.sample:
        requested = set(args.sample)
        filtered = [sample_id for sample_id in all_sample_ids if sample_id in requested]
        missing = requested.difference(filtered)
        if missing:
            print(f"[Compare] Warning: samples not found in any run: {sorted(missing)}")
        all_sample_ids = filtered

    if args.max_samples is not None:
        all_sample_ids = all_sample_ids[: args.max_samples]

    legend_image = _resolve_legend(args.legend, runs)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for sample_id in all_sample_ids:
        column_specs = []
        for run_name, _run_path, visuals in runs:
            column_specs.append((run_name, visuals.get(sample_id, SampleVisuals())))
        rendered = _render_sample(sample_id, column_specs, legend_image, skip_missing=args.skip_missing)
        if rendered is None:
            continue
        output_path = output_dir / f"{sample_id}_comparison.png"
        rendered.save(output_path)
        generated += 1

    print(f"[Compare] Wrote {generated} comparison image(s) to {output_dir}.")


if __name__ == "__main__":
    main()
