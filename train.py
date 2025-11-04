"""Train a segmentation model on any registered dataset.

Example
-------
python train.py --dataset coral --dataset-arg image_dir=data/images --dataset-arg mask_dir=data/masks --num-classes 5
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch

from dataset import DATASET_BUILDERS, create_dataloader
from models import available_models, get_model
from training import fit
from utils import configure_huggingface_environment
from utils.curves import TrainingCurveWriter


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
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
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
        help="Additional keyword arguments forwarded to the dataset builder.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of segmentation classes.")
    parser.add_argument(
        "--model",
        default="baseline_small_unet",
        help=f"Model identifier. Available: {available_models()}",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=None,
        help="Optional ignore index for CrossEntropyLoss.",
    )
    parser.add_argument("--curve-path", default=None, help="Optional path to save training curves.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available.")
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
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    cache_path = configure_huggingface_environment(args.hf_cache_dir, offline=args.hf_offline)
    if cache_path is not None:
        print(f"[train] Using Hugging Face cache directory: {cache_path}")
    if args.hf_offline is not None:
        state = "enabled" if args.hf_offline else "disabled"
        print(f"[train] Hugging Face offline mode {state}.")

    dataset_kwargs = _parse_dataset_args(args.dataset_arg)
    try:
        dataloader = create_dataloader(
            args.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            **dataset_kwargs,
        )
    except TypeError as exc:
        raise SystemExit(f"Failed to construct dataloader for dataset '{args.dataset}': {exc}") from exc

    model = get_model(args.model, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_kwargs = {}
    if args.ignore_index is not None:
        loss_kwargs["ignore_index"] = args.ignore_index
    loss_fn = torch.nn.CrossEntropyLoss(**loss_kwargs)
    curve_writer = TrainingCurveWriter(args.curve_path) if args.curve_path else None
    metrics = fit(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=args.epochs,
        curve_writer=curve_writer,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )
    print(f"Finished training. Final metrics: {metrics}")


if __name__ == "__main__":
    main()
