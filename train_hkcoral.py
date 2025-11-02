"""Train requested segmentation models on the HKCoral dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
import torch.nn as nn

from dataset import build_hk_coral_dataloader
from models import get_model
from training import evaluate, train_one_epoch
from utils.curves import TrainingCurveWriter


HKCORAL_NUM_CLASSES = 7
MODEL_VARIANTS: Dict[str, List[str]] = {
    "deeplabv3": ["deeplabv3_resnet50", "deeplabv3_resnet101"],
    "mask2former": ["mask2former_swin_t", "mask2former_swin_s"],
    "segformer": ["segformer_b0", "segformer_b1"],
}

TARGET_SIZE = (512, 1024)


def resize_to_target(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize inputs to the configured target size, preserving discrete mask labels."""

    image = F.interpolate(image.unsqueeze(0), size=TARGET_SIZE, mode="bilinear", align_corners=False).squeeze(0)
    mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(), size=TARGET_SIZE, mode="nearest"
    ).squeeze(0).squeeze(0).long()
    return image, mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HKCoral segmentation models.")
    parser.add_argument(
        "--data-root",
        default="HKCoral",
        help="Path to the HKCoral dataset root containing 'images' and 'labels' directories.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deeplabv3", "mask2former", "segformer"],
        help="Model families to train. Choices: deeplabv3, mask2former, segformer, or concrete variant names.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per model variant.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for the optimizer.")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Epoch when the sinusoidal LR reaches its peak.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability applied to Dropout layers.")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker count for data loading.")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU even if CUDA is available.")
    parser.add_argument(
        "--curve-dir",
        default=None,
        help="If provided, save training curves under this directory (one file per model variant).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/hkcoral",
        help="Directory to store trained checkpoints. Checkpoints are saved as <variant>.pth.",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional path to store per-variant training and evaluation metrics as JSON.",
    )
    return parser.parse_args()


def resolve_variants(models: Iterable[str]) -> List[str]:
    variants: List[str] = []
    for name in models:
        lower = name.lower()
        if lower in MODEL_VARIANTS:
            variants.extend(MODEL_VARIANTS[lower])
        else:
            variants.append(name)
    return list(dict.fromkeys(variants))


def _apply_dropout_rate(model: nn.Module, dropout: float) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = dropout


def _sinusoidal_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(warmup_epochs, 1)
    total_epochs = max(total_epochs, 1)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if warmup_epochs >= total_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type == "cuda":
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        print("[HKCoral] Warning: CUDA bfloat16 not supported on this platform. Falling back to float32.")
        return None
    if device.type == "cpu":
        return torch.bfloat16
    return None


def train_variant(
    variant: str,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    curve_path: Path | None,
    output_dir: Path,
    dropout: float,
    warmup_epochs: int,
    amp_dtype: torch.dtype | None,
) -> Dict[str, object]:
    print(f"[HKCoral] Training variant '{variant}' on device {device}.")
    model = get_model(variant, num_classes=HKCORAL_NUM_CLASSES).to(device)
    _apply_dropout_rate(model, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _sinusoidal_scheduler(optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    curve_writer = TrainingCurveWriter(str(curve_path)) if curve_path is not None else None

    history: List[Dict[str, Dict[str, float]]] = []
    best_checkpoint_path = output_dir / f"{variant}_best.pth"
    best_miou = -1.0
    best_epoch = 0
    best_val_metrics: Dict[str, float] | None = None

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=str(device),
            amp_dtype=amp_dtype,
            num_classes=HKCORAL_NUM_CLASSES,
            ignore_index=255,
        )

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=str(device),
            amp_dtype=amp_dtype,
            num_classes=HKCORAL_NUM_CLASSES,
            ignore_index=255,
        )

        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})

        if curve_writer is not None:
            log_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            log_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            curve_writer.log_step(epoch, log_metrics)

        current_lr = optimizer.param_groups[0]["lr"]
        print("=" * 50)
        print(
            f"[HKCoral][{variant}] Epoch {epoch + 1}/{epochs} "
            f"Train loss={train_metrics.get('loss', 0.0):.4f} "
            f"acc={train_metrics.get('accuracy', 0.0):.3f} "
            f"mIoU={train_metrics.get('miou', 0.0):.3f} | "
            f"Val loss={val_metrics.get('loss', 0.0):.4f} "
            f"acc={val_metrics.get('accuracy', 0.0):.3f} "
            f"mIoU={val_metrics.get('miou', 0.0):.3f} | lr={current_lr:.2e}"
        )

        val_miou = val_metrics.get("miou", -1.0)
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_checkpoint_path)
            print(
                f"[HKCoral][{variant}] New best mIoU {best_miou:.4f} at epoch {best_epoch}. "
                f"Checkpoint updated: {best_checkpoint_path}"
            )

    if curve_writer is not None:
        curve_writer.save()

    if best_val_metrics is None:
        if history:
            best_val_metrics = history[-1]["val"]
            best_epoch = history[-1]["epoch"]
            best_miou = best_val_metrics.get("miou", best_miou)
        else:
            best_val_metrics = {}
        torch.save(model.state_dict(), best_checkpoint_path)

    best_state = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state)

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=str(device),
        amp_dtype=amp_dtype,
        num_classes=HKCORAL_NUM_CLASSES,
        ignore_index=255,
    )
    print(f"[HKCoral][{variant}] Test metrics: {test_metrics}")

    return {
        "history": history,
        "train_last": history[-1]["train"] if history else {},
        "best_val": best_val_metrics,
        "best_epoch": best_epoch,
        "best_miou": best_miou,
        "best_checkpoint": str(best_checkpoint_path),
        "test": test_metrics,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    amp_dtype = _resolve_amp_dtype(device)

    data_root = Path(args.data_root)
    train_loader = build_hk_coral_dataloader(
        root=str(data_root),
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        transform=resize_to_target,
    )
    val_loader = build_hk_coral_dataloader(
        root=str(data_root),
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        transform=resize_to_target,
    )
    test_loader = build_hk_coral_dataloader(
        root=str(data_root),
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        transform=resize_to_target,
    )

    variants = resolve_variants(args.models)
    curve_root = Path(args.curve_dir) if args.curve_dir is not None else None
    output_dir = Path(args.output_dir)
    all_results: Dict[str, Dict[str, object]] = {}

    for variant in variants:
        curve_path = None
        if curve_root is not None:
            curve_root.mkdir(parents=True, exist_ok=True)
            curve_path = curve_root / f"{variant}_curve.json"

        result = train_variant(
            variant=variant,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            curve_path=curve_path,
            output_dir=output_dir,
            dropout=args.dropout,
            warmup_epochs=args.warmup_epochs,
            amp_dtype=amp_dtype,
        )
        all_results[variant] = result

    if args.metrics_out is not None:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(all_results, fp, indent=2)
        print(f"[HKCoral] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
