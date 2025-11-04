"""Train requested segmentation models on the HKCoral dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from dataset import build_hk_coral_dataloader
from dataset.hk_transforms import resize_to_target
from models import get_model
from training import evaluate, train_one_epoch
from utils import configure_huggingface_environment
from utils.configuration import HKCoralTrainingConfig, load_hkcoral_config
from utils.curves import TrainingCurveWriter
from utils.reproducibility import make_worker_seed_fn, seed_everything


HKCORAL_NUM_CLASSES = 7
MODEL_VARIANTS: Dict[str, List[str]] = {
    "deeplabv3": ["deeplabv3_resnet50", "deeplabv3_resnet101"],
    "mask2former": ["mask2former_swin_base", "mask2former_swin_large"],
    "segformer": ["segformer_b2_cityscapes", "segformer_b5_cityscapes"],
}

LOGGER = logging.getLogger("hkcoral.train")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "hkcoral_default.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HKCoral segmentation models.")
    parser.add_argument("--config", default=None, help="Optional path to a HKCoral JSON configuration file.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to the HKCoral dataset root containing 'images' and 'labels' directories.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model families to train. Choices: deeplabv3, mask2former, segformer, or concrete variant names.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs per model variant.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay for the optimizer.")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Epoch when the sinusoidal LR reaches its peak.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability applied to Dropout layers.")
    parser.add_argument("--num-workers", type=int, default=None, help="Worker count for data loading.")
    parser.add_argument("--hf-cache-dir", default=None, help="Override the HuggingFace cache directory.")
    parser.set_defaults(hf_offline_mode=None)
    parser.add_argument(
        "--force-offline",
        dest="hf_offline_mode",
        action="store_true",
        help="Force offline mode when resolving HuggingFace checkpoints.",
    )
    parser.add_argument(
        "--allow-online",
        dest="hf_offline_mode",
        action="store_false",
        help="Allow remote downloads when resolving HuggingFace checkpoints.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU even if CUDA is available.")
    parser.set_defaults(deterministic=None)
    parser.set_defaults(exclude_label0_metrics=None)
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Enable deterministic training operations where possible.",
    )
    parser.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic training safeguards for higher throughput.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--curve-dir",
        default=None,
        help="If provided, save training curves under this directory (one file per model variant).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store trained checkpoints. Checkpoints are saved as <variant>.pth.",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional path to store per-variant training and evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--exclude-label0-metrics",
        dest="exclude_label0_metrics",
        action="store_true",
        help="Exclude label 0 from accuracy/mIoU calculations.",
    )
    parser.add_argument(
        "--include-label0-metrics",
        dest="exclude_label0_metrics",
        action="store_false",
        help="Include label 0 when computing accuracy/mIoU.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Optional logging level override.",
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


def configure_logging(log_level: Optional[str]) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO) if log_level else logging.INFO
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(level)
    LOGGER.setLevel(level)


def build_training_config(args: argparse.Namespace) -> HKCoralTrainingConfig:
    candidate_paths = []
    if args.config is not None:
        candidate_paths.append(args.config)
    candidate_paths.append(DEFAULT_CONFIG_PATH)
    config = load_hkcoral_config(candidate_paths)
    overrides = {
        "data_root": args.data_root,
        "models": args.models,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "dropout": args.dropout,
        "num_workers": args.num_workers,
        "curve_dir": args.curve_dir,
        "output_dir": args.output_dir,
        "metrics_out": args.metrics_out,
        "hf_cache_dir": args.hf_cache_dir,
        "hf_offline_mode": args.hf_offline_mode,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "exclude_label0_metrics": args.exclude_label0_metrics,
    }
    config.merge_overrides(overrides)
    config.cpu = bool(args.cpu or config.cpu)
    return config


def configure_environment(config: HKCoralTrainingConfig) -> None:
    cache_path = configure_huggingface_environment(
        config.hf_cache_dir,
        offline=config.hf_offline_mode,
    )
    if cache_path is not None:
        LOGGER.info("Using Hugging Face cache directory: %s", cache_path)
    if config.hf_offline_mode:
        LOGGER.info("Hugging Face offline mode enabled.")


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
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            LOGGER.info("Using CUDA bfloat16 autocast for training.")
            return torch.bfloat16
        LOGGER.warning("CUDA bfloat16 not supported on this platform. Falling back to float32.")
        return None
    if device.type == "cpu":
        LOGGER.info("Using CPU bfloat16 autocast for training.")
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
    ignore_labels: Optional[Tuple[int, ...]] = None,
) -> Dict[str, object]:
    variant_logger = LOGGER.getChild(variant)
    variant_logger.info("Starting training on device %s", device)
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
            ignore_labels=ignore_labels,
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
            ignore_labels=ignore_labels,
        )

        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})

        if curve_writer is not None:
            log_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            log_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            curve_writer.log_step(epoch, log_metrics)

        current_lr = optimizer.param_groups[0]["lr"]
        variant_logger.info(
            "Epoch %d/%d | Train: loss=%.4f acc=%.3f mIoU=%.3f | "
            "Val: loss=%.4f acc=%.3f mIoU=%.3f | lr=%.2e",
            epoch + 1,
            epochs,
            train_metrics.get("loss", 0.0),
            train_metrics.get("accuracy", 0.0),
            train_metrics.get("miou", 0.0),
            val_metrics.get("loss", 0.0),
            val_metrics.get("accuracy", 0.0),
            val_metrics.get("miou", 0.0),
            current_lr,
        )

        val_miou = val_metrics.get("miou", -1.0)
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_checkpoint_path)
            variant_logger.info(
                "New best mIoU %.4f at epoch %d. Checkpoint saved to %s",
                best_miou,
                best_epoch,
                best_checkpoint_path,
            )

    if curve_writer is not None:
        curve_writer.save()

    if best_val_metrics is None:
        best_val_metrics = history[-1]["val"] if history else {}
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
        ignore_labels=ignore_labels,
    )
    variant_logger.info("Test metrics: %s", test_metrics)

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
    configure_logging(args.log_level)
    config = build_training_config(args)
    configure_environment(config)

    dataloader_generator: Optional[torch.Generator] = None
    worker_init_fn = None
    if config.seed is not None:
        seed_everything(config.seed, deterministic=config.deterministic)
        LOGGER.info("Seeded RNGs with seed=%d (deterministic=%s)", config.seed, config.deterministic)
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(config.seed)
        worker_init_fn = make_worker_seed_fn(config.seed)

    device = torch.device("cpu" if config.cpu or not torch.cuda.is_available() else "cuda")
    amp_dtype = _resolve_amp_dtype(device)

    LOGGER.info("Using device %s", device)
    LOGGER.debug("Resolved configuration: %s", json.dumps(config.as_dict(), indent=2))

    metric_ignore_labels: Optional[Tuple[int, ...]] = (0,) if config.exclude_label0_metrics else None

    data_root = Path(config.data_root)
    train_loader = build_hk_coral_dataloader(
        root=str(data_root),
        split="train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        transform=resize_to_target,
        worker_init_fn=worker_init_fn,
        generator=dataloader_generator,
    )
    val_loader = build_hk_coral_dataloader(
        root=str(data_root),
        split="val",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        transform=resize_to_target,
        worker_init_fn=worker_init_fn,
        generator=dataloader_generator,
    )
    test_loader = build_hk_coral_dataloader(
        root=str(data_root),
        split="test",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        transform=resize_to_target,
        worker_init_fn=worker_init_fn,
        generator=dataloader_generator,
    )

    variants = resolve_variants(config.models)
    curve_root = Path(config.curve_dir).expanduser() if config.curve_dir is not None else None
    output_dir = Path(config.output_dir).expanduser()
    all_results: Dict[str, Dict[str, object]] = {}

    for variant in variants:
        curve_path = None
        if curve_root is not None:
            curve_root.mkdir(parents=True, exist_ok=True)
            curve_path = curve_root / f"{variant}_curve.png"

        result = train_variant(
            variant=variant,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            curve_path=curve_path,
            output_dir=output_dir,
            dropout=config.dropout,
            warmup_epochs=config.warmup_epochs,
            amp_dtype=amp_dtype,
            ignore_labels=metric_ignore_labels,
        )
        all_results[variant] = result

    if config.metrics_out is not None:
        metrics_path = Path(config.metrics_out).expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(all_results, fp, indent=2)
        LOGGER.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
