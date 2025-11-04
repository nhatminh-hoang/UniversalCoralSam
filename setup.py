#!/usr/bin/env python3
"""Download pretrained segmentation checkpoints required for offline training.

This script downloads all HuggingFace model snapshots used by the HKCoral
training pipeline into a local cache directory. Run it on a machine with
internet access before submitting offline SLURM jobs.

Example
-------
python setup.py --cache-dir artifacts/hf_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - explicit message for missing deps
    raise SystemExit(
        "The 'huggingface_hub' package is required. Install via `pip install huggingface_hub transformers scipy`."
    ) from exc


DEFAULT_MODELS: List[str] = [
    "facebook/mask2former-swin-base-ade-semantic",
    "facebook/mask2former-swin-large-ade-semantic",
    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    # "facebook/dinov3-vits16-pretrain-lvd1689m",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HuggingFace checkpoints for offline training.")
    parser.add_argument(
        "--cache-dir",
        default="artifacts/hf_cache",
        help="Directory where model snapshots will be stored (defaults to artifacts/hf_cache).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of HuggingFace repository IDs to download instead of the defaults.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable snapshot resume support (forces a clean download each run).",
    )
    return parser.parse_args(argv)


def download_model(repo_id: str, cache_dir: Path, *, resume: bool) -> None:
    print(f"[setup] Downloading '{repo_id}' into '{cache_dir}'...")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        resume_download=resume,
        local_files_only=False,
    )
    print(f"[setup] ✓ Finished '{repo_id}'.")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    models = args.models if args.models is not None else DEFAULT_MODELS
    if not models:
        print("[setup] No models specified, nothing to do.")
        return 0

    resume = not args.no_resume
    for repo_id in models:
        try:
            download_model(repo_id, cache_dir, resume=resume)
        except Exception as exc:  # pragma: no cover - surface clear failure to operator
            print(f"[setup] ✗ Failed to download '{repo_id}': {exc}", file=sys.stderr)
            return 1

    print(
        "[setup] All requested models downloaded.\n"
        "         Set HF_HOME to this cache directory before launching offline training:\n"
        f"         export HF_HOME=\"{cache_dir}\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
