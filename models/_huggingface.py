"""Shared HuggingFace utility helpers for model loading.

These helpers centralize snapshot discovery and offline-friendly logic that
other modules (e.g. segmentation backbones and DINO wrappers) rely on.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch

__all__ = [
    "can_use_bfloat16",
    "env_flag_true",
    "hf_cache_root",
    "maybe_raise_torch_upgrade",
    "offline_mode_enabled",
    "resolve_pretrained_identifier",
    "snapshot_has_safetensors",
]


def env_flag_true(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def offline_mode_enabled() -> bool:
    return any(
        env_flag_true(flag)
        for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "LOCAL_FILES_ONLY")
    )


def hf_cache_root() -> Path:
    root = os.environ.get("HF_HOME")
    if root:
        return Path(root).expanduser().resolve()
    return Path.home().expanduser() / ".cache" / "huggingface"


def find_local_snapshot(repo_id: str) -> Path | None:
    cache_root = hf_cache_root()
    models_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    refs_dir = models_dir / "refs"
    snapshots_dir = models_dir / "snapshots"

    if refs_dir.exists():
        for ref_file in refs_dir.iterdir():
            if ref_file.is_file():
                snapshot_name = ref_file.read_text(encoding="utf-8").strip()
                if snapshot_name:
                    candidate = snapshots_dir / snapshot_name
                    if candidate.exists():
                        return candidate

    if snapshots_dir.exists():
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return snapshots[0]
    return None


def resolve_pretrained_identifier(repo_id: str) -> Tuple[str, bool]:
    local_snapshot = find_local_snapshot(repo_id)
    if local_snapshot is not None:
        return str(local_snapshot), True

    if offline_mode_enabled():
        raise FileNotFoundError(
            f"HuggingFace weights '{repo_id}' were not found in the local cache at '{hf_cache_root()}'. "
            "Download them on a machine with internet access via `python setup.py --cache-dir <path>` "
            "and copy the cache before running in offline mode."
        )
    return repo_id, False


def can_use_bfloat16() -> bool:
    if torch.cuda.is_available():
        is_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if is_supported:
            return True
    return False


def snapshot_has_safetensors(path: str) -> bool:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        return False
    if snapshot_path.is_file():
        return snapshot_path.suffix == ".safetensors"
    safetensors = list(snapshot_path.glob("**/*.safetensors"))
    return len(safetensors) > 0


def maybe_raise_torch_upgrade(exc: Exception, repo_id: str) -> None:
    message = str(exc)
    if "upgrade torch to at least v2.6" in message:
        raise RuntimeError(
            "Loading HuggingFace weights failed because torch<2.6 blocks `torch.load`. "
            f"Ensure safetensor weights are available for '{repo_id}' by running the setup downloader "
            "(`python setup.py ...`) or manually converting the checkpoint to safetensors."
        ) from exc
