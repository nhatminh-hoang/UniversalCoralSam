"""Helpers for configuring Hugging Face cache and offline behavior.

This module centralizes the environment variables that need to be toggled
when running in restricted environments (e.g., SLURM clusters without
internet access). Scripts can call :func:`configure_huggingface_environment`
to ensure a cache directory exists and that the appropriate offline flags
are set or cleared.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

__all__ = [
    "configure_huggingface_environment",
    "set_hf_cache_dir",
    "set_hf_offline_mode",
]

_OFFLINE_FLAGS: Iterable[str] = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "LOCAL_FILES_ONLY",
)


def set_hf_cache_dir(cache_dir: str | Path | None) -> Optional[Path]:
    """Resolve and create the Hugging Face cache directory if provided."""

    if cache_dir is None:
        return None
    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    return path


def set_hf_offline_mode(enabled: bool) -> None:
    """Toggle the common Hugging Face offline environment flags."""

    if enabled:
        for flag in _OFFLINE_FLAGS:
            os.environ[flag] = "1"
        return

    for flag in _OFFLINE_FLAGS:
        os.environ.pop(flag, None)


def configure_huggingface_environment(
    cache_dir: str | Path | None = None,
    *,
    offline: Optional[bool] = None,
) -> Optional[Path]:
    """Configure Hugging Face cache directory and offline state.

    Parameters
    ----------
    cache_dir:
        Optional directory where snapshots should be cached. When provided the
        directory is created (if needed) and exported via the ``HF_HOME`` env var.
    offline:
        When ``True`` the standard Hugging Face offline flags are enabled.
        When ``False`` they are cleared. When ``None`` the current environment
        flags are left untouched.

    Returns
    -------
    Optional[Path]
        The resolved cache directory when ``cache_dir`` is provided, otherwise
        ``None``.
    """

    cache_path = set_hf_cache_dir(cache_dir)
    if offline is not None:
        set_hf_offline_mode(bool(offline))
    return cache_path


if __name__ == "__main__":
    configure_huggingface_environment("artifacts/hf_cache_demo", offline=True)
