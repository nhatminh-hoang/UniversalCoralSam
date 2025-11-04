"""Configuration helpers for UniversalCoralSam training workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

HKCORAL_DEFAULT_MODELS = ["deeplabv3", "mask2former", "segformer"]


@dataclass
class HKCoralTrainingConfig:
    """Structured configuration for HKCoral training pipelines."""

    data_root: str = "HKCoral"
    models: list[str] = field(default_factory=lambda: HKCORAL_DEFAULT_MODELS.copy())
    batch_size: int = 2
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 10
    dropout: float = 0.0
    num_workers: int = 4
    cpu: bool = False
    curve_dir: Optional[str] = None
    output_dir: str = "artifacts/hkcoral"
    metrics_out: Optional[str] = None
    hf_cache_dir: Optional[str] = "artifacts/hf_cache"
    hf_offline_mode: bool = True
    seed: Optional[int] = 42
    deterministic: bool = True
    exclude_label0_metrics: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HKCoralTrainingConfig":
        """Instantiate from a raw mapping, ignoring unknown keys."""

        valid_fields = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> "HKCoralTrainingConfig":
        """Load configuration from a JSON file."""

        with Path(path).expanduser().open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, dict):
            raise ValueError(f"HKCoral config file '{path}' must describe an object.")
        return cls.from_dict(payload)

    def merge_overrides(self, overrides: Dict[str, Any]) -> "HKCoralTrainingConfig":
        """Apply explicit overrides (e.g. from CLI arguments)."""

        valid_fields = {field.name for field in fields(self)}
        for key, value in overrides.items():
            if value is None or key not in valid_fields:
                continue
            setattr(self, key, value)
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a primitive dictionary."""

        return {
            "data_root": self.data_root,
            "models": list(self.models),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "dropout": self.dropout,
            "num_workers": self.num_workers,
            "cpu": self.cpu,
            "curve_dir": self.curve_dir,
            "output_dir": self.output_dir,
            "metrics_out": self.metrics_out,
            "hf_cache_dir": self.hf_cache_dir,
            "hf_offline_mode": self.hf_offline_mode,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "exclude_label0_metrics": self.exclude_label0_metrics,
        }


def load_hkcoral_config(candidate_paths: Iterable[str | Path] | None = None) -> HKCoralTrainingConfig:
    """Load the first available HKCoral configuration from the provided paths."""

    if candidate_paths is None:
        return HKCoralTrainingConfig()

    for path in candidate_paths:
        if path is None:
            continue
        expanded = Path(path).expanduser()
        if expanded.exists():
            return HKCoralTrainingConfig.from_json(expanded)
    return HKCoralTrainingConfig()


__all__ = ["HKCoralTrainingConfig", "HKCORAL_DEFAULT_MODELS", "load_hkcoral_config"]
