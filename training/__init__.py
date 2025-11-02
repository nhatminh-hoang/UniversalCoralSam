"""Training and evaluation interface for UniversalCoralSam."""

from __future__ import annotations

from .train_loop import fit, train_one_epoch
from .evaluation import evaluate
from .inference import predict

__all__ = ["fit", "train_one_epoch", "evaluate", "predict"]

