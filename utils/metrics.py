"""Lightweight metric logger for tracking training progress.

Example
-------
>>> from utils.metrics import MetricLogger
>>> logger = MetricLogger()
>>> logger.update(loss=0.5, miou=0.3)
>>> logger.update(loss=0.4)
>>> logger.history["loss"]
[0.5, 0.4]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricLogger:
    """Stores metric values and provides quick summaries."""

    history: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, **metrics: float) -> None:
        for name, value in metrics.items():
            series = self.history.setdefault(name, [])
            series.append(float(value))

    def summary(self) -> Dict[str, float]:
        return {name: sum(values) / max(len(values), 1) for name, values in self.history.items()}


if __name__ == "__main__":
    logger = MetricLogger()
    for step in range(3):
        logger.update(loss=1.0 / (step + 1))
    print(logger.summary())
