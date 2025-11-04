"""Generate and persist simple training curves from metric logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class TrainingCurveWriter:
    """Collects metrics and writes a PNG figure showing their evolution."""

    output_path: str
    steps: List[int] = field(default_factory=list)
    traces: Dict[str, List[float]] = field(default_factory=dict)

    def log_step(self, iteration: int, metrics: Dict[str, float]) -> None:
        self.steps.append(iteration)
        for name, value in metrics.items():
            lower = name.lower()
            if "time" in lower or lower == "samples_per_sec":
                continue
            series = self.traces.setdefault(name, [])
            series.append(float(value))

    def save(self) -> None:
        if not self.steps:
            raise ValueError("No metrics have been logged; call log_step first.")
        plt.figure(figsize=(6, 4))
        for name, values in self.traces.items():
            plt.plot(self.steps, values, label=name)
        plt.xlabel("Iteration")
        plt.ylabel("Metric value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=150)
        plt.close()


if __name__ == "__main__":
    writer = TrainingCurveWriter("training_curve_demo.png")
    for step in range(5):
        writer.log_step(step, {"loss": 1 / (step + 1), "miou": step / 5})
    writer.save()
