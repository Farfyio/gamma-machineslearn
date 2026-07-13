"""Helpers for summarizing evaluation artifacts."""

from pathlib import Path
from typing import Mapping



def write_metrics_report(metrics: Mapping[str, float], output_path: str | Path) -> None:
    """Write evaluation metrics to a simple text report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value:.6f}" for key, value in metrics.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
