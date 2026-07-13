"""Visualization utilities for anomaly detection experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_anomaly_scores(scores: np.ndarray, threshold: Optional[float] = None, ax=None):
    """Plot sample-wise anomaly scores."""
    import matplotlib.pyplot as plt

    scores = np.asarray(scores)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    ax.plot(scores, linewidth=1)
    ax.set_title("Anomaly scores")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Score")

    if threshold is not None:
        ax.axhline(y=threshold, linestyle="--")

    return ax
