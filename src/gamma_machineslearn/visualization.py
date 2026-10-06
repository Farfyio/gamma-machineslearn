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

def save_confusion_matrix(matrix,title="Confusion Matrix", output_path: str = "confusion_matrix.png") -> None:
    """Save a confusion matrix as an image."""

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Pure", "Impure"]

    )

    disp.plot(cmap = "viridis", values_format = "d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()

