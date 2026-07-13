"""Evaluation helpers shared by anomaly detection models."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def evaluate_predictions(predictions: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Evaluate anomaly predictions with or without labels.

    For models that emit ``1`` for inlier and ``-1`` for anomaly, this helper
    reports the anomaly fraction without labels and standard classification
    metrics when labels are provided.
    """
    predictions = np.asarray(predictions)

    if y_true is None:
        return {"anomaly_fraction": float(np.mean(predictions == -1))}

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true)
    return {
        "precision": float(precision_score(y_true, predictions, pos_label=-1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=-1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=-1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }
