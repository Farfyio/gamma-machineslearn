"""Evaluation utilities shared across experiments."""

from __future__ import annotations

from typing import Mapping

import numpy as np
#I did not know we can group imports together like this
from sklearn.metrics import (f1_score, 
                             precision_score, 
                             recall_score, 
                             roc_auc_score,
                             confusion_matrix
                             )



def anomaly_detection_metrics(y_true, y_pred, y_score=None) -> Mapping[str, float]:
    """Compute common anomaly-detection metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_score: Optional anomaly score used for ROC-AUC.
    """

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics



def evaluate_predictions(predictions: np.ndarray, y_true: np.ndarray | None = None) -> dict:
    
    """Evaluate anomaly predictions with or without labels."""
    predictions = np.asarray(predictions)

    if y_true is None:
        return {"anomaly_fraction": float(np.mean(predictions == -1))}

    from sklearn.metrics import accuracy_score

    y_true = np.asarray(y_true)
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[-1, -1]
    
    )

    return {
        "precision": float(precision_score(y_true, predictions, pos_label=-1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=-1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=-1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "confusion_matrix": matrix,
    }
