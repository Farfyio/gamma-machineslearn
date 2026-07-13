"""Evaluation utilities shared across experiments."""

from typing import Mapping

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



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
