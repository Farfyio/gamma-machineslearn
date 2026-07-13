"""Shared evaluation metric helpers."""

from typing import Mapping

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def classification_metrics(y_true, y_pred) -> Mapping[str, float]:
    """Compute a standard set of classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
