"""Shared preprocessing steps for anomaly detection models."""

from __future__ import annotations


def build_preprocessor():
    """Build the default preprocessing pipeline used by all models."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scale", StandardScaler())])
