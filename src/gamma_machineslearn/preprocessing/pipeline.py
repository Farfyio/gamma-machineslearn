"""Preprocessing utilities for research experiments."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def create_preprocessing_pipeline() -> Pipeline:
    """Create a lightweight default preprocessing pipeline."""

    return Pipeline([("scaler", StandardScaler())])


def build_preprocessor() -> Pipeline:
    """Build the default preprocessing pipeline used by all models."""

    return Pipeline([("scale", StandardScaler())])
