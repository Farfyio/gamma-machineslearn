"""Preprocessing utilities for research experiments."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def create_preprocessing_pipeline() -> Pipeline:
    """Create a lightweight default preprocessing pipeline."""

    return Pipeline([("scaler", StandardScaler())])
