"""Reusable preprocessing pipeline building blocks."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def create_preprocessing_pipeline() -> Pipeline:
    """Return a baseline preprocessing pipeline.

    Note:
        Keep this dataset-agnostic so features can be plugged in by callers.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
