"""Placeholder feature transform utilities."""

from pandas import DataFrame



def validate_features(features: DataFrame) -> DataFrame:
    """Placeholder feature validation hook for reusable preprocessing."""

    if features.empty:
        raise ValueError("Input features must not be empty.")
    return features
