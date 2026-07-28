"""Data loading utilities.

Central place for dataset IO. Exported functions are:
- `load_csv(path) -> np.ndarray` — numeric-only CSV loader (lazy pandas import)
- `load_spectra_csv(path, label_column=None) -> (X, y)` — loader that can
  extract a label column and return `(features, labels)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_csv(path: str) -> np.ndarray:
    """Load a CSV file into a numeric NumPy array.

    This performs a lazy import of pandas so pandas remains optional for
    synthetic-data runs. Non-numeric columns are dropped automatically.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required to load CSV data") from exc

    df = pd.read_csv(path)
    # Keep only numeric columns to return a clean feature array
    numeric = df.select_dtypes(include=[np.number])
    return numeric.values


def load_spectra_csv(path: str | Path, label_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load tabular spectra data from CSV.

    Returns (features, labels) where labels is None if `label_column` is
    not provided. This uses pandas and is intended for dataset files.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required to load CSV data") from exc

    frame = pd.read_csv(path)
    labels = None

    if label_column is not None:
        if label_column not in frame.columns:
            raise ValueError(f"Unknown label column: {label_column}")
        labels = frame.pop(label_column).to_numpy()

    return frame.to_numpy(), labels
