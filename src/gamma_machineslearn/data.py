"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_spectra_csv(path: str | Path, label_column: Optional[str] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load tabular spectra data from CSV."""
    import pandas as pd

    frame = pd.read_csv(path)
    labels = None

    if label_column is not None:
        if label_column not in frame.columns:
            raise ValueError(f"Unknown label column: {label_column}")
        labels = frame.pop(label_column).to_numpy()

    return frame.to_numpy(), labels
