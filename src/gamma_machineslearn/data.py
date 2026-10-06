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
    # loads the csv file into a DataFrame
    frame = pd.read_csv(path)
    labels = None

    if label_column is not None:
        if label_column not in frame.columns:
            raise ValueError(f"Unknown label column: {label_column}")
        labels = frame.pop(label_column).to_numpy()

    return frame.to_numpy(), labels

def load_single_spectrum(path:str | Path) -> np.ndarray:
    """Load one processgamma spectrum as one ML sample"""
    
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Pandas is required to load CSV data") from exc

    frame = pd.read_csv(path)

    if "counts per channel" not in frame.columns:
        raise ValueError(f"Unknown column: counts per channel")

    counts = frame["counts per channel"].to_numpy(dtype=float)

    return counts.reshape(1, -1)

def load_spectra_directory(directory: str | Path):
    """Load all spectra in a directory"""

    directory = Path(directory)

    spectra = []
    labels = []

    for file_path in sorted(directory.rglob("combined_spectrum_*.csv")):
        spectrum = load_single_spectrum(file_path)
        spectra.append(spectrum.flatten())

        parts = file_path.stem.split("_")
        impure_value = int(parts[3])

        #Follwoing the namieng convention of the deaflut ML models, we will use 1 for pure and -1 for impure
        if impure_value == 0:
            labels.append(1)
        else:
            labels.append(-1)
    return np.asarray(spectra), np.asarray(labels)
