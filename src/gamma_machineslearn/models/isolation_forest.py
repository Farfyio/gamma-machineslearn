"""Isolation Forest model builder."""

from __future__ import annotations

try:
    from sklearn.ensemble import IsolationForest
except Exception as exc:  # pragma: no cover - environment dependent
    _SKLEARN_IMPORT_ERROR = exc
    IsolationForest = None
else:
    _SKLEARN_IMPORT_ERROR = None


def _require_sklearn() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "scikit-learn is required for Isolation Forest."
        ) from _SKLEARN_IMPORT_ERROR


def isolation_forest_factory(**kwargs):
    _require_sklearn()
    return IsolationForest(**kwargs)