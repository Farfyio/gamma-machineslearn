"""One-Class SVM model builder."""

from __future__ import annotations

try:
    from sklearn.svm import OneClassSVM
except Exception as exc:  # pragma: no cover - environment dependent
    _SKLEARN_IMPORT_ERROR = exc
    OneClassSVM = None
else:
    _SKLEARN_IMPORT_ERROR = None


def _require_sklearn() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "scikit-learn is required for One-Class SVM."
        ) from _SKLEARN_IMPORT_ERROR


def one_class_svm_factory(**kwargs):
    _require_sklearn()
    return OneClassSVM(**kwargs)