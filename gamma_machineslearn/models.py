"""Model factory and registry for anomaly detection algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Protocol


class EstimatorFactory(Protocol):
    def __call__(self, **kwargs): ...


try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
except Exception as exc:  # pragma: no cover - environment dependent
    _SKLEARN_IMPORT_ERROR = exc
    IsolationForest = None
    OneClassSVM = None
else:
    _SKLEARN_IMPORT_ERROR = None


def _require_sklearn() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "scikit-learn is required for built-in models. "
            "Install scikit-learn to use One-Class SVM and Isolation Forest."
        ) from _SKLEARN_IMPORT_ERROR


def _one_class_svm_factory(**kwargs):
    _require_sklearn()
    return OneClassSVM(**kwargs)


def _isolation_forest_factory(**kwargs):
    _require_sklearn()
    return IsolationForest(**kwargs)


@dataclass
class ModelFactory:
    """Registry-backed model factory for extensible anomaly detection experiments."""

    _registry: Dict[str, EstimatorFactory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._registry:
            self.register("one_class_svm", _one_class_svm_factory)
            self.register("isolation_forest", _isolation_forest_factory)

    def register(self, name: str, builder: EstimatorFactory) -> None:
        """Register a model constructor for future experimentation."""
        key = self._normalize_name(name)
        self._registry[key] = builder

    def create(self, name: str, **kwargs):
        """Create a model instance by name."""
        key = self._normalize_name(name)
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(f"Unknown model '{name}'. Available models: {available}")
        return self._registry[key](**kwargs)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower()
