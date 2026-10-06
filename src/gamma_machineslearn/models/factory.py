"""Model factory and registry for anomaly detection algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Protocol

#The estimatorfactory makes a model constrcutor callabe
class EstimatorFactory(Protocol):
    def __call__(self, **kwargs): ...


from .isolation_forest import isolation_forest_factory
from .one_class_svm import one_class_svm_factory
'''
The ModelFactory class, storesa dictionary of model names
 and their corresponding factory functions.
'''
@dataclass
class ModelFactory:
    """Registry-backed model factory for extensible anomaly detection experiments."""

    _registry: Dict[str, EstimatorFactory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._registry:
            self.register("one_class_svm", one_class_svm_factory)
            
            self.register("isolation_forest", isolation_forest_factory)
            
        #the register functionaddsa  model to the registry dicitonary
    def register(self, name: str, builder: EstimatorFactory) -> None:
        """Register a model constructor for future experimentation."""
        key = self._normalize_name(name)
        self._registry[key] = builder
        #the create fucntion createsa model instance by name

    def create(self, name: str, **kwargs):
        """Create a model instance by name."""
        key = self._normalize_name(name)
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(f"Unknown model '{name}'. Available models: {available}")
        return self._registry[key](**kwargs)
    
    #the normalize name fucntion maes the names constistent
    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower()