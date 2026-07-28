"""Reusable experiment pipeline for anomaly detection research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .evaluation import evaluate_predictions
from .models import ModelFactory
from .preprocessing import build_preprocessor


@dataclass
class AnomalyDetectionExperiment:
    """Shared preprocessing/training/evaluation pipeline for anomaly models."""

    model_factory: ModelFactory = field(default_factory=ModelFactory)
    preprocessor: Any = field(default_factory=build_preprocessor)

    def run(
        self,
        x_train,
        x_test,
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None,
        y_test=None,
    ) -> Dict[str, Any]:
        """Train and evaluate an anomaly detection model on preprocessed data."""
        model_params = model_params or {}

        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)

        x_train_transformed = self.preprocessor.fit_transform(x_train)
        x_test_transformed = self.preprocessor.transform(x_test)

        model = self.model_factory.create(model_name, **model_params)
       #Mkaes sure that the modeland params areactually there
        print(f"Running experiment with model: {model}")

        model.fit(x_train_transformed)
        predictions = model.predict(x_test_transformed)

        if hasattr(model, "decision_function"):
            scores = model.decision_function(x_test_transformed)
        elif hasattr(model, "score_samples"):
            scores = model.score_samples(x_test_transformed)
        else:
            scores = None

        metrics = evaluate_predictions(predictions=predictions, y_true=y_test)

        return {
            "model": model,
            "predictions": predictions,
            "scores": scores,
            "metrics": metrics,
            "x_train_transformed": x_train_transformed,
            "x_test_transformed": x_test_transformed,
        }
