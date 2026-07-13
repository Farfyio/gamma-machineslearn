import unittest

import numpy as np

from gamma_machineslearn.models import ModelFactory
from gamma_machineslearn.pipeline import AnomalyDetectionExperiment


class IdentityPreprocessor:
    def fit_transform(self, x):
        return np.asarray(x)

    def transform(self, x):
        return np.asarray(x)


class StubModel:
    def fit(self, x):
        self._mean = float(np.mean(x))
        return self

    def predict(self, x):
        x = np.asarray(x)
        return np.where(np.mean(x, axis=1) > self._mean, -1, 1)

    def score_samples(self, x):
        return np.mean(np.asarray(x), axis=1)


class PipelineTests(unittest.TestCase):
    def test_run_trains_model_and_returns_metrics(self):
        factory = ModelFactory()
        factory.register("stub", lambda: StubModel())

        experiment = AnomalyDetectionExperiment(model_factory=factory, preprocessor=IdentityPreprocessor())

        x_train = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        x_test = np.array([[0.0, 0.0], [0.5, 0.5]])

        result = experiment.run(x_train=x_train, x_test=x_test, model_name="stub")

        self.assertEqual(result["predictions"].tolist(), [1, -1])
        self.assertEqual(result["metrics"], {"anomaly_fraction": 0.5})
        self.assertEqual(len(result["scores"]), 2)


if __name__ == "__main__":
    unittest.main()
