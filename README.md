# gamma-machineslearn

Machine learning models for anomaly detection in HPGe gamma-ray spectra.

## Project structure

- `gamma_machineslearn/data.py`: reusable data loading utilities.
- `gamma_machineslearn/preprocessing.py`: shared preprocessing pipeline.
- `gamma_machineslearn/models.py`: model registry/factory with built-in One-Class SVM and Isolation Forest support.
- `gamma_machineslearn/pipeline.py`: common train/evaluate experiment pipeline.
- `gamma_machineslearn/evaluation.py`: reusable evaluation helpers.
- `gamma_machineslearn/visualization.py`: reusable plotting helpers.

## Confidential research data

Research data is confidential and must never be committed. Keep local datasets in ignored directories such as `data/`, `datasets/`, or `research_data/`.

## Extending with new algorithms

Register a custom estimator constructor in `ModelFactory.register(...)` and run it through `AnomalyDetectionExperiment` to reuse the same preprocessing and evaluation flow.
