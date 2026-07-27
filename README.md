# gamma-machineslearn

Lightweight research project scaffold for anomaly detection on gamma-ray spectra.

## Goals

- Keep shared code small and easy to extend.
- Support rapid experiments for One-Class SVM and Isolation Forest.
- Avoid dataset-specific logic inside reusable modules.

## Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   └── gamma_machineslearn/
│       ├── __init__.py
│       ├── data.py
│       ├── models.py
│       ├── pipeline.py
│       ├── visualization.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── preprocessing/
│           ├── __init__.py
│           └── pipeline.py
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_evaluation.py
    ├── test_models.py
    ├── test_pipeline.py
    └── test_preprocessing.py
```

## What each part is for

- `README.md`: project overview and usage notes.
- `.gitignore`: files/folders excluded from version control.
- `requirements.txt`: Python dependencies.
- `src/gamma_machineslearn/`: main package.
  - `models.py`: register/create anomaly models (**add custom model constructors here**).
  - `pipeline.py`: shared train/predict/evaluate flow.
  - `data.py`: CSV/data loading helpers.
  - `visualization.py`: plotting helpers for anomaly scores.
  - `config/settings.py`: YAML config dataclass + loader.
  - `evaluation/metrics.py`: anomaly metric helpers and prediction evaluation.
  - `preprocessing/pipeline.py`: preprocessing pipeline creators.
  - `__init__.py` files: package/module exports.
- `tests/`: unit tests for config, preprocessing, evaluation, models, and pipeline.

## Where to put your machine learning code

- Put **new model integrations** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/src/gamma_machineslearn/models.py` via `ModelFactory.register(...)`.
- Put **experiment flow updates** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/src/gamma_machineslearn/pipeline.py`.
- Put **feature/data preprocessing steps** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/src/gamma_machineslearn/preprocessing/pipeline.py`.
- Put **evaluation logic or new metrics** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/src/gamma_machineslearn/evaluation/metrics.py`.
- Keep reusable helpers inside package modules, and keep one-off notebooks/scripts outside shared modules.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run tests with `PYTHONPATH=src python -m unittest discover -s tests -v`.
4. Add model-specific experiment logic under `src/gamma_machineslearn/` and keep reusable helpers in package submodules.

## Guardrails

- Keep shared modules dataset-agnostic.
- Add model-specific workflows separately from reusable utilities.
