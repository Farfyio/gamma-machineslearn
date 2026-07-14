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
├── gamma_machineslearn/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluation.py
│   ├── models.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── visualization.py
├── src/
│   └── gamma_machineslearn/
│       ├── __init__.py
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
- `gamma_machineslearn/`: main experiment package.
  - `models.py`: register/create anomaly models (**add custom model constructors here**).
  - `pipeline.py`: shared train/predict/evaluate flow.
  - `preprocessing.py`: shared preprocessing builder.
  - `evaluation.py`: metrics and result summaries.
  - `data.py`: CSV/data loading helpers.
  - `visualization.py`: plotting helpers for anomaly scores.
  - `__init__.py`: package exports.
- `src/gamma_machineslearn/`: modular utilities package.
  - `config/settings.py`: YAML config dataclass + loader.
  - `evaluation/metrics.py`: anomaly metric helpers.
  - `preprocessing/pipeline.py`: preprocessing pipeline creator.
  - `__init__.py` files: package/module exports.
- `tests/`: unit tests for config, preprocessing, evaluation, models, and pipeline.

## Where to put your machine learning code

- Put **new model integrations** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/gamma_machineslearn/models.py` via `ModelFactory.register(...)`.
- Put **experiment flow updates** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/gamma_machineslearn/pipeline.py`.
- Put **feature/data preprocessing steps** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/gamma_machineslearn/preprocessing.py`.
- Put **evaluation logic or new metrics** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/gamma_machineslearn/evaluation.py`.
- Keep reusable helpers inside package modules, and keep one-off notebooks/scripts outside shared modules.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Add model-specific experiment logic under `gamma_machineslearn/` (especially `models.py` and `pipeline.py`) and keep reusable config/metrics helpers in `src/gamma_machineslearn/`.

## Guardrails

- Keep shared modules dataset-agnostic.
- Add model-specific workflows separately from reusable utilities.
