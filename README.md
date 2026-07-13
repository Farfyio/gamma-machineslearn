# gamma-machineslearn

Lightweight research project scaffold for anomaly detection on gamma-ray spectra.

## Goals

- Keep shared code small and easy to extend.
- Support rapid experiments for One-Class SVM and Isolation Forest.
- Avoid dataset-specific logic inside reusable modules.

## Structure

```text
.
├── requirements.txt
├── src/
│   └── gamma_machineslearn/
│       ├── config/
│       ├── evaluation/
│       └── preprocessing/
└── tests/
```

## Included utilities

- `config`: YAML-backed `ProjectConfig` loading
- `preprocessing`: reusable sklearn preprocessing pipeline
- `evaluation`: anomaly-detection metric helpers

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Extend from `src/gamma_machineslearn` for model-specific experiment scripts.

## Guardrails

- Keep shared modules dataset-agnostic.
- Add model-specific workflows separately from reusable utilities.
