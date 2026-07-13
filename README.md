# gamma-machineslearn

Machine learning project scaffolding for anomaly detection workflows in HPGe gamma-ray spectra.

## What is included

- Reusable, dataset-agnostic preprocessing pipeline utilities
- Shared evaluation metrics and report-writing helpers
- Central logging configuration helpers
- YAML-based project configuration loading
- Project documentation skeleton

## Repository layout

```text
.
├── docs/
│   ├── configuration.md
│   └── project_structure.md
├── requirements.txt
├── src/
│   └── gamma_machineslearn/
│       ├── config/
│       ├── evaluation/
│       ├── logging_utils/
│       └── preprocessing/
└── tests/
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Build on top of `src/gamma_machineslearn` modules for dataset-specific experiments.

## Design guardrails

- Shared modules in `src/gamma_machineslearn` must remain dataset-agnostic.
- Dataset-specific loading/cleaning logic should live in dedicated experiment code.
