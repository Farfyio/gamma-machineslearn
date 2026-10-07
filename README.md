# gamma-machineslearn

Research project focused on anomaly detection in gamma spectra using machine learning models.

## Research objective

Build and compare anomaly-detection approaches (currently One-Class SVM and Isolation Forest) to detect unusual gamma spectra patterns.

## Structure

```text
.
├── README.md
├── .gitignore
├── main.py
├── requirements.txt
├── datasets/
│   └── cards_data.csv
├── src/
│   └── gamma_machineslearn/
│       ├── __init__.py
│       ├── data.py
│       ├── pipeline.py
│       ├── visualization.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── pipeline.py
│       └── models/
│           ├── __init__.py
│           ├── factory.py
│           ├── one_class_svm.py
│           ├── isolation_forest.py
│           └── SGD_one_class_svm.py
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_evaluation.py
    ├── test_models.py
    ├── test_pipeline.py
    └── test_preprocessing.py
```

## What each part is for

- `main.py`: command-line entry point for running experiments and generating confusion matrix images.
- `datasets/`: local dataset files for experiments.
- `src/gamma_machineslearn/`: main package.
  - `models/`: model implementations and factory registration.
  - `pipeline.py`: shared train/predict/evaluate flow.
  - `data.py`: data loading helpers.
  - `visualization.py`: plotting helpers (including confusion matrix saving).
  - `config/settings.py`: configuration dataclass and loader.
  - `evaluation/metrics.py`: metrics and confusion matrix calculation.
  - `preprocessing/pipeline.py`: preprocessing pipeline creators.
- `tests/`: unit tests for configuration, preprocessing, evaluation, models, and pipeline behavior.

## Where to put your machine learning code

- Put **new model integrations** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/src/gamma_machineslearn/models/` and register through the model factory.
- Put **experiment flow updates** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/src/gamma_machineslearn/pipeline.py`.
- Put **feature/data preprocessing steps** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/src/gamma_machineslearn/preprocessing/pipeline.py`.
- Put **evaluation logic or new metrics** in `/home/runner/work/gamma-machineslearn/gamma-machineslearn/src/gamma_machineslearn/evaluation/metrics.py`.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `PYTHONPATH=src python -m unittest discover -s tests -v`
4. Run an experiment (example):
   - `PYTHONPATH=src python main.py --model one_class_svm --data /path/to/combined_csv_dir`

## First test results (share here)

Add your first confusion matrices in this section after running experiments.

### One-Class SVM confusion matrix

![One-Class SVM confusion matrix](confusion_matrix_one_class_svm.png)

### Isolation Forest confusion matrix

![Isolation Forest confusion matrix](confusion_matrix_isolation_forest.png)
