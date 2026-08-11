"""Tiny CLI runner for quick experiments.

This script is intentionally minimal so you can quickly run the
experiment pipeline locally. It supports either synthetic data (default)
or loading a CSV dataset via `--data PATH` (CSV must be numeric features).

Edit the `main()` function or the `argparse` options below to change
how experiments are run (seed, model params, data loading, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.ma import exp

from gamma_machineslearn import AnomalyDetectionExperiment
from gamma_machineslearn.data import load_csv

#default data path thatI can change with whatver data i need
deafult_data_path = Path("datasets") / "cards_data.csv"

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny anomaly experiment")
    parser.add_argument("-m", "--model", default="one_class_svm", help="Model name to use (one_class_svm or isolation_forest)")
    parser.add_argument("--n-train", type=int, default=100, help="Number of synthetic training samples")
    parser.add_argument("--n-test", type=int, default=50, help="Number of synthetic test samples")
    parser.add_argument("--features", type=int, default=10, help="Number of features for synthetic data")
    parser.add_argument("--data", type=str, default=deafult_data_path, help="Path to CSV file to use instead of synthetic data")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (0 = no seed)")

    args = parser.parse_args(argv)

    if args.seed:
        np.random.seed(args.seed)

    if args.data:
        X = load_csv(args.data)
        # simple split: first n_train rows -> train, rest -> test (if available)
        n_train = min(args.n_train, X.shape[0])
        X_train = X[:n_train]
        X_test = X[n_train : n_train + args.n_test]
        if X_test.size == 0:
            # If there are not enough rows, reuse a slice for test
            X_test = X[: args.n_test]
    else:
        # Synthetic data: standard normal features
        X_train = np.random.randn(args.n_train, args.features)
        X_test = np.random.randn(args.n_test, args.features)

    exp = AnomalyDetectionExperiment()

    print(f"Running experiment with model={args.model}")
    #result = exp.run(X_train, X_test, model_name=args.model)

    # Set model parameters based on the selected model
    if args.model == "one_class_svm":
        model_params = {
        "nu": 0.05,
        "kernel": "rbf",
        "gamma": "scale",
    }
    elif args.model == "isolation_forest":
        model_params = {
        "contamination": 0.05,
        "random_state": args.seed,
    }
    else:
        model_params = {}

    result = exp.run(
    X_train,
    X_test,
    model_name=args.model,
    model_params=model_params,
)

    print("Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
