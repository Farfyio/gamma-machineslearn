

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


import numpy as np
from numpy.ma import exp
from sklearn.model_selection import train_test_split

from gamma_machineslearn import AnomalyDetectionExperiment
from gamma_machineslearn.visualization import save_confusion_matrix
from gamma_machineslearn.data import load_csv, load_spectra_directory

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
        X, y = load_spectra_directory(args.data)
        #Print out the shape of the data and the # of pure and impure spectra
        print("X shape:", X.shape)
        print("Labels shape:", y.shape)
        print("Pure spectra:", np.sum(y==1))
        print("Impure spectra:", np.sum(y==-1))
        #Sepeerate the two sides
        pure_spectra = X[y == 1]
        impure_spectra = X[y == -1]
        #Split the pure psectra into a 80-20 split for the training and testing
        X_train, X_pure_test = train_test_split(
            pure_spectra,
            test_size = 0.3,
            random_state = args.seed if args.seed else None,
            shuffle = True
        )
        #Combien the 20 from the pure spectra wiht the rest of the impure to use as the test set
        X_test = np.vstack((X_pure_test, impure_spectra))
        y_test = np.conctenate((
            np.ones(len(X_pure_test), dtype = int),
            -np.ones(len(impure_spectra), dtyype = int)

        ))
        #Shuffles the test set so it is not ordered with how it was create above this
        rng = np.random.default_rng(args.seed if args.seed else None)
        indices = rng.permutation(len(X_test))
        X_test = X_test[indices]
        y_test = y_test[indices]


    else:
        # Synthetic data: standard normal features
        X_train = np.random.randn(args.n_train, args.features)
        X_test = np.random.randn(args.n_test, args.features)
        y_test = None  # No labels for synthetic data

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
    y_test=y_test
)
    

    print("Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
