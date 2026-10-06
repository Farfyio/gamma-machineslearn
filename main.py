from __future__ import annotations
import argparse
from typing import Optional
import numpy as np
from numpy.ma import exp
from sklearn.model_selection import train_test_split

from gamma_machineslearn import AnomalyDetectionExperiment
from gamma_machineslearn.visualization import save_confusion_matrix
from gamma_machineslearn.data import load_csv, load_spectra_directory


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny anomaly experiment")
    parser.add_argument(
    "-m",
    "--model",
    default="one_class_svm",
    choices=["one_class_svm", "isolation_forest"],
    help="Anomaly detection model to run"
)
    parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Directory containing combined spectrum CSV files"
)
    parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed for reproducibility"
)


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
            test_size = 0.2,
            random_state = args.seed,
            shuffle = True
        )
        #Combien the 20 from the pure spectra wiht the rest of the impure to use as the test set
        X_test = np.vstack((X_pure_test, impure_spectra))
        y_test = np.concatenate((
            np.ones(len(X_pure_test), dtype = int),
            -np.ones(len(impure_spectra), dtype = int)

        ))
        #Shuffles the test set so it is not ordered with how it was create above this
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(X_test))
        X_test = X_test[indices]
        y_test = y_test[indices]


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
    if y_test is not None:
        save_confusion_matrix(
            result["metrics"]["confusion_matrix"],
            title=f"Confusion Matrix ({args.model})",
            output_path=f"confusion_matrix_{args.model}.png"
        )
    

    print("Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
