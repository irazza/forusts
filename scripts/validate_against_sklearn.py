#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UCR_ROOT = (
    REPO_ROOT.parent / "tsdistances" / "acm_software_package" / "tsdistances" / "Python" / "datasets" / "ucr"
)
DEFAULT_SEED = 42
CLASSIFICATION_DATASETS = ("Adiac", "Beef", "ECG200")
OUTLIER_DATASETS = ("Beef", "ECG200", "Strawberry")


def load_ucr_dataset(root: Path, name: str):
    train = np.loadtxt(root / name / f"{name}_TRAIN.tsv", delimiter="\t")
    test = np.loadtxt(root / name / f"{name}_TEST.tsv", delimiter="\t")
    return train[:, 1:], train[:, 0].astype(int), test[:, 1:], test[:, 0].astype(int)


def run_reference_probe(mode: str, train_path: Path, test_path: Path):
    command = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--example",
        "reference_probe",
        "--",
        mode,
        str(train_path),
        str(test_path),
        "tab",
        str(DEFAULT_SEED),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def minority_as_outlier(y: np.ndarray) -> np.ndarray:
    labels, counts = np.unique(y, return_counts=True)
    minority = labels[np.argmin(counts)]
    return (y == minority).astype(int)


def validate_classification(root: Path):
    rows = []
    for name in CLASSIFICATION_DATASETS:
        train_path = root / name / f"{name}_TRAIN.tsv"
        test_path = root / name / f"{name}_TEST.tsv"
        x_train, y_train, x_test, y_test = load_ucr_dataset(root, name)
        rust = run_reference_probe("rf", train_path, test_path)
        rust_predictions = np.asarray(rust["predictions"], dtype=int)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            bootstrap=True,
            random_state=DEFAULT_SEED,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        sklearn_predictions = model.predict(x_test)

        rows.append(
            {
                "dataset": name,
                "rust_accuracy": accuracy_score(y_test, rust_predictions),
                "sklearn_accuracy": accuracy_score(y_test, sklearn_predictions),
                "prediction_match_rate": float(np.mean(rust_predictions == sklearn_predictions)),
            }
        )
    return rows


def validate_outliers(root: Path):
    rows = []
    for name in OUTLIER_DATASETS:
        train_path = root / name / f"{name}_TRAIN.tsv"
        test_path = root / name / f"{name}_TEST.tsv"
        x_train, y_train, x_test, y_test = load_ucr_dataset(root, name)
        y_test_bin = minority_as_outlier(y_test)
        rust = run_reference_probe("iforest", train_path, test_path)
        rust_scores = np.asarray(rust["scores"], dtype=float)

        model = IsolationForest(
            n_estimators=100,
            max_samples=min(256, len(x_train)),
            max_features=1.0,
            bootstrap=False,
            random_state=DEFAULT_SEED,
            n_jobs=1,
        )
        model.fit(x_train)
        sklearn_scores = -model.score_samples(x_test)

        rows.append(
            {
                "dataset": name,
                "rust_auc": roc_auc_score(y_test_bin, rust_scores),
                "sklearn_auc": roc_auc_score(y_test_bin, sklearn_scores),
                "score_corr": float(np.corrcoef(rust_scores, sklearn_scores)[0, 1]),
            }
        )
    return rows


def main():
    ucr_root = Path(os.environ.get("FORUSTS_UCR_ROOT", DEFAULT_UCR_ROOT))
    if not ucr_root.exists():
        raise SystemExit(f"ucr root not found: {ucr_root}")

    result = {
        "seed": DEFAULT_SEED,
        "ucr_root": str(ucr_root),
        "classification": validate_classification(ucr_root),
        "outlier": validate_outliers(ucr_root),
        "notes": [
            "RandomForestRegressor validation is not included because the repository does not implement a regression forest.",
            "IF_BENCHMARK was not available locally, so the outlier comparison uses the UCR datasets already exercised by the repository's CIsoForest tests.",
        ],
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
