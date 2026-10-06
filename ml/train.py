"""Training pipeline for the congestion-prediction model.

Algorithm choice
----------------
A **Random Forest** (ensemble of decision trees) is used because

* it trains in well under a second on 5 000 rows – no GPU needed,
* it handles mixed scales (latency in ms, loss in %, utilisation in [0,1])
  without feature normalisation,
* it reports *feature importances*, which is exactly what the report needs to
  justify the feature set,
* it is robust to the noise we inject into the synthetic data,
* the decision-tree structure is explainable to a viva panel, unlike a deep
  network.

Deep learning is intentionally **not** used: there is no sequential structure
large enough to justify it, and it would make the project much harder to
explain.

Run ``python -m ml.train`` to (re)train and store ``models/congestion_model.joblib``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config import (
    DATASET_CSV,
    ML_FEATURES,
    ML_MIN_ACCEPTABLE_ACCURACY,
    ML_MODEL_PATH,
    ML_MODEL_REPORT_PATH,
    ML_RANDOM_STATE,
    ML_TARGET,
    ML_TEST_SIZE,
)
from ml.dataset import ensure_dataset
from ml.model import CLASS_LABELS, validate_dataset_columns


class TrainingError(RuntimeError):
    """Raised when the model does not reach the minimum required accuracy."""


def build_model(random_state: int = ML_RANDOM_STATE) -> RandomForestClassifier:
    """Create the classifier with explainable, deterministic settings."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def train_model(
    dataset_path: Path = DATASET_CSV,
    model_path: Path = ML_MODEL_PATH,
    report_path: Path = ML_MODEL_REPORT_PATH,
    save: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train, evaluate and (optionally) persist the congestion model.

    Returns a report dictionary containing accuracy, per-class precision /
    recall / F1, the confusion matrix and the feature importances.
    """
    frame = pd.read_csv(dataset_path) if dataset_path.exists() else ensure_dataset(dataset_path)
    validate_dataset_columns(frame.columns)

    features = frame[ML_FEATURES]
    labels = frame[ML_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
        stratify=labels,
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = float(accuracy_score(y_test, predictions))
    if accuracy < ML_MIN_ACCEPTABLE_ACCURACY:
        raise TrainingError(
            f"Model accuracy {accuracy:.3f} is below the required "
            f"{ML_MIN_ACCEPTABLE_ACCURACY:.2f}"
        )

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    importances = {
        name: float(value)
        for name, value in sorted(
            zip(ML_FEATURES, model.feature_importances_),
            key=lambda pair: -pair[1],
        )
    }

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "dataset_size": int(len(frame)),
        "classes": [CLASS_LABELS[int(c)] for c in sorted(labels.unique())],
        "confusion_matrix": matrix.tolist(),
        "feature_importances": importances,
        "classification_report": {
            str(key): {m: float(v) for m, v in value.items()}
            if isinstance(value, dict) else value
            for key, value in report.items()
        },
        "class_distribution": {
            str(int(k)): int(v) for k, v in labels.value_counts().sort_index().items()
        },
        "parameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_leaf": model.min_samples_leaf,
            "test_size": ML_TEST_SIZE,
            "random_state": ML_RANDOM_STATE,
        },
    }

    if save:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if verbose:
            print(f"Model saved to {model_path}")
            print(f"Report saved to {report_path}")
            print(f"Accuracy: {accuracy:.4f}")

    return result


def load_training_report(path: Path = ML_MODEL_REPORT_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover - manual utility
    summary = train_model(verbose=True)
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print("Feature importances:", summary["feature_importances"])
