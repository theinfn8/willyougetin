"""Shared MLflow setup so every model logs to the same experiment.

By default runs are stored in ``mlflow.db`` (SQLite) at the repo root, so no
tracking server is needed. Set ``MLFLOW_TRACKING_URI`` to log somewhere else,
e.g. ``http://127.0.0.1:8080`` for a running ``mlflow server``.

View the runs with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080
"""

import os

import mlflow

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_NAME = "will-you-get-in"
DEFAULT_TRACKING_URI = f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"
# Absolute, so artifacts land in one place whichever directory a script runs
# from (MLflow otherwise resolves "./mlruns" against the current directory).
ARTIFACT_LOCATION = f"file://{os.path.join(ROOT, 'mlruns')}"


def setup_mlflow() -> None:
    """Point MLflow at the project store and select the shared experiment."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME,
                                 artifact_location=ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT_NAME)


def classification_metrics(y_true, y_pred) -> dict:
    """Test-set metrics with the same names for every model (Accepted = 1)."""
    # Imported here so the module stays light for callers that only need setup.
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        accuracy_score, f1_score, precision_score, recall_score,
    )
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_rows": float(len(y_true)),
    }
