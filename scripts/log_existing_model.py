"""Log the already-trained DistilBERT model to MLflow without retraining.

The fine-tuned model in ``model/saved_model`` was trained before experiment
tracking was added. Rather than spend ~30 GPU-minutes retraining just to get
it into MLflow, this script records it from what already exists:

* training settings -- the constants in ``model/train_model.py``,
* training curves  -- ``artifacts/training_history.json``, extracted from the
  original training log,
* test metrics     -- ``artifacts/eval_results.json``, from
  ``scripts/evaluate.py`` (run that first).

It logs two runs, so the MLflow compare view shows what quantization cost:
``distilbert-pytorch`` (the original model) and ``distilbert-onnx-int8`` (the
model the web service serves, with the .onnx file attached as an artifact).

Re-running replaces those two runs rather than duplicating them.

Usage (from the repo root)::

    .venv/bin/python scripts/log_existing_model.py
"""

import json
import os
import sys
import time

import mlflow
from mlflow.entities import Metric

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model"))

# pylint: disable=wrong-import-position
from tracking import setup_mlflow  # noqa: E402
from train_model import (  # noqa: E402
    BATCH_SIZE, LEARNING_RATE, MAX_SEQ_LENGTH, MODEL_NAME, NON_TEXT_FIELDS,
    NUM_EPOCHS, TEXT_FIELDS,
)

ART_DIR = os.path.join(ROOT, "artifacts")
MODEL_DIR = os.path.join(ROOT, "model", "saved_model")
RUN_NAMES = ("distilbert-pytorch", "distilbert-onnx-int8")


def load_json(name):
    """Load a JSON file from ``artifacts/``."""
    with open(os.path.join(ART_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


def common_params():
    """Training settings shared by both runs (it is the same trained model)."""
    return {
        "model_family": "distilbert-finetuned",
        "base_model": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW",
        "loss": "cross-entropy",
        "text_fields": ",".join(TEXT_FIELDS),
        "non_text_fields": ",".join(NON_TEXT_FIELDS),
        "missing_values": "written as 'Not reported' in the input text",
        "split": "80/20 stratified, random_state=42",
    }


def log_training_history(history):
    """Per-step loss and per-epoch test accuracy from the original training."""
    run_id = mlflow.active_run().info.run_id
    now = int(time.time() * 1000)
    metrics = [Metric("train_loss", s["loss"], now, s["global_step"])
               for s in history["step_losses"]]
    for epoch in history["epochs"]:
        metrics += [
            Metric("avg_train_loss", epoch["avg_train_loss"], now, epoch["epoch"]),
            Metric("epoch_test_accuracy", epoch["test_accuracy"], now,
                   epoch["epoch"]),
        ]
    mlflow.tracking.MlflowClient().log_batch(run_id, metrics=metrics)


def test_metrics(result, test_rows, baseline):
    """Map an eval_results.json entry onto the shared metric names."""
    return {
        "test_accuracy": result["accuracy"],
        "test_precision": result["precision"],
        "test_recall": result["recall"],
        "test_f1": result["f1"],
        "test_rows": float(test_rows),
        "majority_baseline": baseline,
    }


def delete_previous_runs(experiment_id):
    """Remove earlier copies of these runs so re-running doesn't duplicate."""
    for name in RUN_NAMES:
        runs = mlflow.search_runs(
            [experiment_id], filter_string=f"tags.mlflow.runName = '{name}'",
            output_format="list")
        for run in runs:
            mlflow.delete_run(run.info.run_id)


def main():
    """Log both DistilBERT runs."""
    results = load_json("eval_results.json")
    history = load_json("training_history.json")

    setup_mlflow()
    experiment = mlflow.get_experiment_by_name("will-you-get-in")
    delete_previous_runs(experiment.experiment_id)

    source_note = ("Trained before tracking was added; metrics from "
                   "scripts/evaluate.py, curves from the original training log.")
    pytorch_size = os.path.getsize(os.path.join(MODEL_DIR, "model.safetensors"))

    with mlflow.start_run(run_name="distilbert-pytorch"):
        mlflow.log_params({**common_params(), "runtime": "pytorch",
                           "precision": "fp32"})
        mlflow.set_tags({"stage": "final", "framework": "pytorch",
                         "source": source_note})
        log_training_history(history)
        mlflow.log_metrics({
            **test_metrics(results["pytorch_fp32"], results["test_rows"],
                           results["majority_baseline"]),
            "model_size_mb": round(pytorch_size / 1e6, 1),
        })
        mlflow.log_artifact(os.path.join(ART_DIR, "eval_results.json"))
        mlflow.log_artifact(os.path.join(ART_DIR, "training_history.json"))
        mlflow.log_artifact(os.path.join(MODEL_DIR, "metadata.json"), "model")

    int8 = results["onnx_int8"]
    with mlflow.start_run(run_name="distilbert-onnx-int8"):
        mlflow.log_params({**common_params(), "runtime": "onnxruntime",
                           "precision": "int8 (dynamic quantization)"})
        mlflow.set_tags({"stage": "serving", "framework": "onnxruntime",
                         "source": source_note})
        log_training_history(history)
        mlflow.log_metrics({
            **test_metrics(int8, results["test_rows"],
                           results["majority_baseline"]),
            "model_size_mb": int8["size_mb"],
            "agreement_with_pytorch": int8["agreement_with_pytorch"],
            "mean_abs_prob_diff": int8["mean_abs_prob_diff"],
        })
        mlflow.log_artifact(os.path.join(ART_DIR, "eval_results.json"))
        mlflow.log_artifact(os.path.join(ART_DIR, "model.int8.onnx"), "model")
        mlflow.log_artifact(os.path.join(MODEL_DIR, "tokenizer.json"), "model")
        mlflow.log_artifact(os.path.join(MODEL_DIR, "metadata.json"), "model")

    print("Logged runs:", ", ".join(RUN_NAMES))


if __name__ == "__main__":
    main()
