"""Re-evaluate the fine-tuned DistilBERT model on its held-out test split.

Rebuilds the exact test split used in training (same cleaning, unified-text
template, stratified 80/20 split with random_state=42), then scores:

1. the original PyTorch model (``model/saved_model``),
2. an ONNX export of it (fp32),
3. a dynamically int8-quantized ONNX model (the candidate for serving).

The ONNX models are scored with ``onnxruntime`` + ``tokenizers`` only -- the
same lightweight stack the web service will use -- so the reported numbers
reflect what visitors actually get.

Usage (from the repo root)::

    .venv/bin/python scripts/evaluate.py

Writes ``artifacts/model.onnx``, ``artifacts/model.int8.onnx`` and
``artifacts/eval_results.json``.
"""

import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

# pylint: disable=wrong-import-position
from train_model import (  # noqa: E402
    MAX_SEQ_LENGTH, add_unified_text_column, load_and_prepare_dataset,
    split_dataset,
)

DATA_PATH = os.path.join(ROOT, "data", "cleaned_gradcafe.json")
MODEL_DIR = os.path.join(ROOT, "model", "saved_model")
ART_DIR = os.path.join(ROOT, "artifacts")
ONNX_FP32 = os.path.join(ART_DIR, "model.onnx")
ONNX_INT8 = os.path.join(ART_DIR, "model.int8.onnx")
BATCH = 64


def build_test_split():
    """Return the held-out test DataFrame, exactly as training built it."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_and_prepare_dataset(DATA_PATH)
        df = add_unified_text_column(df)
        _, test_df = split_dataset(df)
    return test_df.reset_index(drop=True)


def metrics(y_true, y_pred):
    """Classification metrics with Accepted (1) as the positive class."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix_[acc,rej]x[pred_acc,pred_rej]": cm.tolist(),
    }


def softmax(logits):
    """Row-wise softmax for a 2-D logits array."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def eval_torch(texts):
    """P(Accepted) for each text using the original PyTorch model."""
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], truncation=True, padding=True,
                      max_length=MAX_SEQ_LENGTH, return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
    return np.concatenate(probs)


def export_onnx():
    """Export the saved model to ONNX and make an int8-quantized copy."""
    os.makedirs(ART_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    enc = tok("Program: Computer Science", return_tensors="pt")
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"]), ONNX_FP32,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(ONNX_FP32, ONNX_INT8, weight_type=QuantType.QInt8)


def load_light_tokenizer():
    """The serving tokenizer: ``tokenizers`` only, no transformers/torch."""
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.enable_truncation(MAX_SEQ_LENGTH)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    return tok


def eval_onnx(path, texts, tok):
    """P(Accepted) for each text using an ONNX model via onnxruntime."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    probs = []
    for i in range(0, len(texts), BATCH):
        encs = tok.encode_batch(texts[i:i + BATCH])
        feeds = {
            "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encs],
                                       dtype=np.int64),
        }
        probs.append(softmax(sess.run(None, feeds)[0])[:, 1])
    return np.concatenate(probs)


def main():
    """Run all three evaluations and write a JSON summary."""
    torch.set_num_threads(os.cpu_count() or 1)
    test_df = build_test_split()
    texts = test_df["model_input"].tolist()
    y_true = test_df["label"].to_numpy()
    print(f"Test rows: {len(test_df)}  "
          f"(Accepted={int(y_true.sum())}, Rejected={int((1 - y_true).sum())})")

    # Sanity check: the lightweight tokenizer must match the HF tokenizer.
    light = load_light_tokenizer()
    hf_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    sample = texts[:200]
    mismatches = sum(
        light.encode(t).ids[:sum(light.encode(t).attention_mask)]
        != hf_tok(t, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
        for t in sample
    )
    print(f"Tokenizer mismatches (light vs HF) on {len(sample)} rows: {mismatches}")

    results = {"test_rows": len(test_df),
               "majority_baseline": round(float(1 - y_true.mean()), 4)}

    start = time.time()
    p_torch = eval_torch(texts)
    print(f"PyTorch eval: {time.time() - start:.0f}s")
    results["pytorch_fp32"] = metrics(y_true, (p_torch >= 0.5).astype(int))

    export_onnx()
    for name, path in (("onnx_fp32", ONNX_FP32), ("onnx_int8", ONNX_INT8)):
        start = time.time()
        p_onnx = eval_onnx(path, texts, light)
        print(f"{name} eval: {time.time() - start:.0f}s")
        pred = (p_onnx >= 0.5).astype(int)
        results[name] = metrics(y_true, pred)
        results[name]["size_mb"] = round(os.path.getsize(path) / 1e6, 1)
        results[name]["agreement_with_pytorch"] = round(
            float((pred == (p_torch >= 0.5)).mean()), 4)
        results[name]["mean_abs_prob_diff"] = round(
            float(np.abs(p_onnx - p_torch).mean()), 4)

    with open(os.path.join(ART_DIR, "eval_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
