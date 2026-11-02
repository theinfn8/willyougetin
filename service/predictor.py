"""Load the int8 ONNX model once and run single-applicant predictions.

Serving needs only ``onnxruntime``, ``tokenizers`` and ``numpy`` -- no
PyTorch or transformers -- which keeps the process around 200 MB.

The model directory must contain the files written by
``scripts/package_model.sh``: ``model.int8.onnx``, ``tokenizer.json`` and
``metadata.json``.
"""

import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .textfmt import build_unified_text

MODEL_FILE = "model.int8.onnx"


class Predictor:
    """The deployed admissions classifier."""

    def __init__(self, model_dir: str, threads: int = 1):
        with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        self.labels = self.metadata["label_mapping"]  # {"0": "Rejected", ...}

        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(self.metadata["max_seq_length"])
        self.tokenizer.no_padding()  # one input at a time: nothing to pad

        options = ort.SessionOptions()
        # The Linode has a single shared vCPU; more threads only add overhead.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            os.path.join(model_dir, MODEL_FILE), options,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, applicant: dict) -> dict:
        """Predict one applicant (dict keyed by training-data field names).

        :return: ``{"label", "p_accepted", "confidence", "input_text"}``;
            ``input_text`` is exactly what the model read.
        """
        text = build_unified_text(applicant)
        encoding = self.tokenizer.encode(text)
        feeds = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        logits = self.session.run(None, feeds)[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        index = int(probs.argmax())
        return {
            "label": self.labels[str(index)],
            "p_accepted": float(probs[1]),
            "confidence": float(probs[index]),
            "input_text": text,
        }
