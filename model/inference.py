"""
Module 13: Scale & LM Deployment
inference.py

Standalone inference helpers: load a previously fine-tuned admissions
classifier (saved by train_model.py's Step 6) and run predictions on new
applicant data, without importing anything training-related and without
retraining.

This is the same code path the Flask "Will You Get In?" page uses: build a
unified applicant text (same template as training), tokenize it, and get a
predicted class + confidence back.
"""

import pandas as pd

from train_model import (
    SAVE_DIR,
    DEVICE,
    build_unified_text,
    load_saved_model,
    predict_single,
)


def predict_from_applicant_dict(model, tokenizer, applicant: dict, metadata: dict,
                                 device=DEVICE):
    """Run a prediction for one applicant, given as a plain field dict.

    This is the entry point the Flask app should call: it takes raw form
    field values (as a dict), builds the same unified text representation
    used during training, and returns a prediction.

    :param model: A reloaded classification model (from
        :func:`train_model.load_saved_model`).
    :param tokenizer: The matching tokenizer.
    :param applicant: A dict of raw field values, using the same keys as
        the training data (e.g. ``"llm-generated-program"``,
        ``"llm-generated-university"``, ``"comments"``, ``"Degree"``,
        ``"US/International"``, ``"GPA"``, ``"GRE"``, ``"GRE V"``,
        ``"GRE AW"``). Missing/unknown fields can be omitted or set to
        ``float("nan")`` (numeric) / ``"Unknown"`` (text/categorical).
    :type applicant: dict
    :param metadata: The metadata dict from
        :func:`train_model.load_saved_model`.
    :param device: Device to run inference on.
    :return: ``(predicted_label, confidence, p_accepted)`` -- see
        :func:`train_model.predict_single`.
    :rtype: tuple[str, float, float]
    """
    text = build_unified_text(pd.Series(applicant))
    return predict_single(model, tokenizer, text, metadata, device)


if __name__ == "__main__":
    print("Loading saved model (no retraining)...")
    model, tokenizer, metadata = load_saved_model(SAVE_DIR, DEVICE)
    print(f"Model reloaded successfully from '{SAVE_DIR}/'.")
    print("-" * 70)

    # Two example applicants, run through the reloaded model.
    example_applicants = [
        {
            "llm-generated-program": "Computer Science",
            "llm-generated-university": "Johns Hopkins University",
            "comments": "Applying for the AI track, have research experience and one publication.",
            "Degree": "PhD",
            "US/International": "International",
            "GPA": 3.87,
            "GRE": 168,
            "GRE V": 160,
            "GRE AW": 4.5,
        },
        {
            "llm-generated-program": "Public Health",
            "llm-generated-university": "University of Michigan",
            "comments": "Unknown",
            "Degree": "Master's",
            "US/International": "American",
            "GPA": float("nan"),
            "GRE": float("nan"),
            "GRE V": float("nan"),
            "GRE AW": float("nan"),
        },
    ]

    for i, applicant in enumerate(example_applicants, start=1):
        label, confidence, p_accepted = predict_from_applicant_dict(
            model, tokenizer, applicant, metadata
        )
        print(f"Example {i}: predicted={label}, confidence={confidence:.3f}, "
              f"P(Accepted)={p_accepted:.3f}")
