"""The real packaged model (skipped if service/model_files is missing)."""

import os

import pytest

pytest.importorskip("onnxruntime")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "service", "model_files")
if not os.path.exists(os.path.join(MODEL_DIR, "model.int8.onnx")):
    pytest.skip("model not packaged (run scripts/package_model.sh)", allow_module_level=True)

from service.predictor import Predictor  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture(scope="module")
def predictor():
    return Predictor(MODEL_DIR)


def test_training_log_examples(predictor):
    """The two reload examples from training give the same calls."""
    phd = predictor.predict({
        "llm-generated-program": "Computer Science",
        "llm-generated-university": "Johns Hopkins University",
        "comments": "Applying for the AI track, have research experience and one publication.",
        "Degree": "PhD", "US/International": "International",
        "GPA": 3.87, "GRE": 168, "GRE V": 160, "GRE AW": 4.5,
    })
    masters = predictor.predict({
        "llm-generated-program": "Public Health",
        "llm-generated-university": "University of Michigan",
        "comments": "Unknown", "Degree": "Master's", "US/International": "American",
        "GPA": None, "GRE": None, "GRE V": None, "GRE AW": None,
    })
    # fp32 PyTorch gave 0.144 and 0.947; int8 should land close by.
    assert phd["label"] == "Rejected" and abs(phd["p_accepted"] - 0.144) < 0.05
    assert masters["label"] == "Accepted" and abs(masters["p_accepted"] - 0.947) < 0.05


def test_outcome_words_are_redacted(predictor):
    result = predictor.predict({"comments": "I was accepted last year", "Degree": "PhD"})
    assert "[redacted]" in result["input_text"]
    assert "accepted" not in result["input_text"].lower().replace("[redacted]", "")
