"""The service must build model input text exactly as training did.

Needs the training environment (pandas, torch) and the dataset, so it is
skipped where those aren't available, e.g. in the serving container.
"""

import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "cleaned_gradcafe.json")

pd = pytest.importorskip("pandas")
pytest.importorskip("torch")
if not os.path.exists(DATA):
    pytest.skip("dataset not present (it is not committed)", allow_module_level=True)

sys.path.insert(0, os.path.join(ROOT, "model"))
import train_model  # noqa: E402  pylint: disable=wrong-import-position

from service.textfmt import build_unified_text  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture(scope="module")
def prepared():
    return train_model.load_and_prepare_dataset(train_model.DATA_PATH)


def _as_plain_dict(row):
    """A pandas row as the plain dict the service receives (NaN -> None)."""
    return {key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in row.items()}


def test_every_training_row_renders_identically(prepared):
    mismatches = []
    for _, row in prepared.iterrows():
        expected = train_model.build_unified_text(row)
        actual = build_unified_text(_as_plain_dict(row))
        if expected != actual:
            mismatches.append((expected, actual))
    assert not mismatches, f"{len(mismatches)} rows differ; first: {mismatches[0]}"


def test_constants_match_training():
    from service import textfmt  # pylint: disable=import-outside-toplevel
    assert textfmt.UNIFIED_INPUT_TEMPLATE == train_model.UNIFIED_INPUT_TEMPLATE
    assert textfmt.LEAK_PATTERN.pattern == train_model.LEAK_PATTERN.pattern
    assert textfmt.NOT_REPORTED == train_model.NOT_REPORTED
    assert textfmt.MISSING_PLACEHOLDER == train_model.MISSING_PLACEHOLDER
