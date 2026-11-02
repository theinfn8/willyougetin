"""Build the model's input text from one applicant -- exactly as in training.

The fine-tuned model only ever saw text produced by
``model/train_model.py:build_unified_text``. If the service built its input
even slightly differently (a changed label, a different number format, an
unredacted outcome word), predictions would quietly get worse with no error.

This is a dependency-free copy of that logic (training needs pandas; the
service does not). ``tests/test_parity.py`` checks the two produce identical
text for every row of the training data, so any drift fails the tests.
"""

import math
import re

MISSING_PLACEHOLDER = "Unknown"
NOT_REPORTED = "Not reported"

UNIFIED_INPUT_TEMPLATE = (
    "Program: {program}\n"
    "University: {university}\n"
    "Comments: {comments}\n"
    "Degree: {degree}\n"
    "Citizenship: {citizenship}\n"
    "GPA: {gpa}\n"
    "GRE Quant: {gre_quant}\n"
    "GRE Verbal: {gre_verbal}\n"
    "GRE AW: {gre_aw}"
)

# Outcome words are redacted from comments in training, so the model never
# learned to key off them; they are redacted here for the same reason.
LEAK_PATTERN = re.compile(
    r"\b(accept(?:ed|ance)?"
    r"|reject(?:ed|ion)?"
    r"|wait[- ]?list(?:ed)?"
    r"|admit(?:ted|tance)?"
    r"|den(?:ied|ial)"
    r"|declined)\b",
    re.IGNORECASE,
)


def sanitize_comment(text):
    """Replace outcome-revealing words with ``[redacted]``."""
    if not isinstance(text, str):
        return text
    return LEAK_PATTERN.sub("[redacted]", text)


def format_numeric(value):
    """``Not reported`` for missing values, else a number without trailing zeros."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NOT_REPORTED
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_unified_text(applicant: dict) -> str:
    """Render one applicant (training-data field names) as model input text."""
    return UNIFIED_INPUT_TEMPLATE.format(
        program=applicant.get("llm-generated-program", MISSING_PLACEHOLDER),
        university=applicant.get("llm-generated-university", MISSING_PLACEHOLDER),
        comments=sanitize_comment(applicant.get("comments", MISSING_PLACEHOLDER)),
        degree=applicant.get("Degree", MISSING_PLACEHOLDER),
        citizenship=applicant.get("US/International", MISSING_PLACEHOLDER),
        gpa=format_numeric(applicant.get("GPA")),
        gre_quant=format_numeric(applicant.get("GRE")),
        gre_verbal=format_numeric(applicant.get("GRE V")),
        gre_aw=format_numeric(applicant.get("GRE AW")),
    )
