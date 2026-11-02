"""Server-side checks for the prediction form.

The browser checks the same rules (``static/predict.js``) for a friendly
experience, but anyone can skip the browser, so this module is the real
gate. It never raises on bad input: it returns field-level error messages
that the page shows next to each field.
"""

import math

DEGREES = ("PhD", "Master's", "PsyD")
CITIZENSHIPS = ("American", "International", "Other")

# (label, min, max). Ranges are the real-world scales, not the training
# data's extremes, which include data-entry junk such as a GPA of 9.99.
NUMERIC_FIELDS = {
    "gpa": ("GPA", 0.0, 4.0),
    "gre": ("GRE Quant", 130, 170),
    "gre_v": ("GRE Verbal", 130, 170),
    "gre_aw": ("GRE Analytical Writing", 0.0, 6.0),
}

TEXT_LIMITS = {"program": 120, "university": 120, "comments": 1000}

# Form field name -> field name the model was trained with.
TRAINING_NAMES = {
    "program": "llm-generated-program",
    "university": "llm-generated-university",
    "comments": "comments",
    "degree": "Degree",
    "citizenship": "US/International",
    "gpa": "GPA",
    "gre": "GRE",
    "gre_v": "GRE V",
    "gre_aw": "GRE AW",
}

DEFAULTS = {
    "program": "", "university": "", "comments": "",
    "degree": "PhD", "citizenship": "American",
    "gpa": "", "gre": "", "gre_v": "", "gre_aw": "",
}


def _parse_number(raw, label, low, high):
    """Return ``(value, error)``; blank means 'not reported' (``None``)."""
    if raw == "":
        return None, None
    try:
        value = float(raw)
    except ValueError:
        return None, f"{label} must be a number, or left blank."
    if not math.isfinite(value) or not low <= value <= high:
        return None, f"{label} must be between {low:g} and {high:g}."
    return value, None


def validate(form) -> tuple:
    """Check submitted form data.

    :param form: A mapping of form field name -> submitted string.
    :return: ``(values, applicant, errors)``. ``values`` is the cleaned
        form values to redisplay; ``applicant`` is the model input (keyed
        by training-data names) or ``None`` when there are errors;
        ``errors`` maps field name (or ``"form"``) to a message.
    """
    values = {name: (form.get(name) or default).strip()
              for name, default in DEFAULTS.items()}
    errors = {}

    for name, limit in TEXT_LIMITS.items():
        if len(values[name]) > limit:
            errors[name] = f"Please keep this under {limit} characters."

    if values["degree"] not in DEGREES:
        errors["degree"] = "Please choose a degree from the list."
    if values["citizenship"] not in CITIZENSHIPS:
        errors["citizenship"] = "Please choose a citizenship option from the list."

    numbers = {}
    for name, (label, low, high) in NUMERIC_FIELDS.items():
        numbers[name], error = _parse_number(values[name], label, low, high)
        if error:
            errors[name] = error

    if not any(values[name] for name in
               ("program", "university", "comments", *NUMERIC_FIELDS)):
        errors["form"] = "Please fill in at least one field before submitting."

    if errors:
        return values, None, errors

    applicant = {}
    for name in ("program", "university", "comments", "degree", "citizenship"):
        # Blank text becomes "Unknown", as it did for missing data in training.
        applicant[TRAINING_NAMES[name]] = values[name] or "Unknown"
    for name in NUMERIC_FIELDS:
        applicant[TRAINING_NAMES[name]] = numbers[name]
    return values, applicant, {}
