"""Server-side form validation."""

import pytest

from service.validation import validate

VALID = {"program": "Computer Science", "university": "Johns Hopkins University",
         "degree": "PhD", "citizenship": "International", "gpa": "3.87",
         "gre": "168", "gre_v": "160", "gre_aw": "4.5", "comments": "One publication."}


def test_valid_form_maps_to_training_field_names():
    _, applicant, errors = validate(VALID)
    assert errors == {}
    assert applicant == {
        "llm-generated-program": "Computer Science",
        "llm-generated-university": "Johns Hopkins University",
        "comments": "One publication.",
        "Degree": "PhD", "US/International": "International",
        "GPA": 3.87, "GRE": 168.0, "GRE V": 160.0, "GRE AW": 4.5,
    }


def test_blank_fields_become_training_placeholders():
    _, applicant, errors = validate({"program": "History", "degree": "Master's",
                                     "citizenship": "American"})
    assert errors == {}
    assert applicant["llm-generated-university"] == "Unknown"
    assert applicant["comments"] == "Unknown"
    assert applicant["GPA"] is None and applicant["GRE AW"] is None


def test_whitespace_is_trimmed():
    values, applicant, _ = validate({**VALID, "program": "  Physics  ", "gpa": " 3.5 "})
    assert values["program"] == "Physics"
    assert applicant["GPA"] == 3.5


@pytest.mark.parametrize("field,value", [
    ("gpa", "4.01"), ("gpa", "-1"), ("gpa", "abc"), ("gpa", "3.5abc"),
    ("gpa", "inf"), ("gpa", "nan"), ("gre", "129"), ("gre", "171"),
    ("gre_v", "250"), ("gre_aw", "6.5"), ("gre_aw", "99.99"),
])
def test_bad_numbers_are_rejected(field, value):
    _, applicant, errors = validate({**VALID, field: value})
    assert applicant is None
    assert field in errors


@pytest.mark.parametrize("field,value", [
    ("gpa", "0"), ("gpa", "4.0"), ("gre", "130"), ("gre", "170"), ("gre_aw", "6"),
])
def test_range_edges_are_allowed(field, value):
    assert validate({**VALID, field: value})[2] == {}


@pytest.mark.parametrize("field,value", [
    ("degree", "MBA"), ("degree", "phd"), ("citizenship", "Martian"),
])
def test_tampered_choices_are_rejected(field, value):
    assert field in validate({**VALID, field: value})[2]


@pytest.mark.parametrize("field,limit", [("program", 120), ("comments", 1000)])
def test_long_text_is_rejected(field, limit):
    assert field not in validate({**VALID, field: "x" * limit})[2]
    assert field in validate({**VALID, field: "x" * (limit + 1)})[2]


def test_empty_form_is_rejected():
    _, applicant, errors = validate({"degree": "PhD", "citizenship": "American"})
    assert applicant is None and "form" in errors
