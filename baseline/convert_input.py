"""
convert_input.py

STANDALONE HELPER SCRIPT -- NOT part of the Module 12 assignment submission.

Purpose:
    The existing cleaned_gradcafe.json (produced in earlier modules) is a
    single JSON array of records with ~30 fields per record (Program,
    University, comments, date_added, url, llm-generated-*, and so on),
    most of which Assignment 12 doesn't need at all. Assignment 12's
    Step 1 expects a much smaller JSON Lines file (one JSON object per
    line) with just these seven fields:

        gpa, gre, gre_v, gre_aw   -> numeric fields, provided as STRINGS
        masters_or_phd            -> "Masters" or "PhD" (or other degree names)
        citizenship               -> "International" or "American"/"Local" (or other)
        applicant_status          -> "Accepted" / "Rejected" / etc.

    This script selects out just the needed raw columns (GPA, GRE, GRE V,
    GRE AW, Degree, US/International, outcome) into their own DataFrame
    -- dropping everything else -- before renaming/reshaping into the
    Assignment 12 schema and writing it out. Selecting only the needed
    columns up front means the resulting .jsonl file is dramatically
    smaller than the original cleaned_gradcafe.json.

    This script performs that column selection and field renaming/
    reshaping ONLY. It does NOT filter rows (e.g. it does not drop
    Waitlisted/Interviewed rows or PsyD rows) and it does NOT do any of
    the Assignment 12 preprocessing (that filtering/casting/feature-
    construction work happens inside neural_network.py itself, per the
    assignment spec).

Usage:
    python convert_input.py \
        --input cleaned_gradcafe.json \
        --output gradcafe_module12_input.jsonl
"""

import argparse
import json

import pandas as pd


# Only these raw columns from cleaned_gradcafe.json are needed for
# Assignment 12; everything else (comments, url, llm-generated-*, etc.)
# gets dropped by selecting just this list into its own DataFrame.
NEEDED_RAW_COLUMNS = [
    "GPA",
    "GRE",
    "GRE V",
    "GRE AW",
    "Degree",
    "US/International",
    "outcome",
]

# Map the old "Degree" values to the vocabulary Assignment 12 expects.
# Values not in this map are passed through unchanged.
DEGREE_RENAME = {
    "Master's": "Masters",
}

# Renames the needed raw columns directly to Assignment 12's expected
# field names (all except Degree, which also needs its VALUES remapped
# via DEGREE_RENAME, handled separately below).
COLUMN_RENAME = {
    "GRE V": "gre_v",
    "GRE AW": "gre_aw",
    "GPA": "gpa",
    "GRE": "gre",
    "US/International": "citizenship",
    "outcome": "applicant_status",
}


def select_needed_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select just the columns Assignment 12 needs out of the full,
    ~30-column cleaned_gradcafe.json DataFrame, into their own DataFrame.

    This is the step that shrinks the file: everything not in
    NEEDED_RAW_COLUMNS (Program, University, comments, date_added, url,
    llm-generated-program, llm-generated-university, and a dozen other
    derived/bookkeeping columns) is dropped here rather than carried
    through to the output file.
    """
    return raw_df[NEEDED_RAW_COLUMNS].copy()


def reshape_to_module12_schema(trimmed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename/reshape the trimmed DataFrame (output of select_needed_columns)
    into the exact schema Assignment 12's Step 1 expects.

    Numeric fields (gpa, gre, gre_v, gre_aw) are emitted as STRINGS (or
    None when missing), since Assignment 12's Step 1 explicitly performs
    a string-to-float conversion on these fields.
    """
    reshaped = trimmed_df.rename(columns=COLUMN_RENAME).copy()

    reshaped["masters_or_phd"] = trimmed_df["Degree"].replace(DEGREE_RENAME)

    numeric_columns = ["gpa", "gre", "gre_v", "gre_aw"]
    for col in numeric_columns:
        reshaped[col] = reshaped[col].apply(lambda x: None if pd.isna(x) else str(x))

    final_column_order = [
        "gpa",
        "gre",
        "gre_v",
        "gre_aw",
        "masters_or_phd",
        "citizenship",
        "applicant_status",
    ]
    return reshaped[final_column_order]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert cleaned_gradcafe.json into Assignment 12's expected JSONL input format."
    )
    parser.add_argument("--input", required=True, help="Path to the existing cleaned_gradcafe.json")
    parser.add_argument("--output", required=True, help="Path to write the converted .jsonl file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {args.input}")

    raw_df = pd.DataFrame(records)
    print(f"Raw record has {len(raw_df.columns)} columns: {list(raw_df.columns)}")

    # --- Select out just the needed columns into their own DataFrame -------
    trimmed_df = select_needed_columns(raw_df)
    print(f"Selected down to {len(trimmed_df.columns)} needed columns: {list(trimmed_df.columns)}")

    # --- Rename/reshape into the Assignment 12 schema ------------------------
    output_df = reshape_to_module12_schema(trimmed_df)

    output_df.to_json(args.output, orient="records", lines=True)

    print(f"Wrote {len(output_df)} JSON-Lines records to {args.output}")

    # Quick sanity summary so it's obvious the mapping looks right.
    print("applicant_status counts:", output_df["applicant_status"].value_counts().to_dict())
    print("masters_or_phd counts:", output_df["masters_or_phd"].value_counts().to_dict())


if __name__ == "__main__":
    main()
