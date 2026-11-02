"""
Module 13: Scale & LM Deployment
train_model.py

Step 1: Load and Prepare the Applicant Dataset

This script fine-tunes a pretrained Hugging Face transformer on the
GradCafe admissions dataset to predict Accepted (1) vs Rejected (0),
using both free-text applicant fields and structured numeric/categorical
fields.

Step 1 of this file handles: loading, filtering, cleaning, and field
selection. Later steps (tokenization, model setup, training loop,
evaluation, saving) will be appended below as we build them out.
"""

import argparse
import re
import os
import sys
import json

import mlflow
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Shared MLflow helpers live at the repo root (one level up).
MODEL_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MODEL_ROOT)
sys.path.insert(0, REPO_ROOT)
# pylint: disable=wrong-import-position
from tracking import setup_mlflow, classification_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Cleaned GradCafe export (not committed; see README for how to obtain it).
DATA_PATH = os.path.join(REPO_ROOT, "data", "cleaned_gradcafe.json")

# Text fields: free-text / natural-language content
# NOTE: using the LLM-normalized program/university fields instead of the
# raw scraped "Program"/"University" columns -- they're cleaner and more
# consistent, and keeping both raw + normalized versions would just be
# redundant duplicate text.
TEXT_FIELDS = [
    "comments",
    "llm-generated-program",
    "llm-generated-university",
]

# Non-text fields: structured, scalar, or categorical
# NOTE: "term" excluded per instruction -- not used as a model feature.
NON_TEXT_FIELDS = [
    "GPA",
    "GRE",
    "GRE V",
    "GRE AW",
    "Degree",
    "US/International",
]

# Columns that should be coerced to numeric dtype
NUMERIC_FIELDS = ["GPA", "GRE", "GRE V", "GRE AW"]

# Consistent placeholder for missing values across text/categorical columns
MISSING_PLACEHOLDER = "Unknown"


# ---------------------------------------------------------------------------
# Step 1: Load and Prepare the Applicant Dataset
# ---------------------------------------------------------------------------

def load_and_prepare_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load, filter, and clean the applicant dataset for modeling.

    :param path: Path to the cleaned admissions dataset (CSV).
    :type path: str
    :return: A cleaned DataFrame ready for text-unification and tokenization,
        containing only Accepted/Rejected rows with a binary ``label`` column.
    :rtype: pandas.DataFrame
    """
    df = pd.read_json(path)
    original_row_count = len(df)

    # --- Filter to rows with a decided outcome -----------------------------
    # GradCafe's "outcome" column includes Accepted / Rejected / Waitlisted /
    # Interviewed. We only want the two decided outcomes for this task.
    status_col = "outcome"
    df = df[df[status_col].isin(["Accepted", "Rejected"])].copy()

    # --- Remove duplicate applicant rows ------------------------------------
    # Each row is a distinct GradCafe result page, keyed by its "url"
    # (equivalently "id"). Kept for safety in case of re-scrapes.
    df = df.drop_duplicates(subset=["url"], keep="first")

    # --- Drop rows without enough usable information ------------------------
    # A row needs at least one usable text field AND at least one usable
    # non-text field to be a valid model input.
    available_text_fields = [c for c in TEXT_FIELDS if c in df.columns]
    available_non_text_fields = [c for c in NON_TEXT_FIELDS if c in df.columns]

    has_text = df[available_text_fields].notna().any(axis=1)
    has_non_text = df[available_non_text_fields].notna().any(axis=1)
    df = df[has_text & has_non_text].copy()

    # --- Normalize missing values --------------------------------------------
    # Text/categorical columns: fill NaN with a consistent placeholder.
    categorical_like = [
        c for c in (available_text_fields + available_non_text_fields)
        if c not in NUMERIC_FIELDS and c in df.columns
    ]
    df[categorical_like] = df[categorical_like].fillna(MISSING_PLACEHOLDER)

    # Also normalize common "empty" string variants (e.g. "", "nan", "N/A")
    # to the same placeholder so downstream text formatting stays consistent.
    df[categorical_like] = df[categorical_like].replace(
        to_replace=["", "nan", "None", "N/A", "n/a"], value=MISSING_PLACEHOLDER
    )

    # --- Convert numeric columns to numeric dtype ---------------------------
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Create the binary target variable -----------------------------------
    df["label"] = (df[status_col] == "Accepted").astype(int)

    filtered_row_count = len(df)
    accepted_count = int((df["label"] == 1).sum())
    rejected_count = int((df["label"] == 0).sum())
    fields_used = available_text_fields + available_non_text_fields

    # --- Required output ------------------------------------------------------
    print("=" * 70)
    print("Step 1: Dataset Loading, Filtering, and Field Selection")
    print("=" * 70)
    print(f"Original row count:      {original_row_count}")
    print(f"Filtered row count:      {filtered_row_count}")
    print(f"Accepted count:          {accepted_count}")
    print(f"Rejected count:          {rejected_count}")
    print(f"Text fields used ({len(available_text_fields)}):     {available_text_fields}")
    print(f"Non-text fields used ({len(available_non_text_fields)}): {available_non_text_fields}")
    print(f"Full field list ({len(fields_used)}): {fields_used}")
    print("-" * 70)
    print("Preview of cleaned DataFrame:")
    preview_cols = fields_used + ["label"]
    print(df[preview_cols].head(5).to_string())
    print("=" * 70)

    return df


# ---------------------------------------------------------------------------
# Step 2: Convert Each Applicant into a Unified Model Input
# ---------------------------------------------------------------------------

# The exact template used for every applicant. Field values are substituted
# in at render time; missing values use consistent, human-readable
# placeholders rather than being silently dropped.
#
# NOTE ON MISSING NUMERIC VALUES: unlike the earlier two-layer NN (where we
# mean-filled GPA/GRE because a plain float has no way to express "unknown"),
# here we're writing directly into text that gets tokenized, so we can just
# say so. Mean-filling would insert a fabricated, identical-looking fact
# ("GPA: 3.5") for every applicant who never reported a GPA, which both
# misrepresents the applicant and destroys a potentially real signal: many
# programs are GRE-optional or don't require GPA reporting, so *missingness
# itself* can be predictive. We preserve that by writing "Not reported"
# instead of imputing a number.
#
# NOTE ON THE LABEL: the "Prediction target" line shown in the assignment
# example is illustrative only. It is NOT included in the unified text input
# built below -- the label lives separately in df["label"] and is used only
# as the supervision target during training/evaluation. Including it in the
# model's input text would leak the answer into the input, which is exactly
# the same input format we must reuse (unchanged) for live inference on the
# "Will You Get In?" page, where the label obviously isn't known.
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

NOT_REPORTED = "Not reported"

# --- Comment leakage sanitization -------------------------------------------
# Applicants frequently narrate their own outcome directly in the free-text
# comment (e.g. "Rejected for Master's Thesis...", "Accepted off of
# waitlist..."). If left in, the model can learn to key off these words
# instead of learning generalizable signal -- and that shortcut doesn't
# transfer to the deployed "Will You Get In?" page, where a user fills out
# the form *before* knowing their outcome and would never type "I got
# rejected" into the comments box. We redact this outcome-revealing language
# before building the unified text input so training and live-inference
# comments come from the same distribution.
LEAK_PATTERN = re.compile(
    r"\b(accept(?:ed|ance)?"
    r"|reject(?:ed|ion)?"
    r"|wait[- ]?list(?:ed)?"
    r"|admit(?:ted|tance)?"
    r"|den(?:ied|ial)"
    r"|declined)\b",
    re.IGNORECASE,
)


def sanitize_comment(text) -> str:
    """Redact outcome-revealing words from a free-text applicant comment.

    :param text: The raw comment string (already placeholder-filled by
        :func:`load_and_prepare_dataset`, so never truly null here).
    :type text: str
    :return: The comment with any decision-revealing words (accepted,
        rejected, waitlisted, admitted, denied, declined, etc.) replaced by
        ``"[redacted]"``, leaving the rest of the prose intact.
    :rtype: str
    """
    if not isinstance(text, str):
        return text
    return LEAK_PATTERN.sub("[redacted]", text)


def _format_numeric(value) -> str:
    """Render a possibly-missing numeric value for the unified text template.

    :param value: A numeric value, or NaN/None if not reported.
    :return: ``"Not reported"`` if missing, otherwise a clean string with
        no unnecessary trailing zeros (e.g. ``160`` instead of ``160.0``,
        ``3.87`` instead of ``3.8700000000000001``).
    :rtype: str
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NOT_REPORTED
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_unified_text(row: pd.Series) -> str:
    """Convert a single applicant row into one unified text model input.

    Uses the LLM-normalized program/university fields (cleaner and more
    consistent than the raw scraped ``Program``/``University`` text), the
    raw applicant ``comments``, and the structured fields for term, degree,
    citizenship, GPA, and the three GRE subscores.

    :param row: A row from the cleaned applicant DataFrame.
    :type row: pandas.Series
    :return: The unified, human-readable text representation for this
        applicant, following ``UNIFIED_INPUT_TEMPLATE``. Does not include
        the label.
    :rtype: str
    """
    return UNIFIED_INPUT_TEMPLATE.format(
        program=row.get("llm-generated-program", MISSING_PLACEHOLDER),
        university=row.get("llm-generated-university", MISSING_PLACEHOLDER),
        comments=sanitize_comment(row.get("comments", MISSING_PLACEHOLDER)),
        degree=row.get("Degree", MISSING_PLACEHOLDER),
        citizenship=row.get("US/International", MISSING_PLACEHOLDER),
        gpa=_format_numeric(row.get("GPA")),
        gre_quant=_format_numeric(row.get("GRE")),
        gre_verbal=_format_numeric(row.get("GRE V")),
        gre_aw=_format_numeric(row.get("GRE AW")),
    )


def add_unified_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``model_input`` column holding each applicant's unified text.

    :param df: The cleaned applicant DataFrame from
        :func:`load_and_prepare_dataset`.
    :type df: pandas.DataFrame
    :return: The same DataFrame with an added ``model_input`` column.
    :rtype: pandas.DataFrame
    """
    df = df.copy()

    leaked_rows = int(df["comments"].apply(
        lambda c: isinstance(c, str) and bool(LEAK_PATTERN.search(c))
    ).sum())

    df["model_input"] = df.apply(build_unified_text, axis=1)

    # --- Required output ------------------------------------------------------
    print("=" * 70)
    print("Step 2: Unified Applicant Text Representation")
    print("=" * 70)
    print(
        f"Comments containing outcome-revealing language (redacted): "
        f"{leaked_rows} / {len(df)} ({leaked_rows / len(df) * 100:.1f}%)"
    )
    print("-" * 70)
    print("Template used for every applicant:")
    print("-" * 70)
    print(UNIFIED_INPUT_TEMPLATE)
    print("-" * 70)
    print("Three sample model inputs from the training dataset:")
    for i, sample in enumerate(df["model_input"].head(3), start=1):
        print(f"\n--- Sample {i} ---")
        print(sample)
    print("=" * 70)

    return df


# ---------------------------------------------------------------------------
# Step 3: Split the Data into Training and Testing Sets
# ---------------------------------------------------------------------------

TEST_SIZE = 0.2
RANDOM_STATE = 42

TRAIN_TEST_SPLIT_EXPLANATION = (
    "Train/test separation matters because it is the only way to estimate how "
    "the model will perform on data it has never seen. A model can always fit "
    "its training data well by memorizing patterns specific to those examples."
    "The test set is what tells us whether it has learned something that "
    "generalizes, versus just memorized entries. This is critical when the "
    "trained model will be deployed on a public-facing webpage ('Will You Get "
    "In?'). Every real user who submits via the form is, by definition, a case "
    "the model has never seen during training. If we evaluated only on training "
    "data, we would have no honest way to know how the model will perform for "
    "live users. We would be reporting how well it memorized the past, not how "
    "reliable its predictions will be for someone using the site tomorrow."
)


def split_dataset(df: pd.DataFrame, test_size: float = TEST_SIZE,
                   random_state: int = RANDOM_STATE):
    """Split the cleaned, unified-text dataset into train and test sets.

    Uses an 80/20 split, stratified by ``label`` so both sets preserve the
    overall Accepted/Rejected ratio, unless the dataset is too small to
    support stratification (in which case it falls back to a plain split).

    :param df: The cleaned DataFrame with a ``model_input`` column (from
        :func:`add_unified_text_column`) and a ``label`` column.
    :type df: pandas.DataFrame
    :param test_size: Fraction of rows reserved for testing.
    :type test_size: float
    :param random_state: Seed for reproducibility.
    :type random_state: int
    :return: ``(train_df, test_df)``.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=df["label"],
        )
    except ValueError:
        # Falls back to a non-stratified split only if the dataset is too
        # small / too imbalanced for stratification to be possible.
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    def _class_balance(subset: pd.DataFrame) -> str:
        counts = subset["label"].value_counts().sort_index()
        accepted = int(counts.get(1, 0))
        rejected = int(counts.get(0, 0))
        total = len(subset)
        return (
            f"Accepted={accepted} ({accepted / total * 100:.1f}%), "
            f"Rejected={rejected} ({rejected / total * 100:.1f}%)"
        )

    # --- Required output ------------------------------------------------------
    print("=" * 70)
    print("Step 3: Train/Test Split")
    print("=" * 70)
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size:     {len(test_df)}")
    print(f"Training set class balance: {_class_balance(train_df)}")
    print(f"Test set class balance:     {_class_balance(test_df)}")
    print("-" * 70)
    print("Why train/test separation matters:")
    print(TRAIN_TEST_SPLIT_EXPLANATION)
    print("=" * 70)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Step 4: Fine-Tune a Pretrained PyTorch Language Model
# ---------------------------------------------------------------------------

# --- Model / tokenizer choice -----------------------------------------------
# DistilBERT (distilbert-base-uncased) was chosen over full BERT/RoBERTa/
# ALBERT because it's a distilled version of BERT: ~40% fewer parameters and
# ~60% faster at inference/training, while retaining roughly 97% of BERT's
# language understanding on downstream tasks. For a single binary
# classification head on top of relatively short, template-structured
# admissions text (not open-ended long-form prose), that tradeoff is very
# favorable -- we don't need BERT's full capacity to separate "Accepted" vs
# "Rejected" from this input, and the smaller model makes fine-tuning on
# ~31K examples practical on ordinary (non-multi-GPU) hardware. Its
# tokenizer is the matching pretrained WordPiece tokenizer (uncased, since
# admissions text has no case-sensitive signal we care about), which we load
# directly via AutoTokenizer / AutoModelForSequenceClassification so the
# tokenizer and model vocabulary stay in sync automatically.
MODEL_NAME = "distilbert-base-uncased"

# --- Tokenization settings ---------------------------------------------------
# Max sequence length: 256 tokens comfortably covers the fixed-field portion
# of the unified template plus most Comments text (median real comment is
# ~10 words; even a long comment rarely pushes the whole template past a few
# hundred tokens). Longer comments are truncated rather than dropped, and all
# sequences are padded to this same length so they can be batched as dense
# tensors.
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdmissionsDataset(Dataset):
    """A tokenized PyTorch ``Dataset`` of unified applicant text + labels.

    :param encodings: The batch output of the Hugging Face tokenizer
        (a dict-like object with ``input_ids`` / ``attention_mask`` tensors).
    :param labels: The corresponding list/array of integer labels
        (1 = Accepted, 0 = Rejected).
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load the pretrained tokenizer and sequence-classification model.

    :param model_name: Hugging Face model identifier.
    :type model_name: str
    :return: ``(tokenizer, model)``, with the model already moved to
        :data:`DEVICE` and configured for 2-way (Accepted/Rejected)
        classification.
    :rtype: tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    model.to(DEVICE)
    return tokenizer, model


def build_dataloaders(train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer,
                       max_length: int = MAX_SEQ_LENGTH,
                       batch_size: int = BATCH_SIZE):
    """Tokenize the unified text and build train/test ``DataLoader``s.

    :param train_df: Training split from :func:`split_dataset`, with a
        ``model_input`` text column and a ``label`` column.
    :param test_df: Test split from :func:`split_dataset`.
    :param tokenizer: A Hugging Face tokenizer (from
        :func:`build_model_and_tokenizer`).
    :param max_length: Maximum token sequence length; longer inputs are
        truncated, shorter ones padded to this length.
    :param batch_size: Batch size for both loaders.
    :return: ``(train_loader, test_loader)``.
    :rtype: tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    train_encodings = tokenizer(
        train_df["model_input"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    test_encodings = tokenizer(
        test_df["model_input"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    train_dataset = AdmissionsDataset(train_encodings, train_df["label"].tolist())
    test_dataset = AdmissionsDataset(test_encodings, test_df["label"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate_accuracy(model, data_loader, device=DEVICE) -> float:
    """Compute plain accuracy on a data loader (lightweight post-epoch check).

    Full precision/recall/F1/confusion-matrix evaluation happens separately
    in the Step 5 evaluation section; this is just enough validation-after-
    training to confirm the fine-tuned model is actually learning.

    :param model: The (possibly mid-training) classification model.
    :param data_loader: A ``DataLoader`` of tokenized examples with labels.
    :param device: Device to run inference on.
    :return: Accuracy as a float in ``[0, 1]``.
    :rtype: float
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def train_model(model, train_loader, test_loader,
                 epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE,
                 device=DEVICE, log_every: int = 50):
    """Fine-tune the model with a native PyTorch training loop.

    :param model: A ``AutoModelForSequenceClassification`` instance.
    :param train_loader: Training ``DataLoader``.
    :param test_loader: Test ``DataLoader``, used for a quick post-epoch
        validation accuracy check.
    :param epochs: Number of training epochs.
    :param lr: Learning rate for the optimizer.
    :param device: Device to train on.
    :param log_every: Print a training log line every N steps.
    :return: The fine-tuned model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- Required output: configuration -------------------------------------
    print("=" * 70)
    print("Step 4: Fine-Tuning Configuration")
    print("=" * 70)
    print(f"Model name:          {MODEL_NAME}")
    print(f"Tokenizer name:      {MODEL_NAME} (matching pretrained tokenizer)")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Epochs:              {epochs}")
    print(f"Learning rate:       {lr}")
    print(f"Optimizer:           torch.optim.AdamW")
    print(f"Device:              {device}")
    print("-" * 70)
    print("Training log:")

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % log_every == 0 or step == len(train_loader):
                print(
                    f"  epoch {epoch}/{epochs}  step {step}/{len(train_loader)}  "
                    f"loss={loss.item():.4f}"
                )
                if mlflow.active_run():
                    mlflow.log_metric(
                        "train_loss", loss.item(),
                        step=(epoch - 1) * len(train_loader) + step,
                    )

        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate_accuracy(model, test_loader, device)
        print(f"epoch {epoch}/{epochs} complete -- avg train loss={avg_loss:.4f}, "
              f"test accuracy={val_acc:.4f}")
        if mlflow.active_run():
            mlflow.log_metrics(
                {"avg_train_loss": avg_loss, "epoch_test_accuracy": val_acc},
                step=epoch,
            )
        model.train()

    print("=" * 70)
    return model


# ---------------------------------------------------------------------------
# Step 5: Evaluate the Final Model
# ---------------------------------------------------------------------------

# Held-out test accuracy of the two-layer NumPy NN baseline
# (baseline/neural_network.py, seeded run), for a direct numeric comparison.
EARLIER_TWO_LAYER_NN_TEST_ACCURACY = .7184

N_PROBABILITY_EXAMPLES = 5
N_CORRECT_EXAMPLES = 3
N_INCORRECT_EXAMPLES = 3


def evaluate_model(model, test_loader, device=DEVICE):
    """Run inference over the full test set and collect predictions.

    :param model: The fine-tuned classification model.
    :param test_loader: Test ``DataLoader`` (built with ``shuffle=False``,
        so row order matches the original test DataFrame).
    :param device: Device to run inference on.
    :return: ``(y_true, y_pred, y_prob)`` as NumPy arrays, where ``y_prob``
        is the model's predicted probability of the "Accepted" class (1)
        for each example.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # P(Accepted)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_evaluation_report(y_true, y_pred, y_prob, test_df: pd.DataFrame):
    """Print the full Step 5 evaluation report and interpretation.

    :param y_true: Ground-truth labels for the test set.
    :param y_pred: Model-predicted labels for the test set.
    :param y_prob: Model-predicted probability of "Accepted" (label 1) for
        each test example.
    :param test_df: The original test DataFrame (same row order as
        ``y_true``/``y_pred``/``y_prob``), used to show real examples.
    :type test_df: pandas.DataFrame
    """
    test_df = test_df.reset_index(drop=True)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # [Accepted, Rejected]

    # --- Required output: metrics summary ---------------------------------
    print("=" * 70)
    print("Step 5: Final Model Evaluation")
    print("=" * 70)
    print("Metrics (positive class = Accepted):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 score:  {f1:.4f}")
    print("-" * 70)

    # --- Required output: confusion matrix ----------------------------------
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"{'':>20}{'Pred: Accepted':>18}{'Pred: Rejected':>18}")
    print(f"{'Actual: Accepted':>20}{cm[0][0]:>18}{cm[0][1]:>18}")
    print(f"{'Actual: Rejected':>20}{cm[1][0]:>18}{cm[1][1]:>18}")
    print("-" * 70)

    # --- Class distribution --------------------------------------------------
    true_accepted = int((y_true == 1).sum())
    true_rejected = int((y_true == 0).sum())
    pred_accepted = int((y_pred == 1).sum())
    pred_rejected = int((y_pred == 0).sum())
    total = len(y_true)
    print("Class distribution:")
    print(f"  Actual:    Accepted={true_accepted} ({true_accepted/total*100:.1f}%), "
          f"Rejected={true_rejected} ({true_rejected/total*100:.1f}%)")
    print(f"  Predicted: Accepted={pred_accepted} ({pred_accepted/total*100:.1f}%), "
          f"Rejected={pred_rejected} ({pred_rejected/total*100:.1f}%)")
    print("-" * 70)

    # --- Probability examples -------------------------------------------------
    print(f"Probability examples (first {N_PROBABILITY_EXAMPLES} test rows):")
    for i in range(min(N_PROBABILITY_EXAMPLES, total)):
        actual = "Accepted" if y_true[i] == 1 else "Rejected"
        pred = "Accepted" if y_pred[i] == 1 else "Rejected"
        print(f"  [{i}] actual={actual:<9} predicted={pred:<9} "
              f"P(Accepted)={y_prob[i]:.3f}")
    print("-" * 70)

    # --- Correct / incorrect examples ------------------------------------------
    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    def _print_examples(indices, label, n):
        print(f"{label} (showing up to {n}):")
        for i in indices[:n]:
            actual = "Accepted" if y_true[i] == 1 else "Rejected"
            pred = "Accepted" if y_pred[i] == 1 else "Rejected"
            print(f"  [{i}] actual={actual}, predicted={pred}, "
                f"P(Accepted)={y_prob[i]:.3f}")
            for line in test_df.loc[i, "model_input"].split("\n"):
                print(f"      {line}")
            print()

    _print_examples(correct_idx, "Correctly classified examples", N_CORRECT_EXAMPLES)
    print()
    _print_examples(incorrect_idx, "Incorrectly classified examples", N_INCORRECT_EXAMPLES)
    print("-" * 70)

    # --- Required interpretation -----------------------------------------------
    majority_class_baseline = max(true_accepted, true_rejected) / total
    recall_accepted = recall  # recall for Accepted (pos_label=1)
    recall_rejected = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    bias_note = (
        f"The model recalls Rejected applicants ({recall_rejected:.1%}) "
        f"{'better' if recall_rejected > recall_accepted else 'worse'} than "
        f"Accepted applicants ({recall_accepted:.1%}), predicting Accepted "
        f"{pred_accepted} times vs. {true_accepted} actual Accepted cases in "
        f"the test set. "
        + (
            "This suggests some bias toward predicting the majority class "
            "(Rejected)." if pred_rejected > true_rejected and recall_rejected > recall_accepted
            else "This does not show a strong systematic bias toward either class."
        )
    )

    vs_random_note = (
        f"The majority-class baseline (always predicting the more common "
        f"outcome in the test set) would score {majority_class_baseline:.1%} "
        f"accuracy. The fine-tuned model scores {accuracy:.1%}, which is "
        + ("meaningfully above this baseline, indicating it has learned real "
           "signal beyond guessing the majority class."
           if accuracy > majority_class_baseline + 0.03
           else "close to this baseline, suggesting it may be leaning heavily "
                "on the class imbalance rather than learning strong "
                "discriminative signal.")
    )

    if EARLIER_TWO_LAYER_NN_TEST_ACCURACY is not None:
        nn_note = (
            f"The two-layer NumPy NN scored {EARLIER_TWO_LAYER_NN_TEST_ACCURACY:.1%} "
            f"test accuracy; the fine-tuned transformer scores {accuracy:.1%}, "
            + ("an improvement, consistent with the transformer being able to "
               "use free-text fields (Comments, Program, University) that the "
               "purely numeric NN could not."
               if accuracy > EARLIER_TWO_LAYER_NN_TEST_ACCURACY
               else "not an improvement, suggesting the added text fields and "
                    "model complexity aren't translating into better "
                    "discriminative power on this dataset.")
        )
    else:
        nn_note = (
            "EARLIER_TWO_LAYER_NN_TEST_ACCURACY is not set -- fill in your "
            "two-layer NN's held-out test accuracy from that assignment to "
            "get a direct numeric comparison here."
        )

    reported_pct = test_df[["GPA", "GRE", "GRE V", "GRE AW"]].notna().mean().mean() * 100
    sufficiency_note = (
        f"Only about {reported_pct:.0f}% of GPA/GRE fields are populated on "
        f"average in this test set (many applicants don't report them), and "
        f"labels come from self-reported, unverified GradCafe outcomes with "
        f"no information on program competitiveness, applicant pool size, or "
        f"redacted comment content (~12% of comments needed outcome-language "
        f"redaction, per Step 2). That's enough signal to beat a naive "
        f"majority-class guess, but it falls well short of what a real "
        f"admissions office would need: no verified transcripts, no letters "
        f"of recommendation, no interview data, and no guarantee the sample "
        f"is representative of any single program's applicant pool. This "
        f"model is a reasonable course-project demonstration of text+tabular "
        f"fine-tuning, not a realistic admissions predictor."
    )

    print("Interpretation:")
    print(f"- Bias: {bias_note}")
    print(f"- Vs. random/majority guessing: {vs_random_note}")
    print(f"- Vs. earlier two-layer NN: {nn_note}")
    print(f"- Dataset sufficiency: {sufficiency_note}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 6: Save and Reload the Trained Model
# ---------------------------------------------------------------------------

SAVE_DIR = os.path.join(MODEL_ROOT, "saved_model")
LABEL_MAPPING = {0: "Rejected", 1: "Accepted"}


def save_model_and_artifacts(model, tokenizer, save_dir: str = SAVE_DIR) -> dict:
    """Save the fine-tuned model, tokenizer, and inference metadata to disk.

    Saves everything needed to run inference later without retraining:
    model weights + config (``save_pretrained``), the matching tokenizer,
    and a ``metadata.json`` recording the label mapping and the exact
    preprocessing settings (field lists, max sequence length, placeholders)
    used to build the unified text input -- so ``inference.py`` (or the
    Flask app) can reconstruct the same input format for a live user
    submission.

    :param model: The fine-tuned classification model.
    :param tokenizer: The matching tokenizer.
    :param save_dir: Directory to save into (created if missing).
    :type save_dir: str
    :return: The metadata dict that was written to ``metadata.json``.
    :rtype: dict
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    metadata = {
        "model_name": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "label_mapping": {str(k): v for k, v in LABEL_MAPPING.items()},
        "text_fields": TEXT_FIELDS,
        "non_text_fields": NON_TEXT_FIELDS,
        "missing_placeholder": MISSING_PLACEHOLDER,
        "not_reported_placeholder": NOT_REPORTED,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print("Step 6: Model Saved")
    print("=" * 70)
    print(f"Model weights + config saved to: {save_dir}/")
    print(f"Tokenizer saved to:              {save_dir}/")
    print(f"Metadata saved to:               {save_dir}/metadata.json")
    print(f"  label_mapping:    {metadata['label_mapping']}")
    print(f"  max_seq_length:   {metadata['max_seq_length']}")
    print(f"  text_fields:      {metadata['text_fields']}")
    print(f"  non_text_fields:  {metadata['non_text_fields']}")
    print("=" * 70)

    return metadata


def load_saved_model(save_dir: str = SAVE_DIR, device=DEVICE):
    """Reload a previously saved model, tokenizer, and metadata from disk.

    This loads everything purely from disk -- no training data or training
    loop is touched -- which is the whole point: the model becomes usable
    (e.g. by the Flask "Will You Get In?" page) without ever retraining.

    :param save_dir: Directory previously written by
        :func:`save_model_and_artifacts`.
    :type save_dir: str
    :param device: Device to move the reloaded model to.
    :return: ``(model, tokenizer, metadata)``.
    :rtype: tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    model.to(device)
    model.eval()

    with open(os.path.join(save_dir, "metadata.json")) as f:
        metadata = json.load(f)

    return model, tokenizer, metadata


def predict_single(model, tokenizer, text: str, metadata: dict, device=DEVICE):
    """Run the reloaded model on one unified applicant text string.

    :param model: A reloaded (or in-memory) classification model.
    :param tokenizer: The matching tokenizer.
    :param text: A single unified applicant input string, built the same
        way as during training (see :func:`build_unified_text`).
    :param metadata: The metadata dict from :func:`load_saved_model`.
    :param device: Device to run inference on.
    :return: ``(predicted_label, confidence, p_accepted)`` where
        ``predicted_label`` is ``"Accepted"``/``"Rejected"``, ``confidence``
        is the probability of the predicted class, and ``p_accepted`` is
        specifically the probability of "Accepted" (useful for displaying
        a confidence score regardless of which class was predicted).
    :rtype: tuple[str, float, float]
    """
    max_length = metadata.get("max_seq_length", MAX_SEQ_LENGTH)
    encoding = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    label_mapping = metadata["label_mapping"]
    predicted_label = label_mapping[str(pred_idx)]
    confidence = float(probs[pred_idx].item())
    p_accepted = float(probs[1].item())

    return predicted_label, confidence, p_accepted


def demo_reload_and_predict(save_dir: str = SAVE_DIR, device=DEVICE):
    """Required-output demo: reload the saved model and run 2+ predictions.

    Loads the model fresh from disk (a separate call from the in-memory
    ``model`` used during training/evaluation) and runs inference on two
    brand-new, hand-built applicant examples -- one with full stats
    reported, one with several fields missing -- to show the reloaded
    model handles both the same way live "Will You Get In?" submissions
    would need to be handled.

    :param save_dir: Directory previously written by
        :func:`save_model_and_artifacts`.
    :type save_dir: str
    :param device: Device to run inference on.
    """
    print("=" * 70)
    print("Step 6: Reload and Inference Demonstration")
    print("=" * 70)

    model, tokenizer, metadata = load_saved_model(save_dir, device)
    print(f"Model successfully reloaded from '{save_dir}/' (no retraining).")
    print("-" * 70)

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
            "comments": MISSING_PLACEHOLDER,
            "Degree": "Master's",
            "US/International": "American",
            "GPA": float("nan"),
            "GRE": float("nan"),
            "GRE V": float("nan"),
            "GRE AW": float("nan"),
        },
    ]

    for i, applicant in enumerate(example_applicants, start=1):
        text = build_unified_text(pd.Series(applicant))
        label, confidence, p_accepted = predict_single(
            model, tokenizer, text, metadata, device
        )
        print(f"Example {i} input:")
        print(f"  {text.replace(chr(10), ' | ')}")
        print(f"  --> predicted={label}, confidence={confidence:.3f}, "
              f"P(Accepted)={p_accepted:.3f}")
        print()

    print("=" * 70)


def parse_args():
    """Command-line options; the defaults reproduce the full training run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Train on a random subset (quick smoke tests).")
    parser.add_argument("--save-dir", default=SAVE_DIR,
                        help="Where to save the model (default: %(default)s).")
    parser.add_argument("--run-name", default="distilbert-finetune")
    return parser.parse_args()


def main():
    """Train, evaluate, save, and record the run in MLflow."""
    args = parse_args()
    cleaned_df = load_and_prepare_dataset()
    if args.max_rows:
        cleaned_df = cleaned_df.sample(n=args.max_rows,
                                       random_state=RANDOM_STATE)
    cleaned_df = add_unified_text_column(cleaned_df)
    train_df, test_df = split_dataset(cleaned_df)

    setup_mlflow()
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model_family": "distilbert-finetuned",
            "base_model": MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": args.epochs,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW",
            "loss": "cross-entropy",
            "text_fields": ",".join(TEXT_FIELDS),
            "non_text_fields": ",".join(NON_TEXT_FIELDS),
            "missing_values": "written as 'Not reported' in the input text",
            "split": f"80/20 stratified, random_state={RANDOM_STATE}",
            "runtime": "pytorch",
            "precision": "fp32",
            "device": str(DEVICE),
        })
        mlflow.set_tags({"stage": "smoke-test" if args.max_rows else "final",
                         "framework": "pytorch"})

        tokenizer, model = build_model_and_tokenizer()
        train_loader, test_loader = build_dataloaders(train_df, test_df,
                                                      tokenizer)
        model = train_model(model, train_loader, test_loader,
                            epochs=args.epochs)

        y_true, y_pred, y_prob = evaluate_model(model, test_loader)
        print_evaluation_report(y_true, y_pred, y_prob, test_df)
        mlflow.log_metrics({
            **classification_metrics(y_true, y_pred),
            "majority_baseline": float(max(y_true.mean(), 1 - y_true.mean())),
        })

        save_model_and_artifacts(model, tokenizer, args.save_dir)
        mlflow.log_artifacts(args.save_dir, artifact_path="model")
        demo_reload_and_predict(args.save_dir)


if __name__ == "__main__":
    main()
