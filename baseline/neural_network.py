"""
neural_network.py

Module 12: A rudimentary two-layer neural network (implemented with NumPy
only) for predicting graduate admissions outcomes.

This file is being built incrementally, one assignment step at a time.
Currently implemented:
    Step 1 - Dataset Loading, Filtering, and Feature Construction
    Step 2 - Split and Preprocess the Data
    Step 3 - Build a Two-Layer Neural Network in NumPy
    Step 4 - Train the Model Until Test MSE Stops Improving
    Step 5 - Evaluate the Final Model
    Step 6 - Plot Train and Test MSE Over Time
    Step 7 - Test the Model on Artificial Applicants
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, so this runs headlessly
import matplotlib.pyplot as plt

# Shared MLflow helpers live at the repo root (one level up).
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlflow  # noqa: E402  pylint: disable=wrong-import-position
from mlflow.entities import Metric  # noqa: E402  pylint: disable=wrong-import-position
from tracking import setup_mlflow, classification_metrics  # noqa: E402  pylint: disable=wrong-import-position


# Fixed configuration used throughout this assignment.
RANDOM_SEED = 42
HIDDEN_UNITS = 6
LEARNING_RATE = 0.05
MAX_EPOCHS = 10000
PATIENCE = 100

FEATURE_COLUMNS = [
    "gpa",
    "gre",
    "gre_v",
    "gre_aw",
    "ms_vs_phd",
    "international_vs_local",
]
TARGET_COLUMN = "target"

# Valid score ranges for the modern (post-2011) GRE scale. Values outside
# these bounds are data-entry errors or leftover sentinel values (e.g. 999,
# 99.99) rather than real scores, and are treated as missing.
GRE_VALID_RANGE = (130, 170)       # GRE Quantitative
GRE_V_VALID_RANGE = (130, 170)     # GRE Verbal
GRE_AW_VALID_RANGE = (0, 6)        # GRE Analytical Writing

# GRE bounds keyed by column name, for reuse across Step 1 and Step 7.
GRE_VALID_RANGES = {
    "gre": GRE_VALID_RANGE,
    "gre_v": GRE_V_VALID_RANGE,
    "gre_aw": GRE_AW_VALID_RANGE,
}


# ---------------------------------------------------------------------------
# Shared preprocessing helpers
#
# These three functions implement the pieces of the cleaning/preprocessing
# pipeline that need to be applied identically to the real data (Step 1 /
# Step 2) AND to any new, hand-built applicants we want to run through the
# trained model later (Step 7). Keeping them as standalone functions means
# Step 7 can reuse the exact same logic instead of re-implementing it.
# ---------------------------------------------------------------------------

def clean_out_of_bounds_gre_scores(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Set out-of-bounds gre, gre_v, and gre_aw values to NaN.

    The raw data (and, in principle, any hand-entered applicant) can contain
    sentinel/data-entry-error values (e.g. a gre_v of 999, a gre_aw of
    99.99) that fall outside the valid modern GRE scale. These are treated
    as missing rather than as real scores.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns "gre", "gre_v", "gre_aw".
    verbose : bool
        If True, print how many out-of-bounds values were found per column.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with out-of-bounds values replaced by NaN.
    """
    cleaned = df.copy()
    for col, (low, high) in GRE_VALID_RANGES.items():
        out_of_bounds = ~cleaned[col].between(low, high) & cleaned[col].notna()
        n_out_of_bounds = int(out_of_bounds.sum())
        if verbose and n_out_of_bounds > 0:
            print(
                f"Setting {n_out_of_bounds} out-of-bounds '{col}' value(s) "
                f"(valid range {low}-{high}) to NaN"
            )
        cleaned.loc[out_of_bounds, col] = np.nan
    return cleaned


def fill_missing_with_medians(X_df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """
    Fill missing values in X_df using a fixed, previously-computed set of
    per-feature medians (e.g. the training-set medians).

    This is intentionally a thin wrapper around DataFrame.fillna() so that
    Step 2 (train/test preprocessing) and Step 7 (artificial applicants)
    both fill missing values the exact same way, using the exact same
    stored medians.
    """
    return X_df.fillna(medians)


def standardize_features(X_df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """
    Standardize X_df using a fixed, previously-computed set of per-feature
    means and standard deviations (e.g. the training-set statistics).

    `stds` is expected to already have any zero standard deviations
    replaced with 1 (see split_and_preprocess_data's Step 5), so this
    function does not need to guard against division by zero itself.
    """
    return (X_df - means) / stds


# ---------------------------------------------------------------------------
# Step 1: Load and Prepare the Applicant Dataset
# ---------------------------------------------------------------------------

def load_and_prepare_data(json_lines_path: str) -> pd.DataFrame:
    """
    Load the applicant dataset from a JSON Lines file (one JSON object per
    line) and prepare it for modeling.

    Preprocessing performed:
        1. Load the JSON Lines file into a Pandas DataFrame.
        2. Keep only rows where applicant_status is "Accepted" or "Rejected".
        3. Keep only rows where masters_or_phd is "Masters" or "PhD".
        4. Convert gpa, gre, gre_v, and gre_aw from strings into floats.
        4b. Set out-of-bounds gre, gre_v, and gre_aw values to NaN (these
            are treated as missing rather than real scores).
        5. Create ms_vs_phd: PhD = 1, Masters = 0.
        6. Create international_vs_local: International = 1, Local/American = 0.
        7. Create target: Accepted = 1, Rejected = 0.

    Parameters
    ----------
    json_lines_path : str
        Path to the JSON Lines applicant dataset (each line is a JSON object
        with fields: gpa, gre, gre_v, gre_aw, masters_or_phd, citizenship,
        applicant_status).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame containing the six required input features
        (gpa, gre, gre_v, gre_aw, ms_vs_phd, international_vs_local)
        plus the target column.
    """

    # --- 1. Load the JSON Lines file into a DataFrame -----------------------
    raw_df = pd.read_json(json_lines_path, lines=True)
    original_row_count = len(raw_df)

    # --- 2 & 3. Filter to valid applicant_status and masters_or_phd values ---
    status_mask = raw_df["applicant_status"].isin(["Accepted", "Rejected"])
    degree_mask = raw_df["masters_or_phd"].isin(["Masters", "PhD"])
    filtered_df = raw_df.loc[status_mask & degree_mask].copy()

    # --- 4. Convert numeric string columns into floats -----------------------
    # pd.to_numeric handles both string values ("3.9") and already-missing
    # (None/NaN) values gracefully, coercing anything unparseable to NaN.
    numeric_columns = ["gpa", "gre", "gre_v", "gre_aw"]
    for col in numeric_columns:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

    # --- 4b. Treat out-of-bounds GRE scores as missing -----------------------
    # (See clean_out_of_bounds_gre_scores docstring for rationale.) This
    # helper is shared with Step 7 so artificial applicants are cleaned the
    # same way as real applicant data.
    filtered_df = clean_out_of_bounds_gre_scores(filtered_df)

    # --- 5. Create ms_vs_phd: PhD = 1, Masters = 0 ---------------------------
    filtered_df["ms_vs_phd"] = (filtered_df["masters_or_phd"] == "PhD").astype(int)

    # --- 6. Create international_vs_local: International = 1, else 0 --------
    filtered_df["international_vs_local"] = (
        filtered_df["citizenship"] == "International"
    ).astype(int)

    # --- 7. Create target: Accepted = 1, Rejected = 0 ------------------------
    filtered_df["target"] = (filtered_df["applicant_status"] == "Accepted").astype(int)

    # --- Required printed summary ---------------------------------------------
    final_feature_names = [
        "gpa",
        "gre",
        "gre_v",
        "gre_aw",
        "ms_vs_phd",
        "international_vs_local",
    ]

    accepted_count = int((filtered_df["applicant_status"] == "Accepted").sum())
    rejected_count = int((filtered_df["applicant_status"] == "Rejected").sum())
    masters_count = int((filtered_df["masters_or_phd"] == "Masters").sum())
    phd_count = int((filtered_df["masters_or_phd"] == "PhD").sum())

    print(f"\nOriginal row count:            {original_row_count}")
    print(f"Filtered row count:             {len(filtered_df)}")
    print(f"Accepted count:                 {accepted_count}")
    print(f"Rejected count:                 {rejected_count}")
    print(f"Masters count:                  {masters_count}")
    print(f"PhD count:                      {phd_count}")
    print(f"Final input feature names:      {final_feature_names}")
    print("\nFirst few cleaned rows:")
    print(
        filtered_df[final_feature_names + ["target"]].head()
    )

    return filtered_df[final_feature_names + ["target"]]


# ---------------------------------------------------------------------------
# Step 2: Split and Preprocess the Data
# ---------------------------------------------------------------------------

def split_and_preprocess_data(cleaned_df: pd.DataFrame):
    """
    Split the cleaned applicant data into training and test sets, then
    apply leakage-safe median-fill and standardization preprocessing.

    Steps performed:
        1. Split into 80% training / 20% test using scikit-learn's
           train_test_split(), with test_size=0.2, random_state=42,
           and shuffle=True. scikit-learn is used ONLY for this split;
           no scikit-learn preprocessing utilities are used.
        2. Compute the median of each feature using the training set only.
        3. Fill missing values in both training and test sets using those
           training-set medians.
        4. Compute the mean and standard deviation of each feature using
           the training set only.
        5. Standardize both training and test sets using those
           training-set statistics (replacing any zero standard
           deviation with 1 before scaling).

    Why compute medians/means/standard deviations from the training set
    only:
        The test set is meant to simulate genuinely unseen data. If we
        computed these statistics from the full dataset (train + test),
        information about the test set's distribution would leak into
        the values used to fill and scale the training set, giving the
        model an indirect, overly optimistic preview of the test data.
        Fitting these statistics on the training set only and then
        reusing them on the test set keeps the test set truly held out,
        so that the final evaluation reflects how the model would
        perform on new applicants whose data was never seen during
        training.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Output of load_and_prepare_data(), containing FEATURE_COLUMNS
        and TARGET_COLUMN.

    Returns
    -------
    dict
        {
            "X_train": np.ndarray, shape (n_train, 6), standardized
            "X_test":  np.ndarray, shape (n_test, 6), standardized
            "y_train": np.ndarray, shape (n_train,)
            "y_test":  np.ndarray, shape (n_test,)
            "train_medians": pd.Series (indexed by FEATURE_COLUMNS)
            "train_means":   pd.Series (indexed by FEATURE_COLUMNS)
            "train_stds":    pd.Series (indexed by FEATURE_COLUMNS,
                                          zero std replaced with 1)
            "test_medians":  pd.Series, for reporting/comparison only
            "test_means":    pd.Series, for reporting/comparison only
            "test_stds":     pd.Series, for reporting/comparison only
                              (zero std replaced with 1)
        }
        Note: the "test_*" statistics are computed independently from the
        test set purely so they can be printed alongside the training
        statistics for comparison. They are NOT used to fill or scale
        X_test -- that uses "train_medians"/"train_means"/"train_stds"
        only, per the leakage-safe preprocessing rule.
    """

    X = cleaned_df[FEATURE_COLUMNS]
    y = cleaned_df[TARGET_COLUMN]

    # --- 1. Train/test split (scikit-learn used only for this step) --------
    X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )

    # --- 2 & 3. Median-fill using training-set medians only -----------------
    train_medians = X_train_df.median()
    X_train_filled = fill_missing_with_medians(X_train_df, train_medians)
    X_test_filled = fill_missing_with_medians(X_test_df, train_medians)

    # --- 4. Mean/std using training-set statistics only ---------------------
    train_means = X_train_filled.mean()
    train_stds = X_train_filled.std()

    # --- 5. Guard against zero standard deviation before scaling ------------
    safe_train_stds = train_stds.replace(0, 1)

    X_train_scaled = standardize_features(X_train_filled, train_means, safe_train_stds)
    X_test_scaled = standardize_features(X_test_filled, train_means, safe_train_stds)

    # --- Test-set descriptive statistics (for reporting/comparison ONLY) ----
    # These are NOT used anywhere in the actual fill/scale pipeline above --
    # X_test is filled and standardized using training-set statistics only,
    # exactly as required. Test-set medians/means/stds are computed here
    # purely so we can print a side-by-side comparison against the training
    # set's statistics.
    test_medians = X_test_df.median()
    X_test_filled_own = fill_missing_with_medians(X_test_df, test_medians)
    test_means = X_test_filled_own.mean()
    test_stds = X_test_filled_own.std().replace(0, 1)

    # --- Required printed summary --------------------------------------------
    print(f"Training set size:               {len(X_train_scaled)}")
    print(f"Test set size:                   {len(X_test_scaled)}")

    median_table = pd.DataFrame(
        {"Training Median": train_medians, "Test Median": test_medians}
    )
    means_table = pd.DataFrame(
        {"Training Means": train_means, "Test Means": test_means}
    )
    std_table = pd.DataFrame(
        {"Training StD": safe_train_stds, "Test StD": test_stds}
    )

    print("\nMedians (Training vs. Test):")
    print(median_table)
    print("\nMeans (Training vs. Test):")
    print(means_table)
    print("\nStandard Deviations (Training vs. Test, 0 replaced with 1):")
    print(std_table)

    print(
        "\nNote: only the Training columns above are actually used to fill "
        "and scale the data (per the leakage-safe preprocessing rule below). "
        "The Test columns are shown only for comparison, computed "
        "independently from the test set's own values."
    )
    print(
        "\nWhy compute these from the training set only: the test set stands "
        "in for unseen future applicants. Computing medians, means, and "
        "standard deviations from the full dataset would let information "
        "about the test set leak into the values used to fill and scale the "
        "training data, making test performance look better than it would "
        "on truly new data. Fitting only on the training set and reusing "
        "those exact values on the test set keeps the test set a fair, "
        "held-out measure of generalization."
    )

    return {
        "X_train": X_train_scaled.to_numpy(),
        "X_test": X_test_scaled.to_numpy(),
        "y_train": y_train_series.to_numpy(),
        "y_test": y_test_series.to_numpy(),
        "train_medians": train_medians,
        "train_means": train_means,
        "train_stds": safe_train_stds,
        "test_medians": test_medians,
        "test_means": test_means,
        "test_stds": test_stds,
    }


# ---------------------------------------------------------------------------
# Step 3: Build a Two-Layer Neural Network in NumPy
# ---------------------------------------------------------------------------

def sigmoid(x):
    """
    Numerically stable logistic sigmoid: 1 / (1 + exp(-x)).

    Inputs are clipped to [-50, 50] before exponentiating so that very
    large-magnitude inputs (which could otherwise overflow np.exp) are
    handled gracefully; sigmoid(50) and sigmoid(-50) are already
    indistinguishable from 1.0 and 0.0 at floating-point precision, so
    this clipping does not change the function's behavior in practice.
    """
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def mse(y_true, y_pred):
    """Mean Squared Error between true labels and predicted probabilities."""
    return np.mean((y_true - y_pred) ** 2)


class TwoLayerNet:
    """
    Rudimentary two-layer (one hidden layer) neural network, implemented
    with NumPy only:

        input (6 features) -> hidden layer (6 units, sigmoid)
                            -> output layer (1 unit, sigmoid)

    Parameter dimensions
    ---------------------
    Let n = number of samples in a batch, d = input_dim (6), h = hidden_dim (6).

        W1 : shape (d, h) = (6, 6)  -- maps each of the 6 input features
             onto each of the 6 hidden units. W1[i, j] is the weight from
             input feature i to hidden unit j.
        b1 : shape (1, h) = (1, 6)  -- one bias per hidden unit, broadcast
             across all samples in a batch.
        W2 : shape (h, 1) = (6, 1)  -- maps each of the 6 hidden units onto
             the single output unit. W2[j, 0] is the weight from hidden
             unit j to the output.
        b2 : shape (1, 1)           -- a single bias for the output unit.

    What the hidden layer computes
    -------------------------------
    z1 = X @ W1 + b1 is a linear combination of the six input features (a
    different weighted mix for each of the six hidden units), and
    a1 = sigmoid(z1) squashes each of those six combinations into (0, 1).
    Each hidden unit therefore learns its own nonlinear "detector" over
    some combination of GPA, GRE scores, degree type, and citizenship --
    for example, one hidden unit's weights might end up emphasizing
    high GPA together with PhD status, while another emphasizes low GRE
    scores. Because sigmoid is nonlinear, stacking a hidden layer before
    the output lets the network represent interactions between features
    that a single linear model (like logistic regression) cannot.

    What the output layer computes
    -------------------------------
    z2 = a1 @ W2 + b2 takes the six hidden-unit activations and combines
    them into a single weighted sum (plus bias) -- effectively a learned
    "vote" over the patterns the hidden layer detected. y_hat =
    sigmoid(z2) then squashes that single value into the (0, 1) range.

    Why the output can be interpreted as a probability-like score
    ----------------------------------------------------------------
    Because the final sigmoid activation maps any real-valued input to
    the open interval (0, 1), y_hat behaves like a probability: values
    near 1 correspond to strong evidence for the "Accepted" class
    (target = 1) and values near 0 correspond to strong evidence for
    "Rejected" (target = 0). Sigmoid outputs are monotonic and bounded
    the same way true probabilities are, which is why a threshold (e.g.
    0.5) applied to y_hat can be used to produce a hard Accepted/Rejected
    prediction.
    """

    def __init__(self, input_dim, hidden_dim, seed=RANDOM_SEED):
        rng = np.random.default_rng(seed)

        # Weights: normal distribution, mean 0, standard deviation 0.1.
        self.W1 = rng.normal(loc=0.0, scale=0.1, size=(input_dim, hidden_dim))
        self.W2 = rng.normal(loc=0.0, scale=0.1, size=(hidden_dim, 1))

        # Biases initialized to 0.
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        """
        Run a forward pass through the network.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)

        Returns
        -------
        np.ndarray, shape (n_samples, 1)
            Predicted probability-like scores (y_hat) in (0, 1).
        """
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_hat = sigmoid(self.z2)
        return self.y_hat

    def backward(self, X, y, learning_rate):
        """
        Run backpropagation for one full-batch gradient descent step,
        using MSE as the loss function, and update W1, b1, W2, b2 in place.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)
            Must be the same X passed into the preceding forward() call.
        y : np.ndarray, shape (n_samples, 1)
            True binary target labels (0 or 1).
        learning_rate : float
            Step size for the gradient descent update.
        """
        n = len(X)

        # MSE derivative through the output sigmoid:
        #   MSE = mean((y - y_hat) ** 2)
        #   d(MSE)/d(y_hat) = (2 / n) * (y_hat - y)
        #   d(y_hat)/d(z2)  = y_hat * (1 - y_hat)      [sigmoid derivative]
        #   dz2 = d(MSE)/d(z2) = d(MSE)/d(y_hat) * d(y_hat)/d(z2)
        dz2 = (2.0 / n) * (self.y_hat - y) * self.y_hat * (1.0 - self.y_hat)

        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1.0 - self.a1)  # sigmoid derivative at hidden layer

        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # Gradient descent parameter update.
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict_proba(self, X):
        """Return predicted probability-like scores in (0, 1)."""
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """Return hard 0/1 predictions by thresholding predict_proba(X)."""
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Step 4: Train the Model Until Test MSE Stops Improving
# ---------------------------------------------------------------------------

def train_model(
    net,
    X_train,
    y_train,
    X_test,
    y_test,
    max_epochs=MAX_EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
):
    """
    Train a TwoLayerNet using full-batch gradient descent, with early
    stopping based on test-set MSE.

    Each epoch:
        1. Forward pass on the training set, compute training MSE.
        2. Backpropagation + parameter update (full-batch gradient descent).
        3. Forward pass on the test set, compute test MSE and test accuracy
           (threshold = 0.5).
        4. Record all four values in a history structure.

    Early stopping:
        The best test MSE seen so far, and the parameters (W1, b1, W2, b2)
        that produced it, are tracked throughout training. If test MSE has
        not improved for `patience` consecutive epochs, training stops.
        The best parameters are restored onto `net` before returning, so
        that `net` always reflects the best-performing checkpoint rather
        than whatever it happened to end on.

    Parameters
    ----------
    net : TwoLayerNet
        Network to train (modified in place; ends with the best-epoch
        parameters restored).
    X_train, y_train : np.ndarray
        Training features, shape (n_train, 6), and labels, shape (n_train, 1).
    X_test, y_test : np.ndarray
        Test features, shape (n_test, 6), and labels, shape (n_test, 1).
    max_epochs : int
        Hard cap on the number of epochs (MAX_EPOCHS = 10000).
    learning_rate : float
        Gradient descent step size (LEARNING_RATE = 0.05).
    patience : int
        Number of consecutive non-improving epochs allowed before stopping
        (PATIENCE = 100).

    Returns
    -------
    dict
        {
            "history": {
                "epoch":     list[int],
                "train_mse": list[float],
                "test_mse":  list[float],
                "test_acc":  list[float],
            },
            "best_epoch": int,
            "best_test_mse": float,
        }
    """

    history = {"epoch": [], "train_mse": [], "test_mse": [], "test_acc": []}

    best_test_mse = np.inf
    best_epoch = 0
    best_params = None
    epochs_since_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # --- Forward pass + training MSE -------------------------------
        train_y_hat = net.forward(X_train)
        train_mse = mse(y_train, train_y_hat)

        # --- Backpropagation + parameter update --------------------------
        net.backward(X_train, y_train, learning_rate)

        # --- Forward pass on test set + test MSE / accuracy --------------
        test_y_hat = net.forward(X_test)
        test_mse = mse(y_test, test_y_hat)
        test_acc = float(np.mean((test_y_hat >= 0.5).astype(int) == y_test))

        # --- Save to history ------------------------------------------------
        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)
        history["test_acc"].append(test_acc)

        # --- Early stopping bookkeeping -----------------------------------
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_epoch = epoch
            best_params = {
                "W1": net.W1.copy(),
                "b1": net.b1.copy(),
                "W2": net.W2.copy(),
                "b2": net.b2.copy(),
            }
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # --- Progress printout every 100 epochs ----------------------------
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}/{max_epochs} - "
                f"Train MSE: {train_mse:.6f}, "
                f"Test MSE: {test_mse:.6f}, "
                f"Test Acc: {test_acc:.4f}"
            )

        if epochs_since_improvement >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch}: "
                f"test MSE has not improved for {patience} consecutive epochs."
            )
            break

    # --- Restore best parameters before final evaluation --------------------
    net.W1 = best_params["W1"]
    net.b1 = best_params["b1"]
    net.W2 = best_params["W2"]
    net.b2 = best_params["b2"]

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best test MSE: {best_test_mse:.6f}")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_test_mse": best_test_mse,
    }


# ---------------------------------------------------------------------------
# Step 5: Evaluate the Final Model
# ---------------------------------------------------------------------------

def evaluate_final_model(
    net,
    X_train,
    y_train,
    X_test,
    y_test,
    training_result,
    n_rows_after_filtering,
):
    """
    Compute and print the final evaluation metrics for the trained network,
    after the best (lowest test-MSE) parameters have already been restored
    onto `net` by train_model().

    Parameters
    ----------
    net : TwoLayerNet
        Trained network, with best-epoch parameters already restored.
    X_train, y_train, X_test, y_test : np.ndarray
        Preprocessed features/labels (y_* as column vectors, shape (n, 1)).
    training_result : dict
        Return value of train_model(), containing "best_epoch" and
        "best_test_mse".
    n_rows_after_filtering : int
        Number of rows remaining after Step 1's filtering.

    Returns
    -------
    dict
        {
            "final_train_accuracy": float,
            "final_test_accuracy": float,
        }
    """

    final_train_accuracy = float(np.mean(net.predict(X_train) == y_train))
    final_test_accuracy = float(np.mean(net.predict(X_test) == y_test))

    print(f"Rows used after filtering: {n_rows_after_filtering}")
    print(f"Train size: {len(X_train)}")
    print(f"Test Size: {len(X_test)}")
    print(f"Best Epoch: {training_result['best_epoch']}")
    print(f"Best test MSE: {training_result['best_test_mse']:.6f}")
    print(f"Final train accuracy: {final_train_accuracy:.4f}")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")

    return {
        "final_train_accuracy": final_train_accuracy,
        "final_test_accuracy": final_test_accuracy,
    }


# ---------------------------------------------------------------------------
# Step 6: Plot Train and Test MSE Over Time
# ---------------------------------------------------------------------------

def plot_mse_curve(history, output_path="mse_curve.png"):
    """
    Plot training MSE and test MSE versus epoch, and save the figure.

    Parameters
    ----------
    history : dict
        The "history" entry from train_model()'s return value, containing
        "epoch", "train_mse", and "test_mse" lists.
    output_path : str
        File path to save the resulting plot to (default "mse_curve.png").
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history["epoch"], history["train_mse"], label="Train MSE", linewidth=2)
    ax.plot(history["epoch"], history["test_mse"], label="Test MSE", linewidth=2)

    ax.set_title("Training vs. Test MSE Over Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.legend(title="Curve")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved MSE curve plot to {output_path}")


# ---------------------------------------------------------------------------
# Step 7: Test the Model on Artificial Applicants
# ---------------------------------------------------------------------------

def create_artificial_applicants() -> pd.DataFrame:
    """
    Build a small DataFrame of hand-specified, artificial applicants meant
    to illustrate contrasting cases: strong vs. weaker numeric profiles,
    crossed with PhD vs. Masters and International vs. Local, giving four
    applicants total (one for each PhD/Masters x strong/weaker combination).

    Returns
    -------
    pd.DataFrame
        Columns: "applicant_label" (a human-readable description, not fed
        into the model) plus all of FEATURE_COLUMNS.
    """
    applicants = pd.DataFrame(
        {
            "applicant_label": [
                "PhD, strong profile, International",
                "PhD, weaker profile, US",
                "Masters, strong profile, International",
                "Masters, weaker profile, US",
            ],
            "gpa": [3.90, 3.30, 3.85, 3.20],
            "gre": [168, 150, 165, 148],
            "gre_v": [165, 148, 162, 145],
            "gre_aw": [5.0, 3.5, 4.5, 3.0],
            "ms_vs_phd": [1, 1, 0, 0],
            "international_vs_local": [1, 0, 1, 0],
        }
    )
    return applicants


def predict_artificial_applicants(
    net,
    applicants_df: pd.DataFrame,
    train_medians: pd.Series,
    train_means: pd.Series,
    train_stds: pd.Series,
) -> pd.DataFrame:
    """
    Run the trained network on a DataFrame of artificial applicants,
    applying the exact same preprocessing pipeline used for the real data.

    Steps performed (reusing the same helper functions as Steps 1 and 2):
        1. Treat out-of-bounds gre/gre_v/gre_aw values as missing
           (clean_out_of_bounds_gre_scores).
        2. Fill missing values using the stored TRAINING-set medians
           (fill_missing_with_medians).
        3. Standardize using the stored TRAINING-set means/standard
           deviations (standardize_features).
        4. Convert to a NumPy array and run predict_proba() / predict().

    Parameters
    ----------
    net : TwoLayerNet
        Trained network (with best-epoch parameters restored).
    applicants_df : pd.DataFrame
        Output of create_artificial_applicants() (or any DataFrame
        containing "applicant_label" plus all of FEATURE_COLUMNS).
    train_medians, train_means, train_stds : pd.Series
        The training-set statistics from split_and_preprocess_data().

    Returns
    -------
    pd.DataFrame
        `applicants_df` with three additional columns: "Predicted
        Probability", "Predicted Label" (0/1), and "Predicted Status"
        ("Accepted"/"Rejected").
    """
    X_df = applicants_df[FEATURE_COLUMNS]

    # --- 1. Treat out-of-bounds GRE scores as missing, same as Step 1 -------
    X_cleaned = clean_out_of_bounds_gre_scores(X_df, verbose=False)

    # --- 2. Fill missing values using training-set medians ------------------
    X_filled = fill_missing_with_medians(X_cleaned, train_medians)

    # --- 3. Standardize using training-set means/standard deviations -------
    X_scaled = standardize_features(X_filled, train_means, train_stds)

    # --- 4. Convert to NumPy array and run the trained model ---------------
    X_array = X_scaled.to_numpy()
    predicted_probabilities = net.predict_proba(X_array).flatten()
    predicted_labels = net.predict(X_array).flatten()
    predicted_status = np.where(predicted_labels == 1, "Accepted", "Rejected")

    result_df = applicants_df.copy()
    result_df["Predicted Probability"] = predicted_probabilities
    result_df["Predicted Label"] = predicted_labels
    result_df["Predicted Status"] = predicted_status

    print("Artificial Applicant Predictions:")
    print(result_df.to_string(index=False))

    return result_df


def log_history_to_mlflow(history, every=10):
    """Log per-epoch curves to the active MLflow run (every Nth epoch).

    Uses batched writes: logging 10,000 epochs one call at a time would be
    needlessly slow against the SQLite store.
    """
    run_id = mlflow.active_run().info.run_id
    client = mlflow.tracking.MlflowClient()
    now = int(time.time() * 1000)
    metrics = []
    for i, epoch in enumerate(history["epoch"]):
        if epoch % every and epoch != history["epoch"][-1]:
            continue
        metrics += [
            Metric("train_mse", float(history["train_mse"][i]), now, epoch),
            Metric("epoch_test_mse", float(history["test_mse"][i]), now, epoch),
            Metric("epoch_test_accuracy", float(history["test_acc"][i]), now, epoch),
        ]
    for start in range(0, len(metrics), 1000):  # MLflow's per-batch limit
        client.log_batch(run_id, metrics=metrics[start:start + 1000])


def log_run_to_mlflow(net, split_data, training_result, evaluation_result,
                      n_rows_after_filtering, plot_path):
    """Record this training run (settings, curves, metrics, weights) in MLflow."""
    mlflow.log_params({
        "model_family": "two-layer-nn-numpy",
        "hidden_units": HIDDEN_UNITS,
        "activation": "sigmoid",
        "loss": "mse",
        "optimizer": "full-batch gradient descent",
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "seed": RANDOM_SEED,
        "features": ",".join(FEATURE_COLUMNS),
        "missing_values": "train-median fill",
    })
    mlflow.set_tags({"stage": "baseline", "framework": "numpy"})

    log_history_to_mlflow(training_result["history"])

    y_test = split_data["y_test"].reshape(-1)
    y_pred = net.predict(split_data["X_test"]).reshape(-1)
    mlflow.log_metrics(classification_metrics(y_test, y_pred))
    mlflow.log_metrics({
        "train_accuracy": evaluation_result["final_train_accuracy"],
        "best_test_mse": float(training_result["best_test_mse"]),
        "best_epoch": float(training_result["best_epoch"]),
        "rows_after_filtering": float(n_rows_after_filtering),
        "majority_baseline": float(max(y_test.mean(), 1 - y_test.mean())),
    })

    # Weights + preprocessing statistics: everything needed to rebuild the
    # model outside Python (e.g. in the browser).
    model_state = {
        "feature_columns": FEATURE_COLUMNS,
        "W1": net.W1.tolist(), "b1": net.b1.tolist(),
        "W2": net.W2.tolist(), "b2": net.b2.tolist(),
        "train_medians": split_data["train_medians"].to_dict(),
        "train_means": split_data["train_means"].to_dict(),
        "train_stds": split_data["train_stds"].to_dict(),
    }
    mlflow.log_dict(model_state, "model/two_layer_net.json")
    mlflow.log_artifact(plot_path)


if __name__ == "__main__":
    # Path to the JSON Lines dataset expected by this assignment.
    DATA_PATH = "gradcafe_module12_input.jsonl"

    setup_mlflow()
    mlflow.start_run(run_name="numpy-two-layer-net")

    cleaned_df = load_and_prepare_data(DATA_PATH)

    split_data = split_and_preprocess_data(cleaned_df)

    net = TwoLayerNet(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=HIDDEN_UNITS,
        seed=RANDOM_SEED,
    )
    print("Initialized TwoLayerNet:")
    print(f"  W1 shape: {net.W1.shape}")
    print(f"  b1 shape: {net.b1.shape}")
    print(f"  W2 shape: {net.W2.shape}")
    print(f"  b2 shape: {net.b2.shape}")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"].reshape(-1, 1)
    X_test = split_data["X_test"]
    y_test = split_data["y_test"].reshape(-1, 1)

    print("\n--- Training ---\n")
    training_result = train_model(
        net,
        X_train,
        y_train,
        X_test,
        y_test,
        max_epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
    )

    print("\n--- Final Evaluation ---\n")
    evaluation_result = evaluate_final_model(
        net,
        X_train,
        y_train,
        X_test,
        y_test,
        training_result,
        n_rows_after_filtering=len(cleaned_df),
    )

    plot_mse_curve(training_result["history"], output_path="mse_curve.png")

    print("\n--- Artificial Applicant Predictions ---\n")
    artificial_applicants = create_artificial_applicants()
    artificial_results = predict_artificial_applicants(
        net,
        artificial_applicants,
        train_medians=split_data["train_medians"],
        train_means=split_data["train_means"],
        train_stds=split_data["train_stds"],
    )

    log_run_to_mlflow(net, split_data, training_result, evaluation_result,
                      n_rows_after_filtering=len(cleaned_df),
                      plot_path="mse_curve.png")
    mlflow.log_table(artificial_results, "artificial_applicants.json")
    mlflow.end_run()
    print("\nLogged run 'numpy-two-layer-net' to MLflow.")
