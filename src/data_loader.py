"""
data_loader.py
==============
Load and clean the churn CSV dataset.

Pipeline:
  1. Read CSV
  2. Validate and normalise target column → binary int (0 / 1)
  3. Drop ID-like columns
  4. Auto-coerce numeric-string columns (e.g. TotalCharges)
  5. Detect numeric vs. categorical feature columns
"""
from __future__ import annotations

import logging

import pandas as pd

from config import ID_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


# ── Target normalisation ───────────────────────────────────────────────────────

_TARGET_MAP: dict[str, int] = {
    "yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1,
    "no":  0, "n": 0, "false": 0, "0": 0, "stay":  0,
}


def normalize_target(series: pd.Series) -> pd.Series:
    """Map common churn target values to binary integers 0 / 1."""
    mapped = series.astype(str).str.strip().str.lower().map(_TARGET_MAP)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Unrecognised target labels: {bad}. "
            "Please map your target column to 0/1 before loading."
        )
    return mapped.astype(int)


# ── Feature cleaning ───────────────────────────────────────────────────────────

def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-convert object/string columns that are mostly numeric strings to float.

    Handles real-world cases like ``TotalCharges`` in the Telco dataset,
    which arrives as a string column due to a few blank entries.

    Columns where ≥ 90 % of non-null values parse as numbers are converted.
    All others remain strings (for OneHotEncoder downstream).

    Note: pandas ≥ 2.x reports string columns as dtype ``str``, not ``object``,
    so we use ``pd.api.types.is_string_dtype()`` to catch both variants.
    """
    X = X.copy()
    for col in X.select_dtypes(include=["object", "string"]).columns:
        text    = X[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
        numeric = pd.to_numeric(text, errors="coerce")
        n_valid = int(text.notna().sum())
        if n_valid and (numeric.notna().sum() / n_valid) >= 0.9:
            X[col] = numeric
        else:
            X[col] = text
    return X


# ── Public API ─────────────────────────────────────────────────────────────────

def load_data(
    csv_path: str,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Load, clean, and split the churn dataset.

    Parameters
    ----------
    csv_path   : path to the raw CSV file
    target_col : name of the churn label column

    Returns
    -------
    X              : pd.DataFrame  — cleaned feature matrix
    y              : pd.Series     — binary target (0 = stay, 1 = churn)
    numeric_cols   : list[str]     — numeric feature column names
    categorical_cols : list[str]   — categorical feature column names
    """
    # Auto-detect delimiter (supports both legacy comma files and new semicolon files).
    df = pd.read_csv(csv_path, sep=None, engine="python")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )

    y = normalize_target(df[target_col])
    X = df.drop(columns=[target_col])

    # Drop ID-like columns (use set for O(1) lookup)
    to_drop = [c for c in X.columns if c in ID_COLUMNS]
    if to_drop:
        logger.info("Dropping ID columns: %s", to_drop)
        X = X.drop(columns=to_drop)

    X = clean_features(X)

    # Detect column types AFTER clean_features has run
    numeric_cols     = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    logger.info(
        "Loaded %d rows | %d numeric | %d categorical features",
        len(df), len(numeric_cols), len(categorical_cols),
    )
    logger.info("Target distribution: %s", y.value_counts().to_dict())

    return X, y, numeric_cols, categorical_cols
