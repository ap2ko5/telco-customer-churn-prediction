"""
data_loader.py — Load and clean the churn CSV dataset.

Steps:
  1. Read CSV
  2. Normalize target column to binary 0/1
  3. Drop ID-like columns
  4. Strip whitespace from object columns
  5. Auto-detect numeric vs. categorical columns
"""
from __future__ import annotations

import pandas as pd

from config import ID_COLUMNS, RANDOM_STATE, TARGET_COLUMN


def normalize_target(series: pd.Series) -> pd.Series:
    """Map common churn target values to binary integers 0 / 1."""
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1,
        "no":  0, "n": 0, "false": 0, "0": 0, "stay": 0,
    }
    mapped = text.map(mapping)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Unrecognized target labels: {bad}. "
            "Please map your target column to 0/1 manually."
        )
    return mapped.astype(int)


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-convert object/string columns that are mostly numeric strings into floats.

    This handles real-world cases like `TotalCharges` in the Telco dataset,
    which is stored as a string column due to a few blank values that prevent
    pandas from parsing it as float on read.

    Columns where >= 90% of non-null values parse as numbers are converted.
    All others are kept as strings (for OneHotEncoder).

    Note: pandas 2.x reports string columns as dtype `str` (not `object`),
    so we use pd.api.types.is_string_dtype() to catch both cases.
    """
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_string_dtype(X[col]):
            continue
        text = X[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
        numeric = pd.to_numeric(text, errors="coerce")
        non_null = int(text.notna().sum())
        parse_ratio = float(numeric.notna().sum() / non_null) if non_null else 0.0
        if parse_ratio >= 0.9:
            X[col] = numeric
        else:
            X[col] = text
    return X


def load_data(
    csv_path: str,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Load, clean, and split the churn dataset.

    Returns
    -------
    X : pd.DataFrame        Feature matrix (cleaned)
    y : pd.Series           Binary target (0 / 1)
    numeric_cols : list     Names of numeric feature columns
    categorical_cols : list Names of categorical feature columns
    """
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        available = df.columns.tolist()
        raise ValueError(
            f"Target column '{target_col}' not found. Available: {available}"
        )

    y = normalize_target(df[target_col])
    X = df.drop(columns=[target_col]).copy()

    # Drop ID-like columns
    to_drop = [c for c in X.columns if c in ID_COLUMNS]
    if to_drop:
        print(f"[data_loader] Dropping ID columns: {to_drop}")
        X = X.drop(columns=to_drop)

    # Strip whitespace and auto-coerce numeric strings (e.g. TotalCharges)
    X = clean_features(X)

    # Auto-detect column types AFTER clean_features has run
    numeric_cols     = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print(
        f"[data_loader] Loaded {len(df):,} rows | "
        f"{len(numeric_cols)} numeric | {len(categorical_cols)} categorical features"
    )
    print(f"[data_loader] Target distribution:\n{y.value_counts().to_dict()}")

    return X, y, numeric_cols, categorical_cols

