"""
preprocessor.py — Build and fit the ColumnTransformer preprocessing pipeline.

Numeric pipeline : SimpleImputer(median)  →  StandardScaler
Categorical pipeline : SimpleImputer(most_frequent)  →  OneHotEncoder(handle_unknown="ignore")
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELS_DIR


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Construct (but do not fit) the ColumnTransformer."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train,
    save: bool = True,
) -> ColumnTransformer:
    """Fit the preprocessor on training data and optionally save it."""
    preprocessor.fit(X_train)
    if save:
        out = MODELS_DIR / "preprocessing_pipeline.joblib"
        joblib.dump(preprocessor, out)
        print(f"[preprocessor] Saved to {out}")
    return preprocessor


def transform(preprocessor: ColumnTransformer, X) -> np.ndarray:
    """Apply the fitted preprocessor to X."""
    return preprocessor.transform(X)
