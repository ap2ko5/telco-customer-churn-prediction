"""
preprocessor.py
===============
Build and fit the sklearn ColumnTransformer preprocessing pipeline.

Numeric pipeline  : SimpleImputer(median)       → StandardScaler
Categorical pipeline : SimpleImputer(most_freq) → OneHotEncoder(handle_unknown="ignore")
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELS_DIR

logger = logging.getLogger(__name__)

_PREPROC_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"


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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,   # cleaner feature names
    )


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: "pd.DataFrame",  # noqa: F821 — avoid circular import
    *,
    save: bool = True,
) -> ColumnTransformer:
    """Fit the preprocessor on training data and optionally persist it."""
    preprocessor.fit(X_train)
    if save:
        joblib.dump(preprocessor, _PREPROC_PATH)
        logger.info("Preprocessor saved → %s", _PREPROC_PATH)
    return preprocessor


def transform(preprocessor: ColumnTransformer, X) -> np.ndarray:
    """Apply the fitted preprocessor to X."""
    return preprocessor.transform(X)


def load_preprocessor() -> ColumnTransformer:
    """Load the saved preprocessing pipeline from disk."""
    if not _PREPROC_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at {_PREPROC_PATH}.\n"
            "Run `python src/train_pipeline.py` first."
        )
    return joblib.load(_PREPROC_PATH)
