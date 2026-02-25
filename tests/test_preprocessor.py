"""
tests/test_preprocessor.py
===========================
Unit tests for src/preprocessor.py

What we test:
  1. build_preprocessor() returns a ColumnTransformer
  2. Numeric columns are scaled (mean ≈ 0 after StandardScaler)
  3. Categorical columns are one-hot encoded (binary 0/1 values)

Run with:
  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from preprocessor import build_preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — small synthetic datasets for fast testing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_dataframe():
    """A tiny 10-row synthetic DataFrame with 2 numeric + 2 categorical columns."""
    return pd.DataFrame({
        "tenure":         [1, 12, 24, 36, 48, 60, 6, 18, 30, 72],
        "MonthlyCharges": [20.0, 45.0, 65.0, 80.0, 90.0, 95.0, 30.0, 50.0, 70.0, 110.0],
        "Contract":       ["Month-to-month", "One year", "Two year",
                           "Month-to-month", "One year", "Two year",
                           "Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaymentMethod":  ["Electronic check", "Mailed check", "Bank transfer",
                           "Credit card", "Electronic check", "Mailed check",
                           "Bank transfer", "Credit card", "Electronic check", "Mailed check"],
    })


@pytest.fixture
def preprocessor_with_data(sample_dataframe):
    """Returns a FITTED preprocessor for use in transformation tests."""
    numeric_cols     = ["tenure", "MonthlyCharges"]
    categorical_cols = ["Contract", "PaymentMethod"]
    pp = build_preprocessor(numeric_cols, categorical_cols)
    pp.fit(sample_dataframe)
    return pp, sample_dataframe, numeric_cols, categorical_cols


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPreprocessor:
    """Tests for build_preprocessor()."""

    def test_returns_column_transformer(self, sample_dataframe):
        """build_preprocessor() should return a ColumnTransformer instance."""
        numeric_cols     = ["tenure", "MonthlyCharges"]
        categorical_cols = ["Contract", "PaymentMethod"]
        pp = build_preprocessor(numeric_cols, categorical_cols)
        assert isinstance(pp, ColumnTransformer), (
            f"Expected ColumnTransformer, got {type(pp)}"
        )

    def test_has_num_and_cat_transformers(self, sample_dataframe):
        """The ColumnTransformer should have two named transformers: 'num' and 'cat'."""
        pp = build_preprocessor(["tenure"], ["Contract"])
        pp.fit(sample_dataframe)
        transformer_names = [name for name, _, _ in pp.transformers_]
        assert "num" in transformer_names, "Expected 'num' transformer"
        assert "cat" in transformer_names, "Expected 'cat' transformer"


class TestPreprocessorTransformation:
    """Tests for the output of transform()."""

    def test_output_is_numpy_array(self, preprocessor_with_data):
        """transform() should return a 2D numpy array."""
        pp, df, _, _ = preprocessor_with_data
        X = pp.transform(df)
        assert isinstance(X, np.ndarray), f"Expected numpy array, got {type(X)}"
        assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D"

    def test_numeric_columns_are_scaled(self, preprocessor_with_data):
        """
        After StandardScaler, numeric features should have mean ≈ 0 and std ≈ 1.
        We check only the first 2 columns (the numeric ones).
        """
        pp, df, numeric_cols, _ = preprocessor_with_data
        X = pp.transform(df)
        n_numeric = len(numeric_cols)
        numeric_part = X[:, :n_numeric]

        col_means = numeric_part.mean(axis=0)
        col_stds  = numeric_part.std(axis=0)

        # Means should be very close to 0
        np.testing.assert_allclose(col_means, np.zeros(n_numeric), atol=1e-6,
                                   err_msg="Scaled numeric columns should have mean ≈ 0")
        # Stds should be close to 1
        np.testing.assert_allclose(col_stds, np.ones(n_numeric), atol=0.3,
                                   err_msg="Scaled numeric columns should have std ≈ 1")

    def test_categorical_columns_are_binary(self, preprocessor_with_data):
        """
        One-hot encoded columns should only contain 0.0 and 1.0 values.
        """
        pp, df, numeric_cols, _ = preprocessor_with_data
        X = pp.transform(df)
        n_numeric = len(numeric_cols)
        cat_part = X[:, n_numeric:]  # Slice out the categorical columns

        unique_vals = np.unique(cat_part)
        for v in unique_vals:
            assert v in {0.0, 1.0}, (
                f"One-hot encoded values should only be 0 or 1, found {v}"
            )

    def test_no_nan_after_transform(self, preprocessor_with_data):
        """
        SimpleImputer should fill all missing values. No NaN should remain after
        transform.
        """
        pp, df, _, _ = preprocessor_with_data
        # Introduce a missing value
        df_with_nan = df.copy()
        df_with_nan.loc[0, "MonthlyCharges"] = np.nan
        df_with_nan.loc[1, "Contract"] = np.nan

        X = pp.transform(df_with_nan)
        assert not np.any(np.isnan(X)), "No NaN values should remain after transform"

    def test_output_shape(self, preprocessor_with_data):
        """
        Output should have n_rows rows.
        Columns = n_numeric + total_one_hot_categories.
        """
        pp, df, numeric_cols, categorical_cols = preprocessor_with_data
        X = pp.transform(df)
        assert X.shape[0] == len(df), (
            f"Expected {len(df)} rows, got {X.shape[0]}"
        )
        # At minimum there should be more columns than raw features (due to OHE)
        n_raw = len(numeric_cols) + len(categorical_cols)
        assert X.shape[1] >= n_raw, (
            "After OHE, output should have at least as many columns as input features"
        )
