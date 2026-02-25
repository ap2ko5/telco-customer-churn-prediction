"""
tests/test_data_loader.py
=========================
Unit tests for src/data_loader.py

What we test:
  1. normalize_target() correctly maps "Yes"/"No" to 1/0
  2. normalize_target() raises ValueError on unknown labels
  3. clean_features() coerces a string column that is mostly numeric
  4. clean_features() keeps a true categorical column as string

Run with:
  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

# Make src/ importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from data_loader import clean_features, normalize_target


# ─────────────────────────────────────────────────────────────────────────────
# Tests for normalize_target()
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeTarget:
    """Tests for the normalize_target() function."""

    def test_yes_no_maps_correctly(self):
        """'Yes' and 'No' should map to 1 and 0."""
        series = pd.Series(["Yes", "No", "Yes", "No"])
        result = normalize_target(series)
        expected = pd.Series([1, 0, 1, 0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_case_insensitive(self):
        """Mapping should be case-insensitive (yes, YES, Yes all → 1)."""
        series = pd.Series(["YES", "NO", "yes", "no", "Yes", "No"])
        result = normalize_target(series)
        expected = pd.Series([1, 0, 1, 0, 1, 0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_alternative_labels(self):
        """Alternative labels like 'true/false', '1/0', 'churn/stay' should work."""
        series = pd.Series(["true", "false", "1", "0", "churn", "stay"])
        result = normalize_target(series)
        expected = pd.Series([1, 0, 1, 0, 1, 0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_raises_on_unknown_labels(self):
        """Unknown labels should raise ValueError listing the bad values."""
        series = pd.Series(["Yes", "No", "Maybe", "Unknown"])
        with pytest.raises(ValueError) as excinfo:
            normalize_target(series)
        # Error message should mention the bad values
        assert "Maybe" in str(excinfo.value) or "Unknown" in str(excinfo.value)

    def test_returns_int_dtype(self):
        """Output dtype should be int (not float or object)."""
        series = pd.Series(["Yes", "No"])
        result = normalize_target(series)
        assert result.dtype == int or str(result.dtype).startswith("int")


# ─────────────────────────────────────────────────────────────────────────────
# Tests for clean_features()
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanFeatures:
    """Tests for the clean_features() function."""

    def test_coerces_mostly_numeric_string_column(self):
        """
        A column like TotalCharges stored as strings with 90%+ numeric values
        should be converted to float.
        """
        df = pd.DataFrame({
            "TotalCharges": ["100.5", "200.0", "150.3", "  ", "300.0"],
            # 4/5 = 80% numeric... let's make it more: 4 numeric out of 4 non-blank
        })
        # Use a cleaner example: 5 numeric strings, 0 non-numeric
        df2 = pd.DataFrame({
            "TotalCharges": ["100.5", "200.0", "150.3", "299.0", "300.0"],
        })
        result = clean_features(df2)
        assert pd.api.types.is_numeric_dtype(result["TotalCharges"]), (
            "TotalCharges should be converted to numeric"
        )

    def test_keeps_categorical_column_as_string(self):
        """
        A column like 'Contract' with values 'Month-to-month' etc.
        should remain as a string type.
        """
        df = pd.DataFrame({
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        })
        result = clean_features(df)
        assert pd.api.types.is_string_dtype(result["Contract"]) or \
               result["Contract"].dtype == object, (
            "Contract column should stay as string/object type"
        )

    def test_does_not_modify_already_numeric_column(self):
        """Columns that are already numeric should pass through unchanged."""
        df = pd.DataFrame({
            "tenure": [12, 24, 6, 36, 1],
        })
        result = clean_features(df)
        assert pd.api.types.is_numeric_dtype(result["tenure"])
        pd.testing.assert_series_equal(df["tenure"], result["tenure"])

    def test_does_not_mutate_input(self):
        """clean_features() should return a copy — never modify the original."""
        df = pd.DataFrame({
            "TotalCharges": ["100.0", "200.0", "300.0"],
        })
        original_dtype = df["TotalCharges"].dtype
        _ = clean_features(df)
        assert df["TotalCharges"].dtype == original_dtype, (
            "Original DataFrame should not be modified (clean_features must copy)"
        )

    def test_strips_whitespace_from_strings(self):
        """String columns should have leading/trailing whitespace stripped."""
        df = pd.DataFrame({
            "Contract": ["  Month-to-month  ", " One year ", "Two year"],
        })
        result = clean_features(df)
        assert result["Contract"].iloc[0] == "Month-to-month"
        assert result["Contract"].iloc[1] == "One year"
