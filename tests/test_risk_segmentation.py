"""
tests/test_risk_segmentation.py
================================
Unit tests for src/risk_segmentation.py

What we test:
  1. Each probability range maps to the correct band
  2. Edge cases: exactly 0.0, exactly 1.0, exactly at boundaries
  3. add_risk_band() correctly adds a column to a DataFrame
  4. All output bands are valid band names

Run with:
  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from risk_segmentation import add_risk_band, assign_risk_band


# ─────────────────────────────────────────────────────────────────────────────
# Tests for assign_risk_band()
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignRiskBand:
    """Tests for the assign_risk_band() function."""

    VALID_BANDS = {"Low", "Medium", "High", "Critical"}

    def test_low_band(self):
        """Probabilities in [0.0, 0.3) should be 'Low'."""
        probs = np.array([0.0, 0.1, 0.2, 0.29])
        bands = assign_risk_band(probs)
        for b in bands:
            assert b == "Low", f"Expected 'Low' for low probabilities, got '{b}'"

    def test_medium_band(self):
        """Probabilities in [0.3, 0.6) should be 'Medium'."""
        probs = np.array([0.3, 0.40, 0.50, 0.59])
        bands = assign_risk_band(probs)
        for b in bands:
            assert b == "Medium", f"Expected 'Medium', got '{b}'"

    def test_high_band(self):
        """Probabilities in [0.6, 0.8) should be 'High'."""
        probs = np.array([0.6, 0.65, 0.70, 0.79])
        bands = assign_risk_band(probs)
        for b in bands:
            assert b == "High", f"Expected 'High', got '{b}'"

    def test_critical_band(self):
        """Probabilities in [0.8, 1.0] should be 'Critical'."""
        probs = np.array([0.8, 0.85, 0.9, 0.95, 0.99])
        bands = assign_risk_band(probs)
        for b in bands:
            assert b == "Critical", f"Expected 'Critical', got '{b}'"

    def test_exactly_one(self):
        """Probability of exactly 1.0 should be 'Critical' (edge case)."""
        probs = np.array([1.0])
        bands = assign_risk_band(probs)
        assert bands[0] == "Critical", f"P=1.0 should be 'Critical', got '{bands[0]}'"

    def test_boundary_at_0_3(self):
        """0.3 is the INCLUSIVE lower bound of 'Medium', NOT 'Low'."""
        probs = np.array([0.3])
        bands = assign_risk_band(probs)
        assert bands[0] == "Medium", (
            f"P=0.3 (boundary) should be 'Medium', got '{bands[0]}'"
        )

    def test_boundary_at_0_6(self):
        """0.6 is the INCLUSIVE lower bound of 'High', NOT 'Medium'."""
        probs = np.array([0.6])
        bands = assign_risk_band(probs)
        assert bands[0] == "High", (
            f"P=0.6 (boundary) should be 'High', got '{bands[0]}'"
        )

    def test_all_outputs_are_valid_bands(self):
        """All output band names must be one of the 4 valid bands."""
        probs = np.linspace(0.0, 1.0, 101)  # 0.00, 0.01, ..., 1.00
        bands = assign_risk_band(probs)
        invalid = [b for b in bands if b not in self.VALID_BANDS]
        assert len(invalid) == 0, f"Invalid band names found: {invalid}"

    def test_length_preserved(self):
        """Output length must equal input length."""
        probs = np.random.uniform(0, 1, 50)
        bands = assign_risk_band(probs)
        assert len(bands) == 50, f"Expected 50 band assignments, got {len(bands)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for add_risk_band()
# ─────────────────────────────────────────────────────────────────────────────

class TestAddRiskBand:
    """Tests for the add_risk_band() function."""

    def test_adds_churn_band_column(self):
        """add_risk_band() should add a 'churn_band' column to the DataFrame."""
        df = pd.DataFrame({
            "customerID": ["C001", "C002", "C003"],
            "churn_probability": [0.1, 0.5, 0.9],
        })
        result = add_risk_band(df)
        assert "churn_band" in result.columns, (
            "add_risk_band() should add a 'churn_band' column"
        )

    def test_band_values_match_probabilities(self):
        """The assigned bands should match the probabilities."""
        df = pd.DataFrame({
            "churn_probability": [0.1, 0.45, 0.7, 0.9],
        })
        result = add_risk_band(df)
        assert result["churn_band"].iloc[0] == "Low"
        assert result["churn_band"].iloc[1] == "Medium"
        assert result["churn_band"].iloc[2] == "High"
        assert result["churn_band"].iloc[3] == "Critical"

    def test_does_not_modify_other_columns(self):
        """add_risk_band() should not change existing columns."""
        df = pd.DataFrame({
            "churn_probability": [0.5],
            "tenure": [12],
            "MonthlyCharges": [65.0],
        })
        result = add_risk_band(df)
        assert result["tenure"].iloc[0] == 12
        assert result["MonthlyCharges"].iloc[0] == 65.0
