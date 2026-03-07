"""
tests/test_business_impact.py
==============================
Unit tests for src/business_impact.py

What we test:
  1. Revenue formula: P(churn) × MonthlyCharges × max(0, 24 − tenure)
  2. Output DataFrame is sorted by expected_revenue_loss descending
  3. Missing MonthlyCharges column → uses $65 fallback
  4. Tenure > 24 months → remaining = 0 (no negative remaining)
  5. expected_revenue_loss column is rounded to 2 decimal places

Run with:
  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from business_impact import compute_business_impact
from config import ESTIMATED_CONTRACT_MONTHS, FALLBACK_MONTHLY_CHARGES


# ─────────────────────────────────────────────────────────────────────────────
# Helper to build minimal DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def make_df(tenures, monthly_charges, churn_probs):
    """Create a minimal DataFrame for business impact testing."""
    return pd.DataFrame({
        "tenure":           tenures,
        "MonthlyCharges":   monthly_charges,
        "churn_probability": churn_probs,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRevenueFormula:
    """Tests for the core revenue formula."""

    def test_exact_formula_calculation(self):
        """
        Formula: P(churn) × MonthlyCharges × max(0, ESTIMATED_CONTRACT_MONTHS − tenure)
        Example: 0.8 × 90 × (24 − 6) = 0.8 × 90 × 18 = 1296.0
        """
        df = make_df(
            tenures=[6],
            monthly_charges=[90.0],
            churn_probs=[0.8],
        )
        result = compute_business_impact(df)
        expected = round(0.8 * 90.0 * (ESTIMATED_CONTRACT_MONTHS - 6), 2)
        actual   = result["expected_revenue_loss"].iloc[0]
        assert actual == expected, (
            f"Expected revenue loss {expected}, got {actual}"
        )

    def test_zero_churn_probability_gives_zero_loss(self):
        """A customer with 0% churn probability should have ₹0 expected loss."""
        df = make_df(tenures=[12], monthly_charges=[75.0], churn_probs=[0.0])
        result = compute_business_impact(df)
        assert result["expected_revenue_loss"].iloc[0] == 0.0

    def test_high_tenure_gives_zero_remaining_months(self):
        """
        A customer with tenure > ESTIMATED_CONTRACT_MONTHS (24 months)
        should have remaining = 0, giving expected_revenue_loss = 0.0.
        """
        df = make_df(tenures=[36], monthly_charges=[80.0], churn_probs=[0.9])
        result = compute_business_impact(df)
        # 24 - 36 = -12 → clipped to 0 → 0.9 × 80 × 0 = 0
        assert result["expected_revenue_loss"].iloc[0] == 0.0, (
            "Tenure > contract length should result in ₹0 revenue loss"
        )

    def test_exactly_at_contract_end(self):
        """Customer at exactly 24 months tenure → remaining = 0."""
        df = make_df(
            tenures=[ESTIMATED_CONTRACT_MONTHS],
            monthly_charges=[65.0],
            churn_probs=[0.7],
        )
        result = compute_business_impact(df)
        assert result["expected_revenue_loss"].iloc[0] == 0.0


class TestSortOrder:
    """Tests for output sort order."""

    def test_sorted_descending_by_revenue_loss(self):
        """
        Output DataFrame should be sorted by expected_revenue_loss in descending
        order — highest-risk customer first.
        """
        df = make_df(
            tenures=      [12,   1,   6,   18],
            monthly_charges=[50.0, 95.0, 80.0, 40.0],
            churn_probs=  [0.3,  0.9,  0.7,  0.2],
        )
        result = compute_business_impact(df)
        losses = result["expected_revenue_loss"].tolist()
        assert losses == sorted(losses, reverse=True), (
            f"Output should be sorted descending. Got: {losses}"
        )

    def test_output_has_expected_revenue_loss_column(self):
        """The output DataFrame must contain an 'expected_revenue_loss' column."""
        df = make_df(
            tenures=[6, 12],
            monthly_charges=[60.0, 80.0],
            churn_probs=[0.5, 0.8],
        )
        result = compute_business_impact(df)
        assert "expected_revenue_loss" in result.columns, (
            "Output DataFrame must have 'expected_revenue_loss' column"
        )


class TestFallbackBehavior:
    """Tests for missing data fallback behavior."""

    def test_missing_monthly_charges_column_uses_fallback(self):
        """
        If 'MonthlyCharges' column is missing entirely, the function should
        use FALLBACK_MONTHLY_CHARGES (e.g. ₹65.0) without raising an error.
        """
        df = pd.DataFrame({
            "tenure":            [6],
            "churn_probability": [0.8],
            # No 'MonthlyCharges' column!
        })
        result = compute_business_impact(df)
        expected = round(0.8 * FALLBACK_MONTHLY_CHARGES * (ESTIMATED_CONTRACT_MONTHS - 6), 2)
        actual   = result["expected_revenue_loss"].iloc[0]
        assert actual == expected, (
            f"Expected fallback calculation {expected}, got {actual}"
        )

    def test_nan_monthly_charges_uses_fallback(self):
        """
        NaN values in 'MonthlyCharges' should be filled with FALLBACK_MONTHLY_CHARGES.
        """
        df = make_df(
            tenures=[6],
            monthly_charges=[np.nan],
            churn_probs=[1.0],
        )
        result = compute_business_impact(df)
        expected = round(1.0 * FALLBACK_MONTHLY_CHARGES * (ESTIMATED_CONTRACT_MONTHS - 6), 2)
        actual   = result["expected_revenue_loss"].iloc[0]
        assert actual == expected

    def test_values_rounded_to_2_decimal_places(self):
        """expected_revenue_loss values should be rounded to 2 decimal places."""
        df = make_df(
            tenures=[7],
            monthly_charges=[33.33],
            churn_probs=[0.7777],
        )
        result = compute_business_impact(df)
        loss = result["expected_revenue_loss"].iloc[0]
        # Check that rounding is applied
        assert loss == round(loss, 2), (
            f"expected_revenue_loss should be rounded to 2 decimal places, got {loss}"
        )
