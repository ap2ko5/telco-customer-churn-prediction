"""
tests/conftest.py
=================
Shared pytest fixtures for the Stacked Churn Intelligence test suite.
All test files get access to these automatically — no import needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Ensure src/ is importable in all test files ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_probabilities() -> np.ndarray:
    """A representative spread of churn probabilities across all bands."""
    return np.array([0.05, 0.15, 0.29, 0.30, 0.45, 0.59, 0.60, 0.70, 0.79, 0.80, 0.95, 1.0])


@pytest.fixture()
def sample_churn_df() -> pd.DataFrame:
    """Minimal DataFrame with churn_probability column for integration tests."""
    return pd.DataFrame({
        "customerID":       ["C001", "C002", "C003", "C004"],
        "tenure":           [2, 12, 24, 36],
        "MonthlyCharges":   [85.50, 65.00, 45.00, 30.00],
        "TotalCharges":     [171.00, 780.00, 1080.00, 1080.00],
        "churn_probability": [0.92, 0.65, 0.35, 0.08],
    })


@pytest.fixture()
def minimal_raw_csv(tmp_path: Path) -> Path:
    """Write a minimal synthetic CSV for data_loader tests."""
    df = pd.DataFrame({
        "customerID":       ["A1", "A2", "A3", "A4", "A5"],
        "tenure":            [1, 6, 12, 24, 60],
        "MonthlyCharges":   [80.0, 60.0, 50.0, 40.0, 30.0],
        "TotalCharges":     ["80", "360", "600", "960", "1800"],  # string, as in real data
        "Contract":         ["Month-to-month", "Month-to-month", "One year", "Two year", "Two year"],
        "InternetService":  ["Fiber optic", "DSL", "DSL", "No", "No"],
        "Churn":            ["Yes", "No", "No", "No", "No"],
    })
    p = tmp_path / "test_churn.csv"
    df.to_csv(p, index=False)
    return p
