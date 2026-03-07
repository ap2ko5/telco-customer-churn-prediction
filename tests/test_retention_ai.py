"""
tests/test_retention_ai.py
==========================
Unit tests for explainability-guided payload construction in retention_ai.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retention_ai import _build_payload


def test_build_payload_parses_top_churn_drivers_json() -> None:
    row = pd.Series({
        "tenure": 3,
        "MonthlyCharges": 89.5,
        "churn_probability": 0.82,
        "churn_band": "High",
        "top_churn_drivers": json.dumps([
            {"feature": "Contract_Month-to-month", "impact": 0.42},
            {"feature": "tenure", "impact": 0.19},
        ]),
    })

    payload = _build_payload(row)

    assert payload["risk_band"] == "High"
    assert payload["churn_probability"] == 0.82
    assert isinstance(payload["top_churn_drivers"], list)
    assert len(payload["top_churn_drivers"]) == 2
    assert payload["top_churn_drivers"][0]["feature"] == "Contract_Month-to-month"


def test_build_payload_handles_invalid_top_churn_drivers_json() -> None:
    row = pd.Series({
        "churn_probability": 0.91,
        "churn_band": "Critical",
        "top_churn_drivers": "not-json",
    })

    payload = _build_payload(row)

    assert payload["risk_band"] == "Critical"
    assert payload["top_churn_drivers"] == []
