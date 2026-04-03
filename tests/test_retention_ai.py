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

from retention_ai import _build_payload, _call_gemini, _parse_response, generate_retention_recommendations


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


def test_parse_response_extracts_embedded_json() -> None:
    raw = (
        "Here is your recommendation:\n"
        "```json\n"
        "{\"likely_churn_reason\": \"Price sensitivity\", "
        "\"risk_summary\": \"High risk\", "
        "\"retention_action\": \"Call customer\", "
        "\"offer_recommendation\": \"10% off\", "
        "\"communication_tone\": \"Empathetic\"}\n"
        "```"
    )

    parsed = json.loads(_parse_response(raw))

    assert parsed["likely_churn_reason"] == "Price sensitivity"
    assert parsed["communication_tone"] == "Empathetic"


def test_generate_recommendations_fallback_when_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    df = pd.DataFrame([
        {
            "churn_band": "High",
            "churn_probability": 0.82,
            "tenure": 2,
            "MonthlyCharges": 95.5,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "top_churn_drivers": "[]",
        },
        {
            "churn_band": "Low",
            "churn_probability": 0.12,
            "tenure": 40,
            "MonthlyCharges": 45.0,
            "Contract": "Two year",
            "PaymentMethod": "Bank transfer (automatic)",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "top_churn_drivers": "[]",
        },
    ])

    out = generate_retention_recommendations(df)

    high_rec = json.loads(out.loc[0, "retention_recommendation"])
    assert high_rec["likely_churn_reason"]
    assert high_rec["retention_action"]
    assert out.loc[1, "retention_recommendation"] == "Standard retention — low/medium risk customer."


def test_call_gemini_fails_fast_on_auth_error() -> None:
    class FailingModel:
        def generate_content(self, _prompt: str):
            raise Exception("400 API key expired. Please renew the API key. [reason: \"API_KEY_INVALID\"]")

    try:
        _call_gemini(FailingModel(), "test")
        assert False, "Expected RuntimeError for Gemini auth failure"
    except RuntimeError as exc:
        assert "authentication failed" in str(exc).lower()


def test_fallback_recommendation_uses_top_drivers(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    df = pd.DataFrame([
        {
            "churn_band": "Critical",
            "churn_probability": 0.91,
            "tenure": 1,
            "MonthlyCharges": 101.2,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "top_churn_drivers": json.dumps([
                {"feature": "Contract_Month-to-month", "impact": 0.4},
                {"feature": "TechSupport_No", "impact": 0.3},
            ]),
        }
    ])

    out = generate_retention_recommendations(df)
    rec = json.loads(out.loc[0, "retention_recommendation"])

    assert set(rec.keys()) == {
        "likely_churn_reason",
        "risk_summary",
        "retention_action",
        "offer_recommendation",
        "communication_tone",
    }
    assert "driver-specific steps" in rec["retention_action"].lower()
    assert "% loyalty discount" in rec["offer_recommendation"]
