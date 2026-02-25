"""
retention_ai.py — AI-powered retention recommendations via Google Gemini.

For High and Critical risk customers only:
  - Sends a structured JSON payload to gemini-2.0-flash
  - Returns: churn reason, risk summary, retention action, offer, tone
  - Implements: batch processing, retry logic, rate-limit handling
  - Graceful fallback when API key is not set
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    GEMINI_BATCH_SIZE,
    GEMINI_MODEL,
    GEMINI_RATE_LIMIT_SLEEP,
    GEMINI_RETRY_MAX,
    GEMINI_RETRY_WAIT,
)

load_dotenv()

_PROMPT_TEMPLATE = """
You are an expert customer retention analyst for a telecom company.
Given the following customer profile and churn risk information, provide a comprehensive retention strategy.

Customer Profile:
{profile_json}

Churn Probability: {churn_probability:.2%}
Risk Band: {risk_band}

Respond ONLY with a valid JSON object (no markdown, no code fences) with these exact keys:
{{
  "likely_churn_reason": "...",
  "risk_summary": "...",
  "retention_action": "...",
  "offer_recommendation": "...",
  "communication_tone": "..."
}}
""".strip()


def _build_payload(row: pd.Series) -> dict:
    """Build the customer profile dict from a DataFrame row."""
    profile_fields = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "Contract", "InternetService", "PaymentMethod",
        "TechSupport", "OnlineSecurity", "StreamingTV",
        "StreamingMovies", "MultipleLines", "SeniorCitizen",
    ]
    profile = {}
    for field in profile_fields:
        # Flexible column match (case-insensitive alternatives)
        for col in row.index:
            if col.lower().replace("_", "") == field.lower().replace("_", ""):
                profile[field] = row[col]
                break

    return {
        "customer_profile": profile,
        "churn_probability": float(row.get("churn_probability", 0.0)),
        "risk_band": str(row.get("churn_band", "Unknown")),
    }


def _call_gemini(client, prompt: str) -> str:
    """Call Gemini with retry logic and rate-limit handling."""
    for attempt in range(1, GEMINI_RETRY_MAX + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as exc:
            err_str = str(exc).lower()
            if "quota" in err_str or "rate" in err_str or "429" in err_str:
                wait = GEMINI_RETRY_WAIT * attempt
                print(f"[retention_ai] Rate limit hit. Waiting {wait}s... (attempt {attempt}/{GEMINI_RETRY_MAX})")
                time.sleep(wait)
            else:
                print(f"[retention_ai] API error on attempt {attempt}: {exc}")
                if attempt == GEMINI_RETRY_MAX:
                    raise
                time.sleep(GEMINI_RETRY_WAIT)
    return "{}"


_REQUIRED_KEYS = {
    "likely_churn_reason",
    "risk_summary",
    "retention_action",
    "offer_recommendation",
    "communication_tone",
}


def _parse_response(raw: str) -> str:
    """Extract and validate JSON from Gemini response.

    Ensures all 5 required keys are present.
    Missing keys are filled with "N/A" and a warning is printed.
    Returns a compact JSON string ready for storage.
    """
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"[retention_ai] WARNING: Could not parse JSON from response. Storing raw text.")
        return raw

    # Schema validation — fill missing keys with "N/A"
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        print(f"[retention_ai] WARNING: Missing keys in Gemini response: {sorted(missing)}. Filling with 'N/A'.")
        for key in missing:
            data[key] = "N/A"

    return json.dumps(data, ensure_ascii=False)


def generate_retention_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'retention_recommendation' column to df.

    Only processes High and Critical risk customers.
    All others receive a default message.
    """
    df = df.copy()
    df["retention_recommendation"] = "Standard retention — low/medium risk customer."

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        print(
            "[retention_ai] [WARNING] GEMINI_API_KEY not set. "
            "Skipping AI recommendations. Set GEMINI_API_KEY in .env to enable."
        )
        df.loc[
            df["churn_band"].isin(["High", "Critical"]),
            "retention_recommendation",
        ] = "N/A (no API key configured)"
        return df

    try:
        import google.generativeai as genai
        client = genai.Client(api_key=api_key)
    except ImportError:
        print("[retention_ai] google-generativeai not installed. Run: pip install google-generativeai")
        return df

    # Filter High + Critical customers
    mask = df["churn_band"].isin(["High", "Critical"])
    high_risk_df = df[mask]
    n_customers = len(high_risk_df)

    if n_customers == 0:
        print("[retention_ai] No High/Critical risk customers found.")
        return df

    print(f"[retention_ai] Generating recommendations for {n_customers} high-risk customers...")

    indices  = high_risk_df.index.tolist()
    results  = {}

    # Process in batches
    for batch_start in tqdm(range(0, n_customers, GEMINI_BATCH_SIZE), desc="Gemini batches"):
        batch_indices = indices[batch_start : batch_start + GEMINI_BATCH_SIZE]

        for idx in batch_indices:
            row = df.loc[idx]
            payload = _build_payload(row)

            prompt = _PROMPT_TEMPLATE.format(
                profile_json=json.dumps(payload["customer_profile"], indent=2),
                churn_probability=payload["churn_probability"],
                risk_band=payload["risk_band"],
            )

            raw = _call_gemini(client, prompt)
            results[idx] = _parse_response(raw)

        # Rate-limit buffer between batches
        time.sleep(GEMINI_RATE_LIMIT_SLEEP)

    for idx, rec in results.items():
        df.at[idx, "retention_recommendation"] = rec

    print(f"[retention_ai] [OK] Recommendations generated for {len(results)} customers.")
    return df
