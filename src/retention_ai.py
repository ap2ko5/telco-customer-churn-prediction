"""
retention_ai.py — AI-powered retention recommendations via Google Gemini.

For High and Critical risk customers only:
  - Sends a structured JSON payload to gemini-2.5-flash
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

Top Churn Drivers (model explainability):
{top_drivers_json}

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
                value = row[col]
                # Convert numpy/pandas types to native Python types for JSON serialization
                if hasattr(value, 'item'):
                    value = value.item()
                profile[field] = value
                break

    raw_drivers = row.get("top_churn_drivers", "[]")
    if isinstance(raw_drivers, str):
        try:
            parsed_drivers = json.loads(raw_drivers)
        except json.JSONDecodeError:
            parsed_drivers = []
    elif isinstance(raw_drivers, list):
        parsed_drivers = raw_drivers
    else:
        parsed_drivers = []

    return {
        "customer_profile": profile,
        "churn_probability": float(row.get("churn_probability", 0.0)),
        "risk_band": str(row.get("churn_band", "Unknown")),
        "top_churn_drivers": parsed_drivers,
    }


def _call_gemini(model, prompt: str) -> str:
    """Call Gemini with retry logic and rate-limit handling."""
    for attempt in range(1, GEMINI_RETRY_MAX + 1):
        try:
            response = model.generate_content(prompt)
            return _extract_text_from_gemini_response(response)
        except Exception as exc:
            err_str = str(exc).lower()
            # Authentication/authorization errors are not retriable.
            if (
                "api_key_invalid" in err_str
                or "api key expired" in err_str
                or "invalid api key" in err_str
                or "401" in err_str
                or "permission_denied" in err_str
            ):
                raise RuntimeError(f"Gemini authentication failed: {exc}") from exc

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


def _generate_fallback_recommendation(payload: dict) -> str:
    """Generate deterministic fallback recommendation JSON.

    Used when Gemini is unavailable or returns unparsable content.
    """
    profile = payload.get("customer_profile", {})
    churn_probability = float(payload.get("churn_probability", 0.0))
    risk_band = str(payload.get("risk_band", "Unknown"))

    tenure = float(profile.get("tenure", 0) or 0)
    monthly_charges = float(profile.get("MonthlyCharges", 0) or 0)
    contract = str(profile.get("Contract", "Unknown"))
    payment = str(profile.get("PaymentMethod", "Unknown"))
    tech_support = str(profile.get("TechSupport", "No"))
    online_security = str(profile.get("OnlineSecurity", "No"))

    if tenure <= 3:
        churn_reason = "Customer is very new and may not have seen sustained value yet."
    elif contract == "Month-to-month":
        churn_reason = "No long-term lock-in creates high switching risk."
    elif monthly_charges >= 80:
        churn_reason = "Higher monthly bill may be driving price sensitivity."
    elif payment == "Electronic check":
        churn_reason = "Payment method may add friction versus auto-pay options."
    else:
        churn_reason = "Churn risk appears driven by a combination of service and pricing signals."

    risk_factors = []
    if tenure <= 6:
        risk_factors.append(f"short tenure ({int(tenure)} months)")
    if contract == "Month-to-month":
        risk_factors.append("month-to-month contract")
    if monthly_charges >= 70:
        risk_factors.append(f"high monthly charges ({monthly_charges:.2f})")
    if payment == "Electronic check":
        risk_factors.append("electronic-check billing")
    if tech_support == "No" and online_security == "No":
        risk_factors.append("no support/security add-ons")

    if not risk_factors:
        risk_factors.append("multiple moderate risk indicators")

    risk_summary = (
        f"{risk_band} risk at {churn_probability:.1%}; key factors: "
        + ", ".join(risk_factors)
        + "."
    )

    if tenure <= 3:
        action = (
            "Initiate outreach within 24 hours, run onboarding review, and assign priority support "
            "for 60-90 days."
        )
        offer = "Provide a temporary bill credit and 2-3 months of premium add-ons at no extra cost."
    elif contract == "Month-to-month":
        action = "Pitch annual contract migration with clear savings and a frictionless upgrade path."
        offer = "Offer 15-20% annual-plan discount plus one-time migration bonus credit."
    elif monthly_charges >= 80:
        action = "Perform plan-rightsizing consultation and offer a loyalty pricing adjustment."
        offer = "Offer 10-15% discount for 6 months with optional bundled service optimization."
    else:
        action = "Schedule a retention call to identify pain points and personalize next-best action."
        offer = "Provide a targeted loyalty offer based on usage profile and tenure."

    tone = (
        "Urgent and empathetic"
        if churn_probability >= 0.85 or risk_band == "Critical"
        else "Consultative and proactive"
    )

    recommendation = {
        "likely_churn_reason": churn_reason,
        "risk_summary": risk_summary,
        "retention_action": action,
        "offer_recommendation": offer,
        "communication_tone": tone,
    }
    return json.dumps(recommendation, ensure_ascii=False)


def _extract_json_candidate(text: str) -> dict | None:
    """Try to parse JSON object directly or from embedded text."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract first balanced-looking JSON object by braces.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _parse_response(raw: str, fallback_payload: dict | None = None) -> str:
    """Extract and validate JSON from Gemini response.

    Ensures all 5 required keys are present.
    Missing keys are filled with "N/A" and a warning is printed.
    Returns a compact JSON string ready for storage.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    data = _extract_json_candidate(text)
    if data is None:
        print("[retention_ai] WARNING: Could not parse JSON from response.")
        if fallback_payload is not None:
            print("[retention_ai] Using fallback recommendation for this customer.")
            return _generate_fallback_recommendation(fallback_payload)
        return raw

    # Schema validation — fill missing keys with "N/A"
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        print(f"[retention_ai] WARNING: Missing keys in Gemini response: {sorted(missing)}. Filling with 'N/A'.")
        for key in missing:
            data[key] = "N/A"

    return json.dumps(data, ensure_ascii=False)


def _extract_text_from_gemini_response(response) -> str:
    """Best-effort extraction for Gemini SDK response payloads."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())

    if chunks:
        return "\n".join(chunks)

    raise ValueError("Gemini returned an empty/blocked response without text parts.")


def generate_retention_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'retention_recommendation' column to df.

    Only processes High and Critical risk customers.
    All others receive a default message.
    """
    df = df.copy()
    df["retention_recommendation"] = "Standard retention — low/medium risk customer."

    api_key = os.getenv("GEMINI_API_KEY", "")
    high_risk_mask = df["churn_band"].isin(["High", "Critical"])

    if not api_key or api_key == "your_gemini_api_key_here":
        print(
            "[retention_ai] [WARNING] GEMINI_API_KEY not set. "
            "Using fallback recommendations for High/Critical risk customers. "
            "Set GEMINI_API_KEY in .env to enable Gemini-generated recommendations."
        )
        for idx in df[high_risk_mask].index:
            payload = _build_payload(df.loc[idx])
            df.at[idx, "retention_recommendation"] = _generate_fallback_recommendation(payload)
        return df

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
    except ImportError:
        print(
            "[retention_ai] google-generativeai not installed. "
            "Using fallback recommendations for High/Critical customers."
        )
        for idx in df[high_risk_mask].index:
            payload = _build_payload(df.loc[idx])
            df.at[idx, "retention_recommendation"] = _generate_fallback_recommendation(payload)
        return df

    # Filter High + Critical customers
    high_risk_df = df[high_risk_mask]
    n_customers = len(high_risk_df)

    if n_customers == 0:
        print("[retention_ai] No High/Critical risk customers found.")
        return df

    print(f"[retention_ai] Generating recommendations for {n_customers} high-risk customers...")

    indices  = high_risk_df.index.tolist()
    results  = {}

    # Process in batches
    gemini_enabled = True
    for batch_start in tqdm(range(0, n_customers, GEMINI_BATCH_SIZE), desc="Gemini batches"):
        batch_indices = indices[batch_start : batch_start + GEMINI_BATCH_SIZE]

        for idx in batch_indices:
            row = df.loc[idx]
            payload = _build_payload(row)

            prompt = _PROMPT_TEMPLATE.format(
                profile_json=json.dumps(payload["customer_profile"], indent=2),
                churn_probability=payload["churn_probability"],
                risk_band=payload["risk_band"],
                top_drivers_json=json.dumps(payload["top_churn_drivers"], indent=2),
            )

            if not gemini_enabled:
                results[idx] = _generate_fallback_recommendation(payload)
                continue

            try:
                raw = _call_gemini(model, prompt)
                results[idx] = _parse_response(raw, fallback_payload=payload)
            except Exception as exc:
                print(f"[retention_ai] WARNING: Gemini call failed for index {idx}: {exc}")
                if "authentication failed" in str(exc).lower():
                    print("[retention_ai] Disabling Gemini for this run and using fallback recommendations.")
                    gemini_enabled = False
                results[idx] = _generate_fallback_recommendation(payload)

        # Rate-limit buffer between batches
        time.sleep(GEMINI_RATE_LIMIT_SLEEP)

    for idx, rec in results.items():
        df.at[idx, "retention_recommendation"] = rec

    print(f"[retention_ai] [OK] Recommendations generated for {len(results)} customers.")
    return df
