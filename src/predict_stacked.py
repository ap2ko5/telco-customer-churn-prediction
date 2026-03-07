# -*- coding: utf-8 -*-
"""
predict_stacked.py — Full stacked inference CLI for the Stacked Churn Intelligence System.

Loads the four trained artifacts from train_pipeline.py:
  - preprocessing_pipeline.joblib
  - xgb_model_<run_id>.joblib      (or latest xgb_model_*.joblib)
  - nn_model_<run_id>.joblib
  - meta_model_<run_id>.joblib

Usage:
  python src/predict_stacked.py --input tenure=12 MonthlyCharges=85.0 Contract=Month-to-month

Optional:
  --run-id   <run_id>   Specify exact run to load (default: uses pipeline_info.json)
  --info     models/pipeline_info.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Make src/ importable when run from project root ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR, RISK_BANDS, MONTHLY_CHARGES_COL, TENURE_COL, ESTIMATED_CONTRACT_MONTHS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_kv_pairs(pairs: list[str]) -> dict:
    """
    Parse CLI key=value pairs into a typed Python dict.

    Handles the common Windows/PowerShell issue where a value with a space
    (e.g. 'Fiber optic') is split across multiple arguments.

    Examples
    --------
    parse_kv_pairs(["tenure=12", "MonthlyCharges=85.0", "Contract=Month-to-month"])
    → {"tenure": 12, "MonthlyCharges": 85.0, "Contract": "Month-to-month"}
    """
    data: dict = {}
    current_key: str | None = None

    for pair in pairs:
        if "=" not in pair:
            # Continuation of previous value (e.g. "Fiber optic" split by PowerShell)
            if current_key is None:
                raise ValueError(
                    f"Invalid argument '{pair}'. Expected key=value format."
                )
            data[current_key] = f"{data[current_key]} {pair}".strip()
            continue

        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        current_key = key

        # Auto type-casting
        lower = value.lower()
        if lower in {"true", "false"}:
            data[key] = lower == "true"
        else:
            try:
                data[key] = float(value) if "." in value else int(value)
            except ValueError:
                data[key] = value

    return data


def assign_risk_band(prob: float) -> str:
    """Return the risk band name for a single churn probability."""
    for band_name, (lo, hi) in RISK_BANDS.items():
        if lo <= prob < hi:
            return band_name
    return "Critical"  # Edge case: exactly 1.0


def estimate_revenue_loss(row: dict, prob: float) -> float:
    """
    Estimate expected revenue loss for this customer.

    Formula: P(churn) × MonthlyCharges × max(0, 24 − tenure)
    """
    monthly = float(row.get(MONTHLY_CHARGES_COL, 65.0))
    tenure  = float(row.get(TENURE_COL, 12))
    remaining = max(0.0, ESTIMATED_CONTRACT_MONTHS - tenure)
    return round(prob * monthly * remaining, 2)


def find_latest_artifact(prefix: str) -> Path:
    """
    Find the most recently saved model artifact by prefix.
    Falls back to the plain name (e.g. xgb_model.joblib) if no versioned file found.

    Parameters
    ----------
    prefix : str   e.g. "xgb_model"

    Returns
    -------
    Path to the latest matching .joblib file.
    """
    matches = sorted(MODELS_DIR.glob(f"{prefix}_*.joblib"))
    if matches:
        return matches[-1]          # Last alphabetically = latest by timestamp
    fallback = MODELS_DIR / f"{prefix}.joblib"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No model artifact found for prefix '{prefix}' in {MODELS_DIR}.\n"
        "Run train_pipeline.py first to generate model files."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts(run_id: str | None = None) -> tuple:
    """
    Load all four artifacts needed for stacked inference.

    Parameters
    ----------
    run_id : str or None
        If provided, load artifacts for that specific run.
        If None, automatically find the most recent version.

    Returns
    -------
    (preprocessor, xgb_model, nn_model, meta_model)
    """
    print("[predict_stacked] Loading artifacts...")

    # Preprocessing pipeline is always the same file (not versioned)
    preproc_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at {preproc_path}.\n"
            "Run train_pipeline.py first."
        )
    preprocessor = joblib.load(preproc_path)
    print(f"  Preprocessor  : {preproc_path}")

    if run_id:
        xgb_path  = MODELS_DIR / f"xgb_model_{run_id}.joblib"
        nn_path   = MODELS_DIR / f"nn_model_{run_id}.joblib"
        meta_path = MODELS_DIR / f"meta_model_{run_id}.joblib"
        for p in [xgb_path, nn_path, meta_path]:
            if not p.exists():
                raise FileNotFoundError(f"Artifact not found: {p}")
    else:
        xgb_path  = find_latest_artifact("xgb_model")
        nn_path   = find_latest_artifact("nn_model")
        meta_path = find_latest_artifact("meta_model")

    xgb_model  = joblib.load(xgb_path)
    nn_model   = joblib.load(nn_path)
    meta_model = joblib.load(meta_path)

    print(f"  XGBoost       : {xgb_path.name}")
    print(f"  Neural Net    : {nn_path.name}")
    print(f"  Meta-model    : {meta_path.name}")

    return preprocessor, xgb_model, nn_model, meta_model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(
    customer: dict,
    preprocessor,
    xgb_model,
    nn_model,
    meta_model,
) -> dict:
    """
    Run the full stacked inference for a single customer.

    Steps:
      1. Build a single-row DataFrame from the customer dict.
      2. Preprocess (using the already-fitted ColumnTransformer).
      3. Get P(churn) from XGBoost and Neural Network.
      4. Combine through the meta LogisticRegression → final stacked probability.
      5. Assign risk band and compute expected revenue loss.

    Returns
    -------
    dict with keys: churn_probability, risk_band, expected_revenue_loss,
                    xgb_probability, nn_probability
    """
    # Step 1 — single-row DataFrame
    X_new = pd.DataFrame([customer])

    # Step 2 — preprocess (apply the same transformations as training)
    X_t = preprocessor.transform(X_new)

    # Step 3 — base model probabilities
    p_xgb = float(xgb_model.predict_proba(X_t)[0, 1])
    p_nn  = float(nn_model.predict_proba(X_t)[0, 1])

    # Step 4 — stack (meta-model combines both)
    stacked = np.column_stack([[p_xgb], [p_nn]])
    p_final = float(meta_model.predict_proba(stacked)[0, 1])

    # Step 5 — post-processing
    band    = assign_risk_band(p_final)
    rev_loss = estimate_revenue_loss(customer, p_final)

    return {
        "churn_probability":    round(p_final, 4),
        "risk_band":            band,
        "expected_revenue_loss": rev_loss,
        "xgb_probability":      round(p_xgb, 4),
        "nn_probability":       round(p_nn,  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stacked Churn Inference — predict churn for a new customer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Customer features as key=value pairs.\n"
             "Example: --input tenure=12 MonthlyCharges=85.0 Contract=Month-to-month",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Specific run_id to load (e.g. 20260226_013000).\n"
             "If not provided, the most recent run's artifacts are used.",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Override the models directory path.",
    )
    args = parser.parse_args()

    # Parse customer features
    try:
        customer = parse_kv_pairs(args.input)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Load all 4 model artifacts
    try:
        preprocessor, xgb_model, nn_model, meta_model = load_artifacts(run_id=args.run_id)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Run inference
    print("\n[predict_stacked] Running stacked inference...")
    result = predict_single(customer, preprocessor, xgb_model, nn_model, meta_model)

    # ── Pretty print results ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  CHURN PREDICTION RESULT")
    print("=" * 55)
    print(f"  Churn Probability       : {result['churn_probability']:.1%}")
    print(f"  Risk Band               : {result['risk_band']}")
    from indian_currency import format_indian_currency
    print(f"  Expected Revenue Loss   : {format_indian_currency(result['expected_revenue_loss'])}")
    print("  -- Base Model Details ----------------------")
    print(f"  XGBoost probability     : {result['xgb_probability']:.4f}")
    print(f"  Neural Net probability  : {result['nn_probability']:.4f}")
    print(f"  Stacked (final)         : {result['churn_probability']:.4f}")
    print("=" * 55)

    # Actionable message
    if result["risk_band"] in ("High", "Critical"):
        print(f"\n[!] ACTION REQUIRED: {result['risk_band']} risk customer!")
        print("   -> Recommend AI-driven retention outreach immediately.\n")
    elif result["risk_band"] == "Medium":
        print("\n  Medium risk - consider standard retention campaign.\n")
    else:
        print("\n  Low risk - no immediate action required.\n")


if __name__ == "__main__":
    main()
