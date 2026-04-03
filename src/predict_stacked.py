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

from config import MODELS_DIR as DEFAULT_MODELS_DIR, RISK_BANDS, MONTHLY_CHARGES_COL, TENURE_COL, ESTIMATED_CONTRACT_MONTHS
from indian_currency import format_indian_currency
from safe_csv_writer import DEFAULT_CSV_DELIMITER, POWERBI_STABLE_COLUMNS, enforce_powerbi_schema, safe_write_csv

DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent.parent / "outputs" / "one_customer_prediction.csv"
POWERBI_REQUIRED_COLUMNS = POWERBI_STABLE_COLUMNS


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


def load_customer_from_json(input_file: Path) -> dict:
    """Load one-customer payload from a JSON file."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be a single object of key/value pairs.")

    return payload


def get_required_feature_columns(models_dir: Path) -> list[str]:
    """Read expected raw feature columns from pipeline_info.json if available."""
    info_path = models_dir / "pipeline_info.json"
    if not info_path.exists():
        return []

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    cols = info.get("feature_columns")
    return cols if isinstance(cols, list) else []


def validate_customer_fields(customer: dict, required_cols: list[str]) -> None:
    """Validate required feature fields and raise readable error for missing keys."""
    if not required_cols:
        return

    missing = [c for c in required_cols if c not in customer]
    if missing:
        raise ValueError(
            "Missing required customer fields: "
            + ", ".join(missing)
            + "\nTip: create a JSON file with all required fields from models/pipeline_info.json -> feature_columns."
        )


def build_single_customer_export_row(customer: dict, result: dict) -> pd.DataFrame:
    """Build a one-row DataFrame matching Power BI-friendly schema conventions."""
    row = dict(customer)
    row.update(result)
    row["churn_band"] = result.get("risk_band", "")
    row["retention_recommendation"] = "Manual single-customer test run."
    row["top_churn_drivers"] = "[]"

    # Add rupee-formatted display helpers used by dashboards.
    row["expected_revenue_loss_rupees"] = format_indian_currency(float(result["expected_revenue_loss"]))
    if "MonthlyCharges" in row:
        row["monthly_charges_rupees"] = format_indian_currency(float(row["MonthlyCharges"]))
    if "TotalCharges" in row:
        row["total_charges_rupees"] = format_indian_currency(float(row["TotalCharges"]))

    df = pd.DataFrame([row])
    return enforce_powerbi_schema(df, POWERBI_REQUIRED_COLUMNS, keep_extra_columns=False)


def append_or_create_prediction_csv(output_csv: Path, new_row_df: pd.DataFrame) -> pd.DataFrame:
    """Append new prediction row to existing CSV, or create a new file if it doesn't exist."""
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv, sep=None, engine="python")

        # Preserve old columns and include any new columns deterministically.
        all_cols = list(dict.fromkeys(existing_df.columns.tolist() + new_row_df.columns.tolist()))
        existing_df = existing_df.reindex(columns=all_cols)
        new_row_df = new_row_df.reindex(columns=all_cols)

        return pd.concat([existing_df, new_row_df], ignore_index=True)

    return new_row_df


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


def find_latest_artifact(prefix: str, models_dir: Path) -> Path:
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
    matches = sorted(models_dir.glob(f"{prefix}_*.joblib"))
    if matches:
        return matches[-1]          # Last alphabetically = latest by timestamp
    fallback = models_dir / f"{prefix}.joblib"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No model artifact found for prefix '{prefix}' in {models_dir}.\n"
        "Run train_pipeline.py first to generate model files."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts(run_id: str | None = None, models_dir: Path = DEFAULT_MODELS_DIR) -> tuple:
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
    preproc_path = models_dir / "preprocessing_pipeline.joblib"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at {preproc_path}.\n"
            "Run train_pipeline.py first."
        )
    preprocessor = joblib.load(preproc_path)
    print(f"  Preprocessor  : {preproc_path}")

    if run_id:
        xgb_path  = models_dir / f"xgb_model_{run_id}.joblib"
        nn_path   = models_dir / f"nn_model_{run_id}.joblib"
        meta_path = models_dir / f"meta_model_{run_id}.joblib"
        for p in [xgb_path, nn_path, meta_path]:
            if not p.exists():
                raise FileNotFoundError(f"Artifact not found: {p}")
    else:
        xgb_path  = find_latest_artifact("xgb_model", models_dir)
        nn_path   = find_latest_artifact("nn_model", models_dir)
        meta_path = find_latest_artifact("meta_model", models_dir)

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
        "--input", nargs="+", required=False,
        help="Customer features as key=value pairs.\n"
             "Example: --input tenure=12 MonthlyCharges=85.0 Contract=Month-to-month",
    )
    parser.add_argument(
        "--input-file", default=None,
        help="Path to one-customer JSON file (cleaner than long --input commands).",
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
    parser.add_argument(
        "--write-csv",
        nargs="?",
        const=str(DEFAULT_OUTPUT_CSV),
        default=None,
        help="Optionally append this one-customer result to a CSV file (default: outputs/one_customer_prediction.csv).",
    )
    args = parser.parse_args()

    if not args.input and not args.input_file:
        print("[ERROR] Provide either --input key=value ... OR --input-file path/to/customer.json")
        sys.exit(1)

    if args.input and args.input_file:
        print("[ERROR] Use only one input mode: --input or --input-file")
        sys.exit(1)

    # Parse customer features
    try:
        if args.input_file:
            customer = load_customer_from_json(Path(args.input_file).expanduser().resolve())
        else:
            customer = parse_kv_pairs(args.input)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else DEFAULT_MODELS_DIR

    # Validate keys early for friendlier error messages.
    try:
        required_cols = get_required_feature_columns(models_dir)
        validate_customer_fields(customer, required_cols)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Load all 4 model artifacts
    try:
        preprocessor, xgb_model, nn_model, meta_model = load_artifacts(
            run_id=args.run_id,
            models_dir=models_dir,
        )
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

    # Optional: write one-customer result directly to BI-connected CSV.
    if args.write_csv:
        output_csv = Path(args.write_csv).expanduser().resolve()
        new_row_df = build_single_customer_export_row(customer, result)
        final_df = append_or_create_prediction_csv(output_csv, new_row_df)
        try:
            safe_write_csv(
                final_df,
                output_csv,
                columns_order=final_df.columns.tolist(),
                verbose=True,
                delimiter=DEFAULT_CSV_DELIMITER,
            )
            print(f"[predict_stacked] One-customer result appended → {output_csv} (rows={len(final_df)})")
        except PermissionError:
            print(
                "[ERROR] Could not write CSV because it is locked by another app.\n"
                f"Close the file in Power BI/Excel and retry: {output_csv}"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
