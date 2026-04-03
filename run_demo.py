#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_demo.py
===========
ONE-CLICK DEMO RUNNER
Executes the complete churn prediction pipeline and safely writes results to CSV
for Power BI consumption.

Usage (from command line):
  python run_demo.py
  python run_demo.py --data data/customer_churn.csv
  python run_demo.py --skip-ai      (skip AI recommendations for speed)

The script will:
  1. Load and preprocess customer data
  2. Run the stacked ML model
  3. Assign risk bands
  4. Calculate business impact
  5. Safely write predictions to outputs/churn_predictions.csv
  6. Generate reports and visualizations
  7. Report completion and file location

Exit code: 0 if successful, 1 if failed
"""
from __future__ import annotations

import argparse
import json
import os
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ── Configure logging early ───────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Make src/ importable ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Now safe to import from src/
from src.safe_csv_writer import POWERBI_STABLE_COLUMNS, enforce_powerbi_schema, safe_write_csv
from src.indian_currency import format_indian_currency


POWERBI_REQUIRED_COLUMNS = POWERBI_STABLE_COLUMNS


def _normalize_recommendation_text(value: object) -> str:
    """Convert recommendation payloads to BI-safe plain text."""
    text = "" if value is None else str(value).strip()
    if not text:
        return ""

    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            reason = str(data.get("likely_churn_reason", "")).strip()
            action = str(data.get("retention_action", "")).strip()
            offer = str(data.get("offer_recommendation", "")).strip()
            tone = str(data.get("communication_tone", "")).strip()
            parts = [
                f"Reason: {reason}" if reason else "",
                f"Action: {action}" if action else "",
                f"Offer: {offer}" if offer else "",
                f"Tone: {tone}" if tone else "",
            ]
            return " | ".join(p for p in parts if p)
        except Exception:
            return text

    return text


def _read_existing_output_csv(output_csv_path: Path) -> "pd.DataFrame":
    """Read an existing output CSV defensively, tolerating delimiter auto-detection."""
    import pandas as pd

    try:
        return pd.read_csv(output_csv_path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(output_csv_path)

# Dynamic imports from the main pipeline to avoid issues
def main() -> int:
    """
    Run the complete demo pipeline.

    Returns
    -------
    int
        Exit code: 0 if successful, 1 if error
    """
    import numpy as np

    from src.business_impact import compute_business_impact
    from src.config import (
        MODELS_DIR,
        OUTPUTS_DIR,
        PREDICTIONS_CSV,
    )
    from src.data_loader import load_data
    from src.explainability import add_top_churn_drivers
    from src.nn_model import predict_proba_nn
    from src.predict_stacked import load_artifacts
    from src.reporting import generate_all_reports
    from src.retention_ai import generate_retention_recommendations
    from src.risk_segmentation import add_risk_band
    from src.stacking import stack_predict
    from src.xgb_model import predict_proba_xgb

    parser = argparse.ArgumentParser(
        description="Churn Intelligence Demo Runner — One-click pipeline execution"
    )
    parser.add_argument(
        "--data", default="data/customer_churn.csv", help="Path to customer data CSV"
    )
    parser.add_argument(
        "--target", default="Churn", help="Target column name"
    )
    parser.add_argument(
        "--skip-ai", action="store_true", help="Skip AI retention recommendations (faster)"
    )
    parser.add_argument(
        "--mock-ai",
        action="store_true",
        help="Imitate Gemini response format locally and fill recommendations for all rows.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append new predictions to existing output CSV instead of replacing it.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CHURN INTELLIGENCE DEMO — ONE-CLICK PIPELINE")
    print("=" * 70 + "\n")

    pipeline_start = time.time()

    try:
        # ── Step 1: Load Data ──────────────────────────────────────────────────
        print("📥 Step 1: Loading Customer Data")
        X, _y, _numeric_cols, _categorical_cols = load_data(args.data, target_col=args.target)
        print(f"   ✓ Loaded {len(X):,} customers with {len(X.columns)} features\n")

        # ── Step 2: Load Matched Artifacts (preprocessor + models) ────────────
        print("⚙️  Step 2: Loading Matched Pipeline Artifacts")
        run_id = None
        pipeline_info_path = MODELS_DIR / "pipeline_info.json"
        if pipeline_info_path.exists():
            try:
                info = json.loads(pipeline_info_path.read_text(encoding="utf-8"))
                run_id = info.get("run_id")
            except Exception:
                run_id = None

        preprocessor, xgb_model, nn_model, meta_model = load_artifacts(
            run_id=run_id,
            models_dir=MODELS_DIR,
        )
        X_all_t = preprocessor.transform(X)
        print(f"   ✓ Matched artifacts loaded (run_id={run_id or 'latest'})")
        print(f"   ✓ Preprocessor applied (shape {X_all_t.shape})\n")

        # ── Step 4: Generate Predictions on Full Dataset ────────────────────────
        print("🎯 Step 4: Generating Churn Predictions")
        prob_xgb_all = predict_proba_xgb(xgb_model, X_all_t)
        prob_nn_all = predict_proba_nn(nn_model, X_all_t)
        final_probs = stack_predict(meta_model, prob_xgb_all, prob_nn_all)
        print(f"   ✓ Predictions generated for {len(X):,} customers (using pre-trained models)\n")

        # ── Step 5: Risk Band Assignment ───────────────────────────────────────
        print("🚨 Step 5: Risk Band Segmentation")
        result_df = X.copy().reset_index(drop=True)
        result_df["churn_probability"] = final_probs
        result_df["predicted_status"] = np.where(result_df["churn_probability"] >= 0.5, "Churned", "Unchurned")
        result_df = add_risk_band(result_df)
        print(f"   ✓ Risk bands assigned\n")

        # ── Step 6: Business Impact ───────────────────────────────────────────
        print("💰 Step 6: Business Impact Metrics")
        result_df = compute_business_impact(result_df)
        print(f"   ✓ Revenue at risk calculated\n")

        # ── Step 7: Per-Customer Explainability ────────────────────────────────
        print("🔍 Step 7: Per-Customer Explainability")
        def get_feature_names(preprocessor):
            names = []
            for name, transformer, cols in preprocessor.transformers_:
                if name == "num":
                    names.extend(cols)
                elif name == "cat":
                    try:
                        ohe = transformer.named_steps["onehot"]
                        cat_names = ohe.get_feature_names_out(cols).tolist()
                        names.extend(cat_names)
                    except Exception:
                        names.extend(cols)
            return names

        feature_names = get_feature_names(preprocessor)
        result_df = add_top_churn_drivers(
            df=result_df,
            xgb_model=xgb_model,
            X_transformed=X_all_t,
            feature_names=feature_names,
            top_n=3,
        )
        print(f"   ✓ Top drivers identified per customer\n")

        # ── Step 8: AI Retention Recommendations ──────────────────────────────
        if args.mock_ai:
            print("🤖 Step 8: Mock Gemini Recommendations (local structured output)")
            result_df = generate_retention_recommendations(
                result_df,
                include_all_bands=True,
                force_fallback=True,
            )
            print(f"   ✓ Mock Gemini-style recommendations generated for all customers\n")
        elif not args.skip_ai:
            print("🤖 Step 8: AI Retention Recommendations (Gemini)")
            result_df = generate_retention_recommendations(result_df)
            print(f"   ✓ Retention strategies generated\n")
        else:
            print("⏭️  Step 8: AI Recommendations (skipped)\n")
            result_df["retention_recommendation"] = "Standard retention"

        # ── Step 9: Reports & Visualizations ──────────────────────────────────
        print("📈 Step 9: Generating Reports & Visualizations")
        generate_all_reports(
            df=result_df,
            xgb_model=xgb_model,
            X_transformed=X_all_t,
            feature_names=feature_names,
        )
        print(f"   ✓ Reports and visualizations created\n")

        # ── Step 10: SAFE CSV WRITE ────────────────────────────────────────────
        print("💾 Step 10: SAFE CSV WRITE (Atomic File Replacement)")
        from src.safe_csv_writer import add_rupee_formatted_columns

        # Add formatted currency columns for Power BI
        result_df_for_export = add_rupee_formatted_columns(result_df)

        # Keep recommendations as readable plain text for stable BI ingestion.
        if "retention_recommendation" in result_df_for_export.columns:
            result_df_for_export["retention_recommendation"] = (
                result_df_for_export["retention_recommendation"].apply(_normalize_recommendation_text)
            )

        # Use a fixed path (or override via env var) so Power BI source stays constant.
        csv_override = os.getenv("POWERBI_CSV_PATH", "").strip()
        output_csv_path = Path(csv_override) if csv_override else PREDICTIONS_CSV

        # Enforce stable schema/order to avoid breaking Power BI mappings.
        result_df_for_export = enforce_powerbi_schema(
            result_df_for_export,
            POWERBI_REQUIRED_COLUMNS,
            keep_extra_columns=False,
        )

        col_order = result_df_for_export.columns.tolist()

        rows_before = 0
        rows_new = len(result_df_for_export)

        if args.append_output and output_csv_path.exists():
            existing_df = _read_existing_output_csv(output_csv_path)
            existing_df = enforce_powerbi_schema(
                existing_df,
                POWERBI_REQUIRED_COLUMNS,
                keep_extra_columns=False,
            )
            rows_before = len(existing_df)
            result_df_for_export = pd.concat(
                [existing_df[col_order], result_df_for_export[col_order]],
                ignore_index=True,
            )

        safe_write_csv(
            result_df_for_export[col_order],
            output_csv_path,
            columns_order=col_order,
            verbose=True,
        )

        if args.append_output:
            print(
                f"   ✓ Append mode: existing rows={rows_before:,}, added rows={rows_new:,}, total rows={len(result_df_for_export):,}"
            )
        print()

        # Calculate metrics for final report
        total_revenue_at_risk = result_df["expected_revenue_loss"].sum()
        high_risk_count = len(result_df[result_df["churn_band"].isin(["High", "Critical"])])
        critical_count = len(result_df[result_df["churn_band"] == "Critical"])
        avg_churn_prob = result_df["churn_probability"].mean()

        elapsed = time.time() - pipeline_start
        print("=" * 70)
        print("  ✅ DEMO PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\n📊 KEY METRICS:")
        print(f"   Total Customers      : {len(result_df):,}")
        print(f"   High-Risk Customers  : {high_risk_count:,}")
        print(f"   Critical Risk        : {critical_count:,}")
        print(f"   Avg Churn Probability: {avg_churn_prob:.1%}")
        print(f"   💰 Total Revenue at Risk : {format_indian_currency(total_revenue_at_risk)}")
        print(f"\n📁 Files Created:")
        print(f"   📊 Predictions   : {output_csv_path}")
        print(f"   📄 Summary report: {OUTPUTS_DIR / 'summary_report.txt'}")
        print(f"   📈 Visualizations: {OUTPUTS_DIR}/")
        print(f"\n⏱️  Total time: {elapsed:.1f} seconds\n")
        print("✨ Ready for Power BI! The CSV file is stable and fully written.\n")

        return 0

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        print("\n" + "=" * 70)
        print("  ❌ PIPELINE FAILED")
        print("=" * 70)
        print(f"\nError: {exc}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
