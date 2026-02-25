# -*- coding: utf-8 -*-
from __future__ import annotations
"""
train_pipeline.py - Main orchestration script for the Stacked Churn Intelligence System.

Usage:
  python src/train_pipeline.py --data data/customer_churn.csv --target Churn

Optional:
  --api-key YOUR_GEMINI_KEY   (or set GEMINI_API_KEY in .env)
  --no-calibration            Skip probability calibration
  --no-ai                     Skip AI retention recommendations
"""
import sys
import os
# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# ── Make src/ importable when run from project root ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from business_impact import compute_business_impact
from calibration import calibrate_probabilities
from config import (
    LOGS_DIR,
    META_LR_PARAMS,
    MODELS_DIR,
    N_FOLDS,
    NN_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
    XGB_PARAMS,
    XGB_EARLY_STOPPING_ROUNDS,
)
from data_loader import load_data
from evaluation import compare_models, print_classification_report
from nn_model import predict_proba_nn, train_nn
from preprocessor import build_preprocessor, fit_preprocessor, transform
from reporting import generate_all_reports
from retention_ai import generate_retention_recommendations
from risk_segmentation import add_risk_band
from stacking import generate_oof_predictions, stack_predict, train_meta_model
from xgb_model import predict_proba_xgb, train_xgb

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stacked Churn Intelligence Pipeline"
    )
    parser.add_argument("--data",   required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="Churn", help="Target column name")
    parser.add_argument("--api-key", default=None,  help="Gemini API key (overrides .env)")
    parser.add_argument("--no-calibration", action="store_true", help="Skip probability calibration")
    parser.add_argument("--no-ai",          action="store_true", help="Skip AI retention recommendations")
    parser.add_argument(
        "--calibration-method",
        default="isotonic",
        choices=["isotonic", "sigmoid"],
        help="Calibration method: 'isotonic' (default) or 'sigmoid' (Platt Scaling)",
    )
    return parser.parse_args()


def get_feature_names(preprocessor) -> list[str]:
    """Extract feature names from fitted ColumnTransformer."""
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


def main() -> None:
    pipeline_start = time.time()
    args = parse_args()

    # Override API key if provided via CLI
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_{run_id}.json"

    print("\n" + "=" * 65)
    print("  STACKED CHURN INTELLIGENCE SYSTEM")
    print(f"  Run ID: {run_id}")
    print("=" * 65 + "\n")

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    print("-- Step 1: Loading Data --")
    X, y, numeric_cols, categorical_cols = load_data(args.data, target_col=args.target)

    # ── 2. Train / Test Split ─────────────────────────────────────────────────
    print("\n-- Step 2: Train/Test Split --")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    print("\n-- Step 3: Preprocessing --")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor = fit_preprocessor(preprocessor, X_train, save=True)

    X_train_t = transform(preprocessor, X_train)
    X_test_t  = transform(preprocessor, X_test)
    X_all_t   = transform(preprocessor, X)          # full dataset for final preds

    y_train_arr = y_train.values
    y_test_arr  = y_test.values

    feature_names = get_feature_names(preprocessor)
    print(f"  Transformed train shape: {X_train_t.shape}")

    # ── 4. Stacking (OOF) ────────────────────────────────────────────────────
    print("\n-- Step 4: Generating OOF Predictions (5-Fold Stacking) --")
    neg, pos = int((y_train_arr == 0).sum()), int((y_train_arr == 1).sum())
    oof_xgb, oof_nn = generate_oof_predictions(X_train_t, y_train_arr, neg / max(pos, 1))

    # ── 5. Train Meta-Model ──────────────────────────────────────────────────
    print("\n-- Step 5: Training Meta-Model (Logistic Regression) --")
    meta_model = train_meta_model(
        oof_xgb, oof_nn, y_train_arr,
        save=True,
        model_name=f"meta_model_{run_id}",
    )

    # ── 6. Retrain Base Models on Full Training Set ──────────────────────────
    print("\n-- Step 6: Retraining Base Models on Full Training Data --")
    xgb_model = train_xgb(
        X_train_t, y_train_arr,
        X_val=X_test_t, y_val=y_test_arr,
        save=True,
        model_name=f"xgb_model_{run_id}",
    )
    nn_model = train_nn(
        X_train_t, y_train_arr,
        save=True,
        model_name=f"nn_model_{run_id}",
    )

    # ── 7. Generate Test Probabilities ────────────────────────────────────────
    print("\n-- Step 7: Generating Test Set Probabilities --")
    prob_xgb_test   = predict_proba_xgb(xgb_model, X_test_t)
    prob_nn_test    = predict_proba_nn(nn_model, X_test_t)
    prob_stack_test = stack_predict(meta_model, prob_xgb_test, prob_nn_test)

    # ── 8. Model Evaluation ───────────────────────────────────────────────────
    print("\n-- Step 8: Model Evaluation --")
    eval_df = compare_models(y_test_arr, prob_xgb_test, prob_nn_test, prob_stack_test)
    print_classification_report(y_test_arr, prob_stack_test, "Stacked Ensemble")

    # ── 9. Calibration ────────────────────────────────────────────────────────
    prob_xgb_all   = predict_proba_xgb(xgb_model, X_all_t)
    prob_nn_all    = predict_proba_nn(nn_model, X_all_t)
    prob_stack_all = stack_predict(meta_model, prob_xgb_all, prob_nn_all)

    if not args.no_calibration:
        cal_method = args.calibration_method
        print(f"\n-- Step 9: Probability Calibration ({cal_method}) --")
        # Use test set for calibration (held-out from training)
        calibrated_fn = calibrate_probabilities(
            X_cal=np.column_stack([prob_xgb_test, prob_nn_test]),
            y_cal=y_test_arr,
            prob_fn=lambda Xc: stack_predict(meta_model, Xc[:, 0], Xc[:, 1]),
            method=cal_method,
        )
        final_probs = calibrated_fn(np.column_stack([prob_xgb_all, prob_nn_all]))
    else:
        print("\n-- Step 9: Calibration skipped (--no-calibration) --")
        final_probs = prob_stack_all

    # ── 10. Risk Band Segmentation ────────────────────────────────────────────
    print("\n-- Step 10: Risk Band Segmentation --")
    result_df = X.copy().reset_index(drop=True)
    result_df["churn_probability"] = final_probs
    result_df = add_risk_band(result_df)

    # ── 11. Business Impact ───────────────────────────────────────────────────
    print("\n-- Step 11: Business Impact Metrics --")
    result_df = compute_business_impact(result_df)

    # ── 12. AI Retention Recommendations ─────────────────────────────────────
    if not args.no_ai:
        print("\n-- Step 12: AI Retention Recommendations --")
        result_df = generate_retention_recommendations(result_df)
    else:
        print("\n-- Step 12: AI Recommendations skipped (--no-ai) --")
        result_df["retention_recommendation"] = "Skipped"

    # ── 13. Reporting & Visualizations ───────────────────────────────────────
    print("\n-- Step 13: Generating Reports & Visualizations --")
    generate_all_reports(
        df=result_df,
        xgb_model=xgb_model,
        X_transformed=X_all_t,
        feature_names=feature_names,
    )

    # ── 14. Log Run Metadata ──────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    log_data = {
        "run_id":            run_id,
        "timestamp":         datetime.now().isoformat(),
        "total_time_s":      round(elapsed, 2),
        "dataset":           args.data,
        "target":            args.target,
        "calibration_method": getattr(args, "calibration_method", "isotonic"),
        "n_samples":         len(X),
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_features_raw":    len(X.columns),
        "n_features_after":  X_train_t.shape[1],
        "numeric_columns":   numeric_cols,
        "categorical_columns": categorical_cols,
        "xgb_params":        XGB_PARAMS,
        "xgb_early_stopping": XGB_EARLY_STOPPING_ROUNDS,
        "nn_params":         NN_PARAMS,
        "meta_lr_params":    META_LR_PARAMS,
        "n_folds":           N_FOLDS,
        "random_state":      RANDOM_STATE,
        "calibration":       not args.no_calibration,
        "model_artifacts": {
            "xgb":  f"xgb_model_{run_id}.joblib",
            "nn":   f"nn_model_{run_id}.joblib",
            "meta": f"meta_model_{run_id}.joblib",
            "preprocessor": "preprocessing_pipeline.joblib",
        },
        "evaluation": {
            row["model"]: {
                k: round(v, 4) for k, v in row.items() if k != "model"
            }
            for row in eval_df.reset_index().to_dict("records")
        },
    }

    log_path.write_text(json.dumps(log_data, indent=2, default=str), encoding="utf-8")

    # ── Save pipeline_info.json for inference scripts ─────────────────────────
    pipeline_info = {
        "run_id":            run_id,
        "target_column":     args.target,
        "feature_columns":   X.columns.tolist(),
        "numeric_columns":   numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_names_after_ohe": feature_names,
        "model_artifacts":   log_data["model_artifacts"],
        "xgb_params":        XGB_PARAMS,
        "nn_params":         NN_PARAMS,
        "meta_lr_params":    META_LR_PARAMS,
        "calibration_method": getattr(args, "calibration_method", "isotonic"),
    }
    info_path = MODELS_DIR / "pipeline_info.json"
    info_path.write_text(json.dumps(pipeline_info, indent=2, default=str), encoding="utf-8")
    print(f"[train_pipeline] Pipeline info saved → {info_path}")

    print("\n" + "=" * 65)
    print("  [OK] PIPELINE COMPLETE")
    print(f"  Total time      : {elapsed:.1f}s")
    print(f"  Run ID          : {run_id}")
    print(f"  Log             : {log_path}")
    print(f"  Pipeline info   : models/pipeline_info.json")
    print(f"  Predictions     : outputs/churn_predictions.csv")
    print(f"  Summary report  : outputs/summary_report.txt")
    print(f"  Visualizations  : outputs/")
    print(f"  Models          : models/xgb_model_{run_id}.joblib")
    print(f"                    models/nn_model_{run_id}.joblib")
    print(f"                    models/meta_model_{run_id}.joblib")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
