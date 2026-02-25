# -*- coding: utf-8 -*-
"""
evaluation.py — Compare XGBoost, Neural Network, and Stacked model.

Metrics: ROC-AUC, Precision, Recall, F1, Brier Score.
Selects best model based on ROC-AUC + Brier Score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    """
    Compute evaluation metrics for a single model.

    Returns a dict with: model, roc_auc, precision, recall, f1, brier_score
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model":       model_name,
        "roc_auc":     roc_auc_score(y_true, y_prob),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
    }
    return metrics


def compare_models(
    y_true: np.ndarray,
    prob_xgb: np.ndarray,
    prob_nn: np.ndarray,
    prob_stack: np.ndarray,
) -> pd.DataFrame:
    """
    Compare all three models and print a formatted table.

    Returns the results DataFrame sorted by ROC-AUC descending.
    """
    results = [
        evaluate_model(y_true, prob_xgb,   "XGBoost"),
        evaluate_model(y_true, prob_nn,     "Neural Network"),
        evaluate_model(y_true, prob_stack,  "Stacked Ensemble"),
    ]

    df = pd.DataFrame(results).set_index("model")
    df = df.sort_values("roc_auc", ascending=False)

    print("\n" + "=" * 65)
    print(" MODEL EVALUATION REPORT")
    print("=" * 65)
    float_cols = ["roc_auc", "precision", "recall", "f1", "brier_score"]
    print(df[float_cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 65)

    best = df.index[0]
    print(f"\n[Best] Best model by ROC-AUC: {best}  (AUC={df.loc[best,'roc_auc']:.4f})\n")

    return df


def print_classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n-- Classification Report: {model_name} --")
    print(classification_report(y_true, y_pred, digits=4))
