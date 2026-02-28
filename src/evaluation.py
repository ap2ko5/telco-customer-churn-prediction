"""
evaluation.py
=============
Compare XGBoost, Neural Network, and Stacked model performance.

Metrics reported: ROC-AUC, Precision, Recall, F1, Brier Score.

Why these metrics?
  - ROC-AUC   : threshold-agnostic discriminative power
  - Precision  : of predicted churners, how many actually churn?
  - Recall     : of actual churners, how many did we catch?
  - F1         : harmonic mean of precision/recall (class-imbalance robust)
  - Brier Score: mean squared error of probabilities (lower = better calibrated)
"""
from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    model_name: str,
    threshold:  float = 0.5,
) -> dict:
    """
    Compute evaluation metrics for a single model.

    Returns
    -------
    dict with keys: model, roc_auc, precision, recall, f1, brier_score
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model":       model_name,
        "roc_auc":     round(roc_auc_score(y_true, y_prob),                          4),
        "precision":   round(precision_score(y_true, y_pred, zero_division=0),        4),
        "recall":      round(recall_score(y_true, y_pred, zero_division=0),            4),
        "f1":          round(f1_score(y_true, y_pred, zero_division=0),                4),
        "brier_score": round(brier_score_loss(y_true, y_prob),                        4),
    }


def compare_models(
    y_true:     np.ndarray,
    prob_xgb:   np.ndarray,
    prob_nn:    np.ndarray,
    prob_stack: np.ndarray,
) -> pd.DataFrame:
    """
    Compare all three models and log a formatted table.

    Returns the results DataFrame sorted by ROC-AUC descending.
    """
    results = [
        evaluate_model(y_true, prob_xgb,   "XGBoost"),
        evaluate_model(y_true, prob_nn,     "Neural Network"),
        evaluate_model(y_true, prob_stack,  "Stacked Ensemble"),
    ]

    df   = pd.DataFrame(results).set_index("model").sort_values("roc_auc", ascending=False)
    best = df.index[0]

    separator = "=" * 65
    logger.info(separator)
    logger.info(" MODEL EVALUATION REPORT")
    logger.info(separator)
    logger.info(
        "\n%s",
        df.to_string(float_format=lambda x: f"{x:.4f}"),
    )
    logger.info(separator)
    logger.info(
        "Best model: %s  (ROC-AUC=%.4f)", best, df.loc[best, "roc_auc"]
    )

    return df


def print_classification_report(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    model_name: str,
    threshold:  float = 0.5,
) -> None:
    """Log sklearn's full classification report for *model_name*."""
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, digits=4)
    logger.info("Classification Report — %s\n%s", model_name, report)
