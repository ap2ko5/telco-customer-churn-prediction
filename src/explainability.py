"""
explainability.py
=================
Per-customer explainability utilities for churn predictions.

This module computes top churn drivers using SHAP values from the trained
XGBoost model and attaches them to the output DataFrame so downstream modules
(e.g., Gemini retention recommendations) can consume customer-level drivers.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def _normalize_shap_values(shap_values) -> np.ndarray:
    """Normalize SHAP output format across versions to shape (n_samples, n_features)."""
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            return np.empty((0, 0), dtype=float)
        return np.asarray(shap_values[-1])
    return np.asarray(shap_values)


def _top_drivers_for_row(
    row_shap: np.ndarray,
    feature_names: list[str],
    top_n: int,
) -> list[dict]:
    """Extract top positive churn drivers for one customer."""
    row_shap = np.asarray(row_shap, dtype=float)

    positive_idx = np.where(row_shap > 0)[0]
    if len(positive_idx) > 0:
        ranked = positive_idx[np.argsort(row_shap[positive_idx])[::-1]]
    else:
        ranked = np.argsort(np.abs(row_shap))[::-1]

    ranked = ranked[:top_n]
    drivers = []
    for idx in ranked:
        drivers.append({
            "feature": feature_names[int(idx)],
            "impact": round(float(row_shap[int(idx)]), 6),
        })
    return drivers


def add_top_churn_drivers(
    df: pd.DataFrame,
    xgb_model: XGBClassifier,
    X_transformed: np.ndarray,
    feature_names: list[str],
    *,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Add a `top_churn_drivers` JSON column for High/Critical risk customers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains `churn_band`.
    xgb_model : XGBClassifier
        Fitted XGBoost model.
    X_transformed : np.ndarray
        Fully transformed feature matrix aligned row-by-row with `df`.
    feature_names : list[str]
        Feature names corresponding to transformed matrix columns.
    top_n : int
        Number of top drivers to store per customer.

    Returns
    -------
    pd.DataFrame
        Copy of input with `top_churn_drivers` column added.
    """
    df = df.copy()
    df["top_churn_drivers"] = "[]"

    if "churn_band" not in df.columns:
        logger.warning("add_top_churn_drivers: 'churn_band' not found. Skipping explainability.")
        return df

    mask = df["churn_band"].isin(["High", "Critical"]).to_numpy()
    idx = np.where(mask)[0]
    if len(idx) == 0:
        logger.info("add_top_churn_drivers: no High/Critical customers found.")
        return df

    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping top_churn_drivers generation.")
        return df

    logger.info("Computing per-customer SHAP churn drivers for %d customers...", len(idx))
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = _normalize_shap_values(explainer.shap_values(X_transformed[idx]))

    if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
        logger.warning(
            "Unexpected SHAP output shape %s (expected n x %d). Skipping driver extraction.",
            getattr(shap_values, "shape", None),
            len(feature_names),
        )
        return df

    for local_pos, global_idx in enumerate(idx):
        drivers = _top_drivers_for_row(shap_values[local_pos], feature_names, top_n=top_n)
        df.at[global_idx, "top_churn_drivers"] = json.dumps(drivers, ensure_ascii=False)

    return df
