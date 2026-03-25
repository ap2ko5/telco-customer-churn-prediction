"""
risk_segmentation.py
====================
Assign churn risk bands to customers based on predicted probability.

Bands (configured in config.RISK_BANDS):
  Low      : [0.00, 0.30)
  Medium   : [0.30, 0.60)
  High     : [0.60, 0.80)
  Critical : [0.80, 1.00]
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import RISK_BANDS

logger = logging.getLogger(__name__)

# Pre-compute sorted bands once at import time for fast vectorised assignment
_BANDS_SORTED = list(RISK_BANDS.items())   # [("Low", (0.0, 0.3)), ...]


def assign_risk_band(churn_probs: np.ndarray) -> np.ndarray:
    """
    Vectorised mapping from probability array → risk band name array.

    Parameters
    ----------
    churn_probs : np.ndarray, shape (n,)

    Returns
    -------
    bands : np.ndarray of dtype object, shape (n,)
            Values are one of: "Low", "Medium", "High", "Critical"
    """
    probs = np.asarray(churn_probs, dtype=np.float64)

    conditions = []
    choices = []
    for band_name, (lo, hi) in _BANDS_SORTED:
        conditions.append((probs >= lo) & (probs < hi))
        choices.append(band_name)

    bands = np.select(conditions, choices, default="Critical").astype(object)

    # Defensive logging for out-of-range values (outside [0, 1]).
    invalid_mask = (probs < 0.0) | (probs > 1.0)
    if invalid_mask.any():
        logger.warning(
            "assign_risk_band: %d out-of-range probabilities detected; assigned 'Critical'.",
            int(invalid_mask.sum()),
        )

    return bands


def add_risk_band(df: pd.DataFrame, prob_col: str = "churn_probability") -> pd.DataFrame:
    """
    Add a ``churn_band`` column to df based on *prob_col*.

    Returns a copy — the original DataFrame is never mutated.
    """
    df = df.copy()
    df["churn_band"] = assign_risk_band(df[prob_col].to_numpy())

    counts = df["churn_band"].value_counts()
    total  = len(df)
    logger.info("Risk band distribution:")
    for band in ["Critical", "High", "Medium", "Low"]:
        n   = counts.get(band, 0)
        pct = 100 * n / max(total, 1)
        logger.info("  %-10s: %5d  (%5.1f%%)", band, n, pct)

    return df
