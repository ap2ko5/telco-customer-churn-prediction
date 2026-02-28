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
    bands = np.empty(len(probs), dtype=object)

    for band_name, (lo, hi) in _BANDS_SORTED:
        bands[(probs >= lo) & (probs < hi)] = band_name

    # Edge case: exactly 1.0 falls outside all half-open intervals
    bands[probs >= 1.0] = "Critical"

    # Safety: fill any remaining None (shouldn't happen with valid inputs)
    none_mask = bands == None  # noqa: E711
    if none_mask.any():
        logger.warning(
            "assign_risk_band: %d values did not match any band. Defaulting to 'Critical'.",
            none_mask.sum(),
        )
        bands[none_mask] = "Critical"

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
