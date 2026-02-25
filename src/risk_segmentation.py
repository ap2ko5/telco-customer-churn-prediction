# -*- coding: utf-8 -*-
"""
risk_segmentation.py — Assign churn risk bands based on predicted probability.

Bands:
  Low      : [0.0, 0.3)
  Medium   : [0.3, 0.6)
  High     : [0.6, 0.8)
  Critical : [0.8, 1.0]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RISK_BANDS


def assign_risk_band(churn_probs: np.ndarray) -> np.ndarray:
    """
    Map each probability to a named risk band.

    Parameters
    ----------
    churn_probs : np.ndarray  shape (n,)

    Returns
    -------
    bands : np.ndarray of str, shape (n,)
    """
    bands = np.empty(len(churn_probs), dtype=object)
    for band_name, (lo, hi) in RISK_BANDS.items():
        mask = (churn_probs >= lo) & (churn_probs < hi)
        bands[mask] = band_name

    # Edge case: exactly 1.0
    bands[churn_probs >= 1.0] = "Critical"

    return bands


def add_risk_band(df: pd.DataFrame, prob_col: str = "churn_probability") -> pd.DataFrame:
    """Add a 'churn_band' column to df based on prob_col."""
    df = df.copy()
    df["churn_band"] = assign_risk_band(df[prob_col].values)

    # Print distribution
    counts = df["churn_band"].value_counts()
    total  = len(df)
    print("\n-- Churn Risk Band Distribution --")
    for band in ["Critical", "High", "Medium", "Low"]:
        n = counts.get(band, 0)
        pct = 100 * n / max(total, 1)
        print(f"  {band:10s}: {n:5,d}  ({pct:5.1f}%)")

    return df
