"""
business_impact.py
==================
Compute expected revenue loss per customer and rank by financial risk.

Formula:
  expected_revenue_loss = P(churn) × MonthlyCharges × max(0, CONTRACT_MONTHS − tenure)

Interpretation: the expected rupee value the business will lose if this
customer churns, assuming they would have remained for the rest of their
typical contract period.
"""
from __future__ import annotations

import logging

import pandas as pd

from config import (
    ESTIMATED_CONTRACT_MONTHS,
    FALLBACK_MONTHLY_CHARGES,
    FALLBACK_TENURE,
    MONTHLY_CHARGES_COL,
    TENURE_COL,
)
from indian_currency import format_indian_currency

logger = logging.getLogger(__name__)


def _resolve_column(df: pd.DataFrame, col: str, fallback: float) -> pd.Series:
    """Return a numeric Series for *col*, filling missing values with *fallback*."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(fallback)
    logger.warning("Column '%s' not found — using fallback value %.1f.", col, fallback)
    return pd.Series(fallback, index=df.index)


def compute_business_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``expected_revenue_loss`` column and sort by it descending.

    The sort makes the highest-value customers appear first in exports
    and reports, so business teams can immediately prioritise outreach.
    """
    df = df.copy()

    monthly   = _resolve_column(df, MONTHLY_CHARGES_COL, FALLBACK_MONTHLY_CHARGES)
    tenure    = _resolve_column(df, TENURE_COL,           FALLBACK_TENURE)
    remaining = (ESTIMATED_CONTRACT_MONTHS - tenure).clip(lower=0)

    df["expected_revenue_loss"] = (df["churn_probability"] * monthly * remaining).round(2)
    df = df.sort_values("expected_revenue_loss", ascending=False)

    logger.info(
        "Total expected revenue at risk: %s", format_indian_currency(df["expected_revenue_loss"].sum())
    )
    return df
