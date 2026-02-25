"""
business_impact.py — Compute expected revenue loss and rank customers.

Formula:
  expected_revenue_loss = churn_probability × MonthlyCharges × estimated_remaining_months
  estimated_remaining_months = max(0, ESTIMATED_CONTRACT_MONTHS - tenure)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ESTIMATED_CONTRACT_MONTHS,
    FALLBACK_MONTHLY_CHARGES,
    FALLBACK_TENURE,
    MONTHLY_CHARGES_COL,
    TENURE_COL,
)


def compute_business_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add expected_revenue_loss column and sort by it descending.

    Handles missing MonthlyCharges / tenure with fallback values.
    """
    df = df.copy()

    # Resolve monthly charges
    if MONTHLY_CHARGES_COL in df.columns:
        monthly = pd.to_numeric(df[MONTHLY_CHARGES_COL], errors="coerce").fillna(
            FALLBACK_MONTHLY_CHARGES
        )
    else:
        print(
            f"[business_impact] '{MONTHLY_CHARGES_COL}' column not found. "
            f"Using fallback ${FALLBACK_MONTHLY_CHARGES:.0f}/month."
        )
        monthly = pd.Series(FALLBACK_MONTHLY_CHARGES, index=df.index)

    # Resolve tenure
    if TENURE_COL in df.columns:
        tenure = pd.to_numeric(df[TENURE_COL], errors="coerce").fillna(FALLBACK_TENURE)
    else:
        print(
            f"[business_impact] '{TENURE_COL}' column not found. "
            f"Using fallback tenure={FALLBACK_TENURE} months."
        )
        tenure = pd.Series(FALLBACK_TENURE, index=df.index)

    remaining = (ESTIMATED_CONTRACT_MONTHS - tenure).clip(lower=0)

    df["expected_revenue_loss"] = (
        df["churn_probability"] * monthly * remaining
    ).round(2)

    df = df.sort_values("expected_revenue_loss", ascending=False)

    total_at_risk = df["expected_revenue_loss"].sum()
    print(
        f"[business_impact] Total expected revenue at risk: "
        f"${total_at_risk:,.2f}"
    )

    return df
