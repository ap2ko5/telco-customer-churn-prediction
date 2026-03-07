"""
reporting.py
============
Generate all final outputs for the churn intelligence pipeline.

Outputs:
  1. churn_predictions.csv  — full customer-level prediction table
  2. summary_report.txt     — band distribution + revenue at risk + top-50
  3. prob_distribution.png  — histogram of churn probabilities
  4. band_distribution.png  — bar chart of risk band counts
  5. shap_importance.png    — SHAP bar chart for XGBoost (optional)
"""
from __future__ import annotations

import logging
import textwrap

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must precede plt import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import (
    BAND_PLOT,
    OUTPUTS_DIR,
    PREDICTIONS_CSV,
    PROB_PLOT,
    SHAP_PLOT,
    SUMMARY_TXT,
)
from indian_currency import format_indian_currency

logger = logging.getLogger(__name__)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Design tokens ──────────────────────────────────────────────────────────────
_BG_DARK   = "#0f1117"
_BG_PANEL  = "#1a1d27"
_SPINE_CLR = "#444"

BAND_COLORS = {
    "Low":      "#2ecc71",
    "Medium":   "#f1c40f",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}
BAND_ORDER = ["Low", "Medium", "High", "Critical"]


def _apply_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply the project's dark theme to a figure/axes pair."""
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_CLR)


# ── Step 1: Save predictions ───────────────────────────────────────────────────

def save_predictions(df: pd.DataFrame) -> None:
    """Save the full prediction DataFrame to CSV with priority columns first."""
    priority = [
        "churn_probability", "churn_band",
        "expected_revenue_loss", "retention_recommendation",
    ]
    ordered  = priority + [c for c in df.columns if c not in priority]
    out_cols = [c for c in ordered if c in df.columns]
    df[out_cols].to_csv(PREDICTIONS_CSV, index=False)
    logger.info("Predictions saved → %s", PREDICTIONS_CSV)


# ── Step 2: Summary report ─────────────────────────────────────────────────────

def write_summary_report(df: pd.DataFrame) -> None:
    """Write a human-readable text summary report."""
    total           = len(df)
    band_counts     = df["churn_band"].value_counts()
    total_rev_risk  = df["expected_revenue_loss"].sum() if "expected_revenue_loss" in df.columns else 0.0

    lines = [
        "=" * 65,
        "  CHURN INTELLIGENCE – SUMMARY REPORT",
        "=" * 65,
        f"\nTotal customers analysed : {total:,}",
        f"Total revenue at risk    : {format_indian_currency(total_rev_risk)}\n",
        "-- Risk Band Distribution --",
    ]
    for band in BAND_ORDER:
        n   = band_counts.get(band, 0)
        pct = 100 * n / max(total, 1)
        lines.append(f"  {band:10s}: {n:5,d}  ({pct:5.1f}%)")

    lines.append("\n-- Top 50 High-Risk Customers by Expected Revenue Loss --")
    top50_cols = [
        c for c in ["churn_probability", "churn_band", "expected_revenue_loss",
                    "MonthlyCharges", "tenure"]
        if c in df.columns
    ]
    top50 = (
        df.nlargest(50, "expected_revenue_loss") if "expected_revenue_loss" in df.columns
        else df.head(50)
    )
    lines.append(top50[top50_cols].reset_index(drop=True).to_string(float_format=lambda x: f"{x:.2f}"))
    lines.append("\n" + "=" * 65)

    report = "\n".join(lines)
    SUMMARY_TXT.write_text(report, encoding="utf-8")
    logger.info("Summary report saved → %s", SUMMARY_TXT)


# ── Step 3: Probability distribution plot ─────────────────────────────────────

def plot_probability_distribution(df: pd.DataFrame) -> None:
    """Histogram of churn probabilities with risk-band threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(fig, ax)

    ax.hist(df["churn_probability"].to_numpy(), bins=50,
            color="#5b6af0", edgecolor=_BG_DARK, alpha=0.85)

    # Draw vertical threshold lines
    for threshold, color, label in [
        (0.3, BAND_COLORS["Medium"],   "Med"),
        (0.6, BAND_COLORS["High"],     "High"),
        (0.8, BAND_COLORS["Critical"], "Crit"),
    ]:
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.95,
                label, color=color, fontsize=8, va="top")

    ax.set_xlabel("Churn Probability", color="white")
    ax.set_ylabel("Number of Customers", color="white")
    ax.set_title("Churn Probability Distribution", color="white", fontsize=14, pad=12)

    plt.tight_layout()
    plt.savefig(PROB_PLOT, dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)
    logger.info("Probability distribution plot saved → %s", PROB_PLOT)


# ── Step 4: Band distribution bar chart ───────────────────────────────────────

def plot_band_distribution(df: pd.DataFrame) -> None:
    """Bar chart of customer counts per risk band."""
    total       = len(df)
    band_counts = {b: int((df["churn_band"] == b).sum()) for b in BAND_ORDER}

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_theme(fig, ax)

    bars = ax.bar(
        BAND_ORDER,
        [band_counts[b] for b in BAND_ORDER],
        color=[BAND_COLORS[b] for b in BAND_ORDER],
        edgecolor=_BG_DARK,
        width=0.6,
    )
    for bar, band in zip(bars, BAND_ORDER):
        n   = band_counts[band]
        pct = 100 * n / max(total, 1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{n:,}\n({pct:.1f}%)",
            ha="center", va="bottom", color="white", fontsize=9,
        )

    ax.set_xlabel("Risk Band", color="white")
    ax.set_ylabel("Number of Customers", color="white")
    ax.set_title("Churn Risk Band Distribution", color="white", fontsize=14, pad=12)

    plt.tight_layout()
    plt.savefig(BAND_PLOT, dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)
    logger.info("Band distribution plot saved → %s", BAND_PLOT)


# ── Step 5: SHAP feature importance ───────────────────────────────────────────

def plot_shap_importance(
    xgb_model:    XGBClassifier,
    X_sample:     np.ndarray,
    feature_names: list[str],
    max_display:  int = 20,
) -> None:
    """Mean |SHAP| bar chart for XGBoost — top *max_display* features."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plot. Run: pip install shap")
        return

    logger.info("Computing SHAP values (may take a moment)...")
    explainer = shap.TreeExplainer(xgb_model)

    # Cap at 500 rows for speed
    sample_size = min(500, len(X_sample))
    idx         = np.random.default_rng(42).choice(len(X_sample), sample_size, replace=False)
    shap_values = explainer.shap_values(X_sample[idx])

    mean_abs    = np.abs(shap_values).mean(axis=0)
    sorted_idx  = np.argsort(mean_abs)[::-1][:max_display]
    feat_labels = [feature_names[i] for i in sorted_idx]
    feat_vals   = mean_abs[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark_theme(fig, ax)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(sorted_idx)))
    ax.barh(range(len(sorted_idx)), feat_vals[::-1], color=colors[::-1], edgecolor=_BG_DARK)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(feat_labels[::-1], color="white")
    ax.set_xlabel("Mean |SHAP Value|", color="white")
    ax.set_title("XGBoost — Top Feature Importance (SHAP)", color="white", fontsize=13, pad=10)

    plt.tight_layout()
    plt.savefig(SHAP_PLOT, dpi=150, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)
    logger.info("SHAP importance plot saved → %s", SHAP_PLOT)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def generate_all_reports(
    df:            pd.DataFrame,
    xgb_model:     XGBClassifier | None,
    X_transformed: np.ndarray | None,
    feature_names: list[str] | None,
) -> None:
    """Run all five reporting steps in sequence."""
    save_predictions(df)
    write_summary_report(df)
    plot_probability_distribution(df)
    plot_band_distribution(df)

    if xgb_model is not None and X_transformed is not None and feature_names:
        plot_shap_importance(xgb_model, X_transformed, feature_names)
    else:
        logger.info("Skipping SHAP plot — model or feature names not provided.")
