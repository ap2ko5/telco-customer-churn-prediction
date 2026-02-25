# -*- coding: utf-8 -*-
"""
reporting.py — Generate all final outputs for the churn intelligence pipeline.

Outputs:
  1. churn_predictions.csv     — full customer predictions
  2. summary_report.txt        — band distribution, revenue at risk, top-50
  3. prob_distribution.png     — histogram of churn probabilities
  4. band_distribution.png     — bar chart of risk band counts
  5. shap_importance.png       — SHAP feature importance for XGBoost
"""
from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier

from config import (
    BAND_PLOT,
    OUTPUTS_DIR,
    PREDICTIONS_CSV,
    PROB_PLOT,
    SHAP_PLOT,
    SUMMARY_TXT,
)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette consistent with risk bands ────────────────────────────────────────
BAND_COLORS = {
    "Low":      "#2ecc71",
    "Medium":   "#f1c40f",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}
BAND_ORDER = ["Low", "Medium", "High", "Critical"]


def save_predictions(df: pd.DataFrame) -> None:
    """Save the full prediction DataFrame to CSV."""
    cols_first = [
        "churn_probability", "churn_band",
        "expected_revenue_loss", "retention_recommendation",
    ]
    other_cols = [c for c in df.columns if c not in cols_first]
    ordered = cols_first + other_cols
    out_cols = [c for c in ordered if c in df.columns]

    df[out_cols].to_csv(PREDICTIONS_CSV, index=False)
    print(f"[reporting] Predictions saved → {PREDICTIONS_CSV}")


def write_summary_report(df: pd.DataFrame) -> None:
    """Write a human-readable summary report."""
    total = len(df)
    band_counts = df["churn_band"].value_counts()
    total_rev_risk = df["expected_revenue_loss"].sum() if "expected_revenue_loss" in df.columns else 0.0

    lines = []
    lines.append("=" * 65)
    lines.append("  CHURN INTELLIGENCE – SUMMARY REPORT")
    lines.append("=" * 65)
    lines.append(f"\nTotal customers analysed : {total:,}")
    lines.append(f"Total revenue at risk    : ${total_rev_risk:,.2f}\n")

    lines.append("-- Risk Band Distribution --")
    for band in BAND_ORDER:
        n   = band_counts.get(band, 0)
        pct = 100 * n / max(total, 1)
        lines.append(f"  {band:10s}: {n:5,d}  ({pct:5.1f}%)")

    lines.append("\n-- Top 50 High-Risk Customers by Expected Revenue Loss --")
    top50 = df.nlargest(50, "expected_revenue_loss") if "expected_revenue_loss" in df.columns else df.head(50)
    top50_cols = [
        c for c in [
            "churn_probability", "churn_band",
            "expected_revenue_loss", "MonthlyCharges", "tenure",
        ] if c in top50.columns
    ]
    lines.append(top50[top50_cols].reset_index(drop=True).to_string(float_format=lambda x: f"{x:.2f}"))
    lines.append("\n" + "=" * 65)

    report = "\n".join(lines)
    SUMMARY_TXT.write_text(report, encoding="utf-8")
    print(f"[reporting] Summary report saved → {SUMMARY_TXT}")
    print(report)


def plot_probability_distribution(df: pd.DataFrame) -> None:
    """Histogram of churn probabilities coloured by risk band."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    probs = df["churn_probability"].values
    ax.hist(probs, bins=50, color="#5b6af0", edgecolor="#0f1117", alpha=0.85)

    # Vertical band boundaries
    for threshold, color, label in [
        (0.3, "#f1c40f", "Med"),
        (0.6, "#e67e22", "High"),
        (0.8, "#e74c3c", "Critical"),
    ]:
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.95, label,
                color=color, fontsize=8, va="top")

    ax.set_xlabel("Churn Probability", color="white")
    ax.set_ylabel("Number of Customers", color="white")
    ax.set_title("Churn Probability Distribution", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(PROB_PLOT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[reporting] Probability distribution plot saved → {PROB_PLOT}")


def plot_band_distribution(df: pd.DataFrame) -> None:
    """Bar chart of customer counts per risk band."""
    band_counts = {
        band: (df["churn_band"] == band).sum() for band in BAND_ORDER
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    bars = ax.bar(
        BAND_ORDER,
        [band_counts[b] for b in BAND_ORDER],
        color=[BAND_COLORS[b] for b in BAND_ORDER],
        edgecolor="#0f1117",
        width=0.6,
    )

    total = len(df)
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
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(BAND_PLOT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[reporting] Band distribution plot saved → {BAND_PLOT}")


def plot_shap_importance(
    xgb_model: XGBClassifier,
    X_sample: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> None:
    """SHAP beeswarm/bar plot for XGBoost feature importance."""
    try:
        import shap
    except ImportError:
        print("[reporting] shap not installed. Skipping SHAP plot.")
        return

    print("[reporting] Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(xgb_model)

    # Use a sample to keep computation fast (max 500 rows)
    sample_size = min(500, len(X_sample))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_sample), sample_size, replace=False)
    X_s = X_sample[idx]

    shap_values = explainer.shap_values(X_s)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # Mean |SHAP| per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:max_display]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_vals     = mean_abs[sorted_idx]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(sorted_idx)))
    ax.barh(range(len(sorted_idx)), sorted_vals[::-1], color=colors[::-1], edgecolor="#0f1117")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(sorted_features[::-1], color="white")
    ax.set_xlabel("Mean |SHAP Value|", color="white")
    ax.set_title("XGBoost — Top Feature Importance (SHAP)", color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(SHAP_PLOT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[reporting] SHAP importance plot saved → {SHAP_PLOT}")


def generate_all_reports(
    df: pd.DataFrame,
    xgb_model: XGBClassifier | None,
    X_transformed: np.ndarray | None,
    feature_names: list[str] | None,
) -> None:
    """Run all reporting steps in sequence."""
    save_predictions(df)
    write_summary_report(df)
    plot_probability_distribution(df)
    plot_band_distribution(df)

    if xgb_model is not None and X_transformed is not None and feature_names:
        plot_shap_importance(xgb_model, X_transformed, feature_names)
    else:
        print("[reporting] Skipping SHAP plot (model or feature names not provided).")
