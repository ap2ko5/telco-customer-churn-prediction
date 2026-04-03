#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def _prediction_label_from_band(band: object) -> str:
    value = str(band).strip().lower()
    mapping = {
        "critical": "Very High Churn Risk",
        "high": "High Churn Risk",
        "medium": "Medium Churn Risk",
        "low": "Low Churn Risk",
    }
    return mapping.get(value, "Unknown")


def _short_action(row: pd.Series) -> str:
    band = str(row.get("churn_band", "")).strip().lower()
    if band == "critical":
        return "Call today, give strongest retention offer, fast-track support."
    if band == "high":
        return "Reach out within 24 hours with personalized plan or discount."
    if band == "medium":
        return "Send targeted value message and monitor usage signals."
    return "Continue engagement and routine loyalty communication."


def build_readable_output(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    if "customerID" in df.columns:
        out["Customer ID"] = df["customerID"]

    if "churn_probability" in df.columns:
        out["Churn Probability"] = df["churn_probability"].apply(_to_percent)

    if "churn_band" in df.columns:
        out["Risk Band"] = df["churn_band"]
        out["Prediction Summary"] = df["churn_band"].apply(_prediction_label_from_band)

    if "expected_revenue_loss_rupees" in df.columns:
        out["Expected Revenue Loss"] = df["expected_revenue_loss_rupees"]
    elif "expected_revenue_loss" in df.columns:
        out["Expected Revenue Loss"] = df["expected_revenue_loss"].map(lambda x: f"INR {x:,.2f}" if pd.notna(x) else "")

    if "top_churn_drivers" in df.columns:
        out["Top Churn Drivers"] = df["top_churn_drivers"]

    if "retention_recommendation" in df.columns:
        out["Detailed Recommendation"] = df["retention_recommendation"]

    out["Quick Action"] = df.apply(_short_action, axis=1)

    # Keep all rows and avoid index column in output.
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a human-friendly prediction output CSV.")
    parser.add_argument("--input", required=True, help="Path to raw predictions CSV")
    parser.add_argument("--output", required=True, help="Path to readable predictions CSV")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    readable = build_readable_output(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    readable.to_csv(out_path, index=False)

    print(f"[OK] Readable output written: {out_path}")
    print(f"[INFO] Rows: {len(readable):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())