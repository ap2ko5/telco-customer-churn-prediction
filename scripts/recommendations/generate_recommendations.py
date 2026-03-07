"""
Generate AI Retention Recommendations for Existing Predictions
================================================================
This script loads the churn predictions CSV and generates personalized
retention recommendations using Gemini 2.5 Flash for all High and Critical
risk customers.

Usage:
    python scripts/recommendations/generate_recommendations.py

Outputs:
    - Updates outputs/churn_predictions.csv with AI recommendations
    - Creates outputs/retention_recommendations_report.txt with summary
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add src/ to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from retention_ai import generate_retention_recommendations
from indian_currency import format_indian_currency

def main():
    # Load predictions
    predictions_path = PROJECT_ROOT / "outputs" / "churn_predictions.csv"
    
    if not predictions_path.exists():
        print(f"❌ ERROR: {predictions_path} not found.")
        print("   Please run training pipeline first: python -m src.train_pipeline --data data/customer_churn.csv")
        return
    
    print("=" * 80)
    print("AI RETENTION RECOMMENDATIONS GENERATOR")
    print("=" * 80)
    print(f"\n📂 Loading predictions from: {predictions_path}")
    
    df = pd.read_csv(predictions_path)
    print(f"   ✓ Loaded {len(df):,} customer predictions")
    
    # Show current risk distribution
    risk_counts = df["churn_band"].value_counts()
    print(f"\n📊 Risk Distribution:")
    for band in ["Critical", "High", "Medium", "Low"]:
        count = risk_counts.get(band, 0)
        pct = 100 * count / len(df)
        print(f"   {band:10s}: {count:5,d} ({pct:5.1f}%)")
    
    # Count customers needing AI recommendations
    high_risk_mask = df["churn_band"].isin(["High", "Critical"])
    n_high_risk = high_risk_mask.sum()
    
    print(f"\n🤖 Generating AI recommendations for {n_high_risk:,} high-risk customers...")
    print(f"   Using: Gemini 2.5 Flash")
    print(f"   This may take 2-5 minutes depending on API response time...\n")
    
    # Generate recommendations
    df_with_recs = generate_retention_recommendations(df)
    
    # Save updated predictions
    df_with_recs.to_csv(predictions_path, index=False)
    print(f"\n✅ Updated predictions saved to: {predictions_path}")
    
    # Generate summary report
    generate_summary_report(df_with_recs)
    
    print("\n" + "=" * 80)
    print("✨ Retention recommendations generation complete!")
    print("=" * 80)


def generate_summary_report(df: pd.DataFrame):
    """Create a summary report of retention recommendations."""
    report_path = PROJECT_ROOT / "outputs" / "retention_recommendations_report.txt"
    
    high_risk_df = df[df["churn_band"].isin(["High", "Critical"])].copy()
    high_risk_df = high_risk_df.sort_values("expected_revenue_loss", ascending=False)
    
    lines = []
    lines.append("=" * 80)
    lines.append("  AI RETENTION RECOMMENDATIONS - SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total high-risk customers: {len(high_risk_df):,}")
    lines.append(f"Total revenue at risk: {format_indian_currency(high_risk_df['expected_revenue_loss'].sum())}")
    
    # Show top 10 customers with recommendations
    lines.append("\n" + "=" * 80)
    lines.append("TOP 10 CRITICAL CUSTOMERS WITH RETENTION STRATEGIES")
    lines.append("=" * 80)
    
    for i, (idx, row) in enumerate(high_risk_df.head(10).iterrows(), 1):
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Customer #{i}")
        lines.append(f"{'─' * 80}")
        lines.append(f"Churn Probability    : {row['churn_probability']:.1%}")
        lines.append(f"Risk Band            : {row['churn_band']}")
        lines.append(f"Expected Revenue Loss: {format_indian_currency(row['expected_revenue_loss'])}")
        lines.append(f"Tenure               : {row['tenure']} months")
        lines.append(f"Monthly Charges      : {format_indian_currency(row['MonthlyCharges'])}")
        lines.append(f"Contract             : {row['Contract']}")
        lines.append(f"Internet Service     : {row['InternetService']}")
        lines.append(f"Payment Method       : {row['PaymentMethod']}")
        
        # Parse and display recommendation
        rec = row['retention_recommendation']
        if rec and rec != "Skipped" and not rec.startswith("N/A"):
            try:
                import json
                rec_data = json.loads(rec)
                lines.append(f"\n🤖 AI RETENTION STRATEGY:")
                lines.append(f"   Likely Churn Reason : {rec_data.get('likely_churn_reason', 'N/A')}")
                lines.append(f"   Risk Summary        : {rec_data.get('risk_summary', 'N/A')}")
                lines.append(f"   Retention Action    : {rec_data.get('retention_action', 'N/A')}")
                lines.append(f"   Offer Recommendation: {rec_data.get('offer_recommendation', 'N/A')}")
                lines.append(f"   Communication Tone  : {rec_data.get('communication_tone', 'N/A')}")
            except:
                lines.append(f"\n🤖 AI RECOMMENDATION: {rec[:200]}...")
        else:
            lines.append(f"\n🤖 AI RECOMMENDATION: {rec}")
    
    lines.append("\n" + "=" * 80)
    
    # Save report
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"   📄 Summary report saved to: {report_path}")


if __name__ == "__main__":
    main()
