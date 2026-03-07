"""
Generate Rule-Based Retention Recommendations (No API Required)
================================================================
This script creates retention recommendations using business rules
instead of AI, suitable when Gemini API is not available.

Usage:
    python generate_recommendations_manual.py
"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add src/ to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from indian_currency import format_indian_currency


def generate_rule_based_recommendation(row: pd.Series) -> str:
    """Generate retention recommendation using business rules."""
    
    # Extract key features
    tenure = row.get('tenure', 0)
    contract = row.get('Contract', 'Unknown')
    charges = row.get('MonthlyCharges', 0)
    internet = row.get('InternetService', 'No')
    payment = row.get('PaymentMethod', 'Unknown')
    tech_support = row.get('TechSupport', 'No')
    online_security = row.get('OnlineSecurity', 'No')
    churn_prob = row.get('churn_probability', 0)
    risk_band = row.get('churn_band', 'Unknown')
    
    # Determine likely churn reason
    if tenure <= 3:
        churn_reason = "New customer - poor onboarding experience or unmet expectations"
    elif contract == "Month-to-month":
        churn_reason = "No long-term commitment - easily switches to competitors"
    elif charges > 80:
        churn_reason = "High monthly charges - price sensitivity"
    elif payment == "Electronic check":
        churn_reason = "Payment friction - inconvenient payment method"
    elif internet == "Fiber optic" and tech_support == "No":
        churn_reason = "Premium service without adequate support"
    else:
        churn_reason = "Multiple risk factors - dissatisfaction with service value"
    
    # Risk summary
    risk_factors = []
    if tenure <= 6:
        risk_factors.append(f"very short tenure ({tenure} months)")
    if contract == "Month-to-month":
        risk_factors.append("month-to-month contract")
    if charges > 70:
        risk_factors.append(f"high charges (₹{charges:.2f})")
    if payment == "Electronic check":
        risk_factors.append("electronic check payment")
    if tech_support == "No" and online_security == "No":
        risk_factors.append("no security/support add-ons")
    
    risk_summary = f"{risk_band} risk ({churn_prob:.1%}) due to: " + ", ".join(risk_factors)
    
    # Retention action
    if tenure <= 3:
        retention_action = "Immediate outreach within 24 hours. Assign dedicated account manager for first 90 days. Conduct satisfaction survey to identify pain points."
    elif contract == "Month-to-month" and charges > 70:
        retention_action = "Offer annual contract with 20-25% discount. Provide contract migration incentive (₹500 bill credit). Highlight long-term savings."
    elif contract == "Month-to-month":
        retention_action = "Propose annual or 2-year contract with 15-20% discount. Offer free device upgrade or premium channel for 3 months."
    elif charges > 80:
        retention_action = "Review bill with customer. Offer customized plan optimization. Provide loyalty discount (10-15%) for 6 months."
    else:
        retention_action = "Schedule retention call. Understand concerns. Offer service upgrades or competitive rate matching."
    
    # Offer recommendation
    if tenure <= 3:
        offer = f"• Welcome-back offer: ₹{int(charges * 2)} bill credit split over 6 months\n" \
                f"• Free premium add-ons for 3 months (worth ₹{int(charges * 0.3)})\n" \
                f"• Priority customer support and dedicated helpline"
    elif contract == "Month-to-month":
        annual_discount = int(charges * 12 * 0.20)
        offer = f"• Annual contract with 20% discount (save ₹{annual_discount}/year)\n" \
                f"• Contract signing bonus: ₹500 bill credit\n" \
                f"• Free router upgrade (worth ₹2,500)"
    elif charges > 80:
        monthly_discount = int(charges * 0.15)
        offer = f"• Loyalty discount: ₹{monthly_discount}/month for 6 months\n" \
                f"• Waive installation fees for service changes\n" \
                f"• Complimentary premium channel package (3 months)"
    elif payment == "Electronic check":
        offer = f"• Switch to auto-pay: get ₹5/month discount\n" \
                f"• One-time ₹200 incentive for payment method change\n" \
                f"• Paperless billing bonus: ₹100 credit"
    else:
        offer = f"• Personalized retention offer: {int(charges * 0.1)}-15% discount\n" \
                f"• Service upgrade options at discounted rates\n" \
                f"• Extended warranty or premium support inclusion"
    
    # Communication tone
    if churn_prob > 0.85:
        tone = "Urgent and empathetic. Express genuine concern about losing them. " \
               "Emphasize their value to the company. Make decision-maker available immediately."
    elif risk_band == "Critical":
        tone = "Proactive and solution-focused. Acknowledge service gaps. " \
               "Present concrete retention offers. Schedule immediate callback."
    elif risk_band == "High":
        tone = "Friendly and consultative. Position as partnership review. " \
               "Focus on optimizing their experience and value."
    else:
        tone = "Professional and appreciative. Thank for loyalty. " \
               "Present as customer appreciation initiative."
    
    # Build JSON response
    recommendation = {
        "likely_churn_reason": churn_reason,
        "risk_summary": risk_summary,
        "retention_action": retention_action,
        "offer_recommendation": offer,
        "communication_tone": tone,
        "source": "Rule-based engine"
    }
    
    return json.dumps(recommendation, ensure_ascii=False)


def main():
    predictions_path = PROJECT_ROOT / "outputs" / "churn_predictions.csv"
    
    if not predictions_path.exists():
        print(f"❌ ERROR: {predictions_path} not found.")
        return
    
    print("=" * 80)
    print("RULE-BASED RETENTION RECOMMENDATIONS GENERATOR")
    print("=" * 80)
    print(f"\n📂 Loading predictions from: {predictions_path}")
    
    df = pd.read_csv(predictions_path)
    print(f"   ✓ Loaded {len(df):,} customer predictions")
    
    # Show risk distribution
    risk_counts = df["churn_band"].value_counts()
    print(f"\n📊 Risk Distribution:")
    for band in ["Critical", "High", "Medium", "Low"]:
        count = risk_counts.get(band, 0)
        pct = 100 * count / len(df)
        print(f"   {band:10s}: {count:5,d} ({pct:5.1f}%)")
    
    # Generate recommendations for High and Critical customers
    high_risk_mask = df["churn_band"].isin(["High", "Critical"])
    n_high_risk = high_risk_mask.sum()
    
    print(f"\n📋 Generating rule-based recommendations for {n_high_risk:,} high-risk customers...")
    
    # Initialize recommendation column
    df["retention_recommendation"] = "Standard retention — low/medium risk customer."
    
    # Generate for high-risk customers
    for idx in df[high_risk_mask].index:
        row = df.loc[idx]
        df.at[idx, "retention_recommendation"] = generate_rule_based_recommendation(row)
    
    # Save updated predictions
    df.to_csv(predictions_path, index=False)
    print(f"\n✅ Updated predictions saved to: {predictions_path}")
    
    # Generate summary report
    generate_summary_report(df)
    
    print("\n" + "=" * 80)
    print("✨ Rule-based retention recommendations generation complete!")
    print("=" * 80)
    print("\n💡 Note: These are rule-based recommendations.")
    print("   For AI-powered recommendations, configure a valid GEMINI_API_KEY in .env")


def generate_summary_report(df: pd.DataFrame):
    """Create a summary report of retention recommendations."""
    report_path = PROJECT_ROOT / "outputs" / "retention_recommendations_report.txt"
    
    high_risk_df = df[df["churn_band"].isin(["High", "Critical"])].copy()
    high_risk_df = high_risk_df.sort_values("expected_revenue_loss", ascending=False)
    
    lines = []
    lines.append("=" * 80)
    lines.append("  RETENTION RECOMMENDATIONS - SUMMARY REPORT (RULE-BASED)")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total high-risk customers: {len(high_risk_df):,}")
    lines.append(f"Total revenue at risk: {format_indian_currency(high_risk_df['expected_revenue_loss'].sum())}")
    
    # Show top 15 customers with recommendations
    lines.append("\n" + "=" * 80)
    lines.append("TOP 15 CRITICAL CUSTOMERS WITH RETENTION STRATEGIES")
    lines.append("=" * 80)
    
    for i, (idx, row) in enumerate(high_risk_df.head(15).iterrows(), 1):
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Customer #{i}")
        lines.append(f"{'─' * 80}")
        lines.append(f"Churn Probability    : {row['churn_probability']:.1%}")
        lines.append(f"Risk Band            : {row['churn_band']}")
        lines.append(f"Expected Revenue Loss: {format_indian_currency(row['expected_revenue_loss'])}")
        lines.append(f"Tenure               : {int(row['tenure'])} months")
        lines.append(f"Monthly Charges      : {format_indian_currency(row['MonthlyCharges'])}")
        lines.append(f"Contract             : {row['Contract']}")
        lines.append(f"Internet Service     : {row['InternetService']}")
        lines.append(f"Payment Method       : {row['PaymentMethod']}")
        lines.append(f"Tech Support         : {row['TechSupport']}")
        lines.append(f"Online Security      : {row['OnlineSecurity']}")
        
        # Parse and display recommendation
        rec = row['retention_recommendation']
        if rec and rec != "Skipped" and not rec.startswith("Standard"):
            try:
                rec_data = json.loads(rec)
                lines.append(f"\n📋 RETENTION STRATEGY:")
                lines.append(f"\n   🔍 Likely Churn Reason:")
                lines.append(f"      {rec_data.get('likely_churn_reason', 'N/A')}")
                lines.append(f"\n   ⚠️  Risk Summary:")
                lines.append(f"      {rec_data.get('risk_summary', 'N/A')}")
                lines.append(f"\n   🎯 Retention Action:")
                lines.append(f"      {rec_data.get('retention_action', 'N/A')}")
                lines.append(f"\n   💰 Offer Recommendation:")
                for line in rec_data.get('offer_recommendation', 'N/A').split('\n'):
                    lines.append(f"      {line}")
                lines.append(f"\n   💬 Communication Tone:")
                lines.append(f"      {rec_data.get('communication_tone', 'N/A')}")
            except:
                lines.append(f"\n📋 RECOMMENDATION: {rec[:300]}...")
    
    lines.append("\n" + "=" * 80)
    
    # Save report
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"   📄 Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
