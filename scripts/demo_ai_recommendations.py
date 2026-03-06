"""
Demo: AI-based retention recommendations using Gemini 2.5 Flash
=================================================================
This script demonstrates how the AI recommendation system generates
personalized retention strategies for high-risk customers.

Usage:
    python scripts/demo_ai_recommendations.py

Requirements:
    - GEMINI_API_KEY set in .env file
    - google-generativeai package installed
"""
import sys
import json
from pathlib import Path

# Add src/ to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from retention_ai import generate_retention_recommendations


def main():
    # Create a small test DataFrame with one Critical-risk customer
    test_data = pd.DataFrame([{
        "churn_probability": 0.85,
        "churn_band": "Critical",
        "tenure": 2,
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "OnlineSecurity": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "MultipleLines": "No",
        "SeniorCitizen": 0
    }])

    print("=" * 70)
    print("AI RETENTION RECOMMENDATIONS DEMO")
    print("=" * 70)
    print("\nTest Customer Profile:")
    print(f"  Churn Probability: {test_data['churn_probability'].iloc[0]:.1%}")
    print(f"  Risk Band: {test_data['churn_band'].iloc[0]}")
    print(f"  Tenure: {test_data['tenure'].iloc[0]} months")
    print(f"  Monthly Charges: ${test_data['MonthlyCharges'].iloc[0]:.2f}")
    print(f"  Contract: {test_data['Contract'].iloc[0]}")
    print(f"  Internet: {test_data['InternetService'].iloc[0]}")
    print("\nGenerating AI recommendation from Gemini 2.5 Flash...\n")

    # Generate AI recommendation
    try:
        result_df = generate_retention_recommendations(test_data)

        # Display the result
        rec_raw = result_df['retention_recommendation'].iloc[0]

        if rec_raw.startswith("{"):
            # Parse the JSON response
            rec_json = json.loads(rec_raw)
            print("\n" + "=" * 70)
            print("AI-GENERATED RETENTION STRATEGY")
            print("=" * 70)
            print(f"\n📊 LIKELY CHURN REASON:")
            print(f"   {rec_json.get('likely_churn_reason', 'N/A')}")
            print(f"\n⚠️  RISK SUMMARY:")
            print(f"   {rec_json.get('risk_summary', 'N/A')}")
            print(f"\n💡 RETENTION ACTION:")
            print(f"   {rec_json.get('retention_action', 'N/A')}")
            print(f"\n🎁 OFFER RECOMMENDATION:")
            print(f"   {rec_json.get('offer_recommendation', 'N/A')}")
            print(f"\n📞 COMMUNICATION TONE:")
            print(f"   {rec_json.get('communication_tone', 'N/A')}")
            print("\n" + "=" * 70)
        else:
            print(f"\nResult: {rec_raw}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. GEMINI_API_KEY set in .env file")
        print("  2. google-generativeai package installed")
        print("  3. Valid API quota available")


if __name__ == "__main__":
    main()
