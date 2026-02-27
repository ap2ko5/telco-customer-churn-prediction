import sys
from pathlib import Path

# Important: Add 'src' to the path so python can find our modules
sys.path.insert(0, str(Path("src").resolve()))

# Import our inference functions from predict_stacked
from predict_stacked import load_artifacts, predict_single

def main():
    print("Loading AI Models (XGBoost + Neural Network + Meta-Model)...\n")
    # 1. Load the 4 trained artifacts from the models/ directory
    preprocessor, xgb_model, nn_model, meta_model = load_artifacts()

    # 2. Create your customer data dictionary
    # You must provide all 19 features the model was trained on
    high_risk_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,                           # Very new customer
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",      # Expensive service
        "OnlineSecurity": "No",                # No added benefits
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",          # Easy to cancel
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,               # High bill
        "TotalCharges": 171.00
    }

    # 3. Pass the data through the stacked ensemble
    result = predict_single(
        customer=high_risk_customer,
        preprocessor=preprocessor,
        xgb_model=xgb_model,
        nn_model=nn_model,
        meta_model=meta_model
    )

    # 4. Print the result
    print("\n" + "="*50)
    print(" [~] CHURN PREDICTION RESULT")
    print("="*50)
    print(f"Final Churn Probability :  {result['churn_probability']:.1%}")
    print(f"Assigned Risk Band      :  {result['risk_band']}")
    print(f"Expected Revenue Loss   : ${result['expected_revenue_loss']:,.2f}")
    print("\n--- Model Breakdown ---")
    print(f"XGBoost Says            :  {result['xgb_probability']:.1%}")
    print(f"Neural Network Says     :  {result['nn_probability']:.1%}")
    print("="*50)

if __name__ == "__main__":
    main()
