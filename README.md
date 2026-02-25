# Customer Churn Prediction

This project trains churn models from a CSV dataset and predicts churn for new customer inputs.

## 1) Put your dataset
Place your CSV file at:

`data/customer_churn.csv`

The target column should be `Churn` (values like `Yes/No`, `1/0`, `True/False` are handled).

## 2) Install dependencies
From your virtual environment:

```powershell
pip install -r requirements.txt
```

## 3) Train model
Choose one model type: `logistic`, `xgboost`, `mlp`, or `stacking`.

```powershell
python src/train.py --data data/customer_churn.csv --target Churn --model-type xgboost
```

Stacking example:

```powershell
python src/train.py --data data/customer_churn.csv --target Churn --model-type stacking
```

This saves:
- `models/churn_pipeline.joblib`
- `models/model_info.json`

## 4) Predict for one current customer sample
Pass feature values as key=value pairs:

```powershell
python src/predict.py --model models/churn_pipeline.joblib --threshold 0.5 --input gender=Female SeniorCitizen=0 Partner=Yes Dependents=No tenure=5 PhoneService=Yes MultipleLines=No InternetService=Fiber\ optic OnlineSecurity=No OnlineBackup=No DeviceProtection=No TechSupport=No StreamingTV=Yes StreamingMovies=Yes Contract=Month-to-month PaperlessBilling=Yes PaymentMethod=Electronic\ check MonthlyCharges=85.5 TotalCharges=450.2
```

Output includes:
- Predicted churn class
- Churn probability
- Decision threshold used for classification

## Notes
- Keep feature names exactly as dataset column names (except target).
- If your dataset has `customerID`, it is dropped automatically.
- Numeric-like text columns (for example `TotalCharges`) are auto-converted to numeric.
