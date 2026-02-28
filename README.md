# Telco Customer Churn Prediction

A stacked ensemble churn intelligence system using XGBoost + Neural Network + Meta-Model, with a Streamlit dashboard for business users.

## 1) Dataset

Place your CSV file at:

```
data/customer_churn.csv
```

The target column should be `Churn` (values like `Yes/No`, `1/0`, `True/False` are handled automatically). A `customerID` column is dropped automatically if present.

## 2) Install dependencies

From your virtual environment:

```powershell
pip install -r requirements.txt
```

## 3) Train the pipeline

```powershell
python src/train_pipeline.py --data data/customer_churn.csv --target Churn
```

**Optional flags:**

| Flag | Description |
|------|-------------|
| `--no-ai` | Skip Gemini AI retention recommendations |
| `--no-calibration` | Skip probability calibration |
| `--calibration-method` | `isotonic` (default) or `sigmoid` |
| `--api-key YOUR_KEY` | Gemini API key (or set `GEMINI_API_KEY` in `.env`) |

This saves model artifacts to `models/` and outputs to `outputs/`.

## 4) Launch the dashboard

```powershell
streamlit run app.py
```

Upload any customer CSV in the sidebar to run live predictions.

## 5) Predict for a single customer (CLI)

```powershell
python src/predict_stacked.py --input tenure=2 MonthlyCharges=85.50 Contract=Month-to-month InternetService=Fiber optic
```

Use `--run-id <run_id>` to target a specific training run instead of the latest.

## 6) Run tests

```powershell
python -m pytest tests/ -v
```

## Notes

- Keep feature names exactly as they appear in your dataset column headers.
- Numeric-like text columns (e.g. `TotalCharges`) are auto-converted to numeric.
- Model artifacts are versioned by run timestamp (e.g. `xgb_model_20260226_013000.joblib`).
