# Project Organization & Module Connections
## Telco Customer Churn Prediction System

**Last Updated:** March 6, 2026

---

## 📁 Project Structure

```
telco-customer-churn-prediction/
│
├── 📄 Configuration Files
│   ├── .env                        ← Environment variables (GEMINI_API_KEY)
│   ├── .gitignore                  ← Git ignore rules
│   ├── pyproject.toml              ← Project metadata
│   ├── requirements.txt            ← Python dependencies
│   ├── README.md                   ← Quick start guide
│   ├── PROJECT_DOCUMENTATION.md    ← Full system documentation
│   └── STUDENT_GUIDE.md            ← Educational walkthrough
│
├── 📊 Data
│   └── data/
│       └── customer_churn.csv      ← Training dataset
│
├── 🤖 Trained Models
│   └── models/
│       ├── preprocessing_pipeline.joblib
│       ├── xgb_model.joblib
│       ├── nn_model.joblib
│       ├── meta_model.joblib
│       └── model_info.json
│
├── 📈 Outputs
│   └── outputs/
│       ├── churn_predictions.csv   ← Prediction results
│       ├── summary_report.txt      ← Text summary
│       ├── prob_distribution.png   ← Probability histogram
│       ├── band_distribution.png   ← Risk band chart
│       └── shap_importance.png     ← Feature importance
│
├── 📝 Logs
│   └── logs/
│       └── run_YYYYMMDD_HHMMSS.json
│
├── 🎯 Main Entry Points
│   ├── app.py                      ← Streamlit dashboard
│   └── example_predict.py          ← Simple prediction example
│
├── 🔧 Scripts (Utilities & Demos)
│   └── scripts/
│       ├── __init__.py
│       └── demo_ai_recommendations.py
│
├── 🧪 Core Source Code
│   └── src/
│       ├── __init__.py
│       ├── config.py               ← Central configuration
│       ├── data_loader.py          ← CSV loading & cleaning
│       ├── preprocessor.py         ← Feature engineering pipeline
│       ├── xgb_model.py            ← XGBoost model
│       ├── nn_model.py             ← Neural network model
│       ├── stacking.py             ← Ensemble stacking
│       ├── calibration.py          ← Probability calibration
│       ├── evaluation.py           ← Model evaluation metrics
│       ├── risk_segmentation.py    ← Risk band classification
│       ├── business_impact.py      ← Revenue impact calculation
│       ├── retention_ai.py         ← Gemini AI recommendations
│       ├── reporting.py            ← Report & visualization generation
│       ├── train_pipeline.py       ← Main training orchestrator
│       └── predict_stacked.py      ← Inference CLI
│
└── ✅ Tests
    └── tests/
        ├── __init__.py
        ├── conftest.py             ← Shared pytest fixtures
        ├── test_data_loader.py
        ├── test_preprocessor.py
        ├── test_business_impact.py
        └── test_risk_segmentation.py
```

---

## 🔗 Module Connection Map

### **Central Hub: `config.py`**
All modules import from config.py for:
- File paths (MODELS_DIR, DATA_DIR, OUTPUTS_DIR)
- Hyperparameters (XGB_PARAMS, NN_PARAMS)
- Constants (RISK_BANDS, RANDOM_STATE)
- API settings (GEMINI_MODEL, GEMINI_BATCH_SIZE)

```
                    ┌─────────────┐
                    │  config.py  │ ← Single Source of Truth
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    [All src modules] [app.py]  [tests/conftest.py]
```

### **Training Pipeline Flow**

```
train_pipeline.py (Orchestrator)
    │
    ├──▶ data_loader.py
    │       └── Loads & cleans CSV
    │
    ├──▶ preprocessor.py
    │       └── Builds ColumnTransformer
    │           ├── Numeric: Impute → StandardScale
    │           └── Categorical: Impute → OneHotEncode
    │
    ├──▶ stacking.py
    │       ├── Imports: xgb_model.py, nn_model.py
    │       └── Generates out-of-fold predictions
    │           ├── xgb_model.train_xgb()
    │           ├── nn_model.train_nn()
    │           └── Trains LogisticRegression meta-model
    │
    ├──▶ calibration.py
    │       └── Isotonic/Platt probability calibration
    │
    ├──▶ evaluation.py
    │       └── ROC-AUC, Precision, Recall, F1, Brier
    │
    ├──▶ risk_segmentation.py
    │       └── Assigns risk bands (Low/Medium/High/Critical)
    │
    ├──▶ business_impact.py
    │       └── Calculates expected revenue loss
    │
    ├──▶ retention_ai.py
    │       └── Gemini API calls for High/Critical customers
    │
    └──▶ reporting.py
            └── Generates CSV, TXT, PNG outputs
                ├── Uses SHAP for feature importance
                └── Uses matplotlib/seaborn for charts
```

### **Inference Pipeline Flow**

```
app.py (Streamlit Dashboard)
    │
    ├── Loads joblib artifacts from models/
    │   ├── preprocessing_pipeline.joblib
    │   ├── xgb_model.joblib
    │   ├── nn_model.joblib
    │   └── meta_model.joblib
    │
    ├── Imports from src/:
    │   ├── preprocessor.transform()
    │   ├── xgb_model.predict_proba_xgb()
    │   ├── nn_model.predict_proba_nn()
    │   ├── stacking.stack_predict()
    │   ├── risk_segmentation.add_risk_band()
    │   └── business_impact.compute_business_impact()
    │
    └── Displays interactive dashboard with:
        ├── KPI cards (total customers, revenue at risk)
        ├── Risk distribution charts
        ├── Customer detail table
        └── SHAP feature importance

example_predict.py (Simple CLI Example)
    │
    └── Imports from src/:
        └── predict_stacked.py
            ├── load_artifacts()
            └── predict_single()
```

### **Test Suite Connections**

```
tests/conftest.py
    ├── Sets up sys.path for src/ imports
    └── Provides shared fixtures:
        ├── sample_probabilities
        ├── sample_churn_df
        └── minimal_raw_csv

tests/test_*.py files
    ├── Import from src/ via sys.path
    └── Use conftest fixtures automatically
```

---

## 🔌 Import Path Configuration

### **Pattern 1: Entry Points (Root Level)**
Files: `app.py`, `example_predict.py`
```python
sys.path.insert(0, str(Path("src").resolve()))
# OR
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

### **Pattern 2: Source Modules (src/ Level)**
Files: All modules in `src/`
```python
# Direct imports (no prefix needed, they're in same directory)
from config import MODELS_DIR
from data_loader import load_data
from xgb_model import train_xgb
```

### **Pattern 3: Tests (tests/ Level)**
Files: All test files
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

### **Pattern 4: Scripts (scripts/ Level)**
Files: Demo and utility scripts
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

---

## 🚀 How to Run Everything

### **1. Training**
```powershell
python src/train_pipeline.py --data data/customer_churn.csv --target Churn
```

### **2. Dashboard**
```powershell
streamlit run app.py
```

### **3. Single Prediction**
```powershell
python example_predict.py
```

### **4. CLI Prediction**
```powershell
python src/predict_stacked.py --input tenure=12 MonthlyCharges=85 Contract=Month-to-month
```

### **5. AI Recommendations Demo**
```powershell
python scripts/demo_ai_recommendations.py
```

### **6. Tests**
```powershell
python -m pytest tests/ -v
```

---

## ✅ Connection Verification Checklist

- [x] All src/ modules can import from config.py
- [x] train_pipeline.py can import all required modules
- [x] app.py can import from src/
- [x] example_predict.py can import from src/
- [x] All tests can import from src/
- [x] Scripts directory created with proper imports
- [x] No circular dependencies
- [x] All __init__.py files in place
- [x] .env file exists with API key
- [x] All directories created (models/, outputs/, logs/)

---

## 📊 Module Responsibilities

| Module | Purpose | Key Dependencies |
|--------|---------|------------------|
| `config.py` | Central configuration | None (base module) |
| `data_loader.py` | Load & clean CSV | config |
| `preprocessor.py` | Feature engineering | config, sklearn |
| `xgb_model.py` | XGBoost training/inference | config, xgboost |
| `nn_model.py` | Neural network training/inference | config, sklearn |
| `stacking.py` | Ensemble orchestration | config, xgb_model, nn_model |
| `calibration.py` | Probability calibration | config, sklearn |
| `evaluation.py` | Model metrics | sklearn.metrics |
| `risk_segmentation.py` | Assign risk bands | config |
| `business_impact.py` | Calculate revenue loss | config |
| `retention_ai.py` | Gemini AI recommendations | config, google.generativeai |
| `reporting.py` | Generate outputs | config, matplotlib, shap |
| `train_pipeline.py` | Main orchestrator | ALL above modules |
| `predict_stacked.py` | CLI inference | config, joblib |

---

## 🔧 Environment Variables

Required in `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

---

## 📝 Notes

1. **No Circular Dependencies**: config.py is imported by all, but imports nothing from the project
2. **Reproducibility**: All random operations use `RANDOM_STATE=42`
3. **Error Handling**: Import errors are caught gracefully (e.g., SHAP, Gemini API)
4. **Cross-Platform**: Uses `Path` objects for file operations (works on Windows/Linux/Mac)
5. **Extensibility**: Add new modules following the same import pattern

---

**Status**: ✅ All files organized and connected properly
