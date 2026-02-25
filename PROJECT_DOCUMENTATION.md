# STACKED CHURN INTELLIGENCE SYSTEM
## Complete Project Documentation
### Telco Customer Churn Prediction
**Version:** 1.0 | **Date:** February 2026 | **Author:** Abel Abraham

---

# TABLE OF CONTENTS

1. [Technology Incorporated](#1-technology-incorporated)
2. [System Architecture](#2-system-architecture)
3. [ER Diagram](#3-er-diagram-entity-relationship)
4. [Software Requirements Specification (SRS)](#4-software-requirements-specification-srs)
   - 4.1 Functional Requirements
   - 4.2 Non-Functional Requirements
     - Design & Implementation Constraints
     - External Interfaces
     - Other Non-Functional Requirements
   - 4.3 Goal of Implementation
5. [Half-Implementation Analysis](#5-half-implementation-analysis)
6. [Pipeline Step-by-Step Walkthrough](#6-pipeline-step-by-step-walkthrough)

---

# 1. TECHNOLOGY INCORPORATED

## 1.1 Core Language & Runtime
| Technology | Version | Role |
|---|---|---|
| **Python** | 3.10+ | Primary language for all ML and data processing |

## 1.2 Data & ML Libraries
| Library | Version (min) | Role in System |
|---|---|---|
| **pandas** | ≥ 2.0 | CSV ingestion, DataFrame manipulation, feature pipeline |
| **scikit-learn** | ≥ 1.3 | Preprocessing (ColumnTransformer, StandardScaler, OneHotEncoder, SimpleImputer), Stacking (StackingClassifier), Meta-model (LogisticRegression), Calibration (IsotonicRegression), Evaluation (ROC-AUC, F1, Brier) |
| **XGBoost** | ≥ 2.0 | Base Model A — gradient-boosted trees with early stopping and class imbalance handling via `scale_pos_weight` |
| **scikit-learn MLPClassifier** | (via sklearn) | Base Model B — Multi-Layer Perceptron Neural Network with adaptive learning rate, early stopping |
| **joblib** | ≥ 1.3 | Serialization / deserialization of all model artifacts (`.joblib` files) |
| **NumPy** | (via sklearn) | Numerical array operations, OOF stacking arrays |
| **SHAP** | ≥ 0.44 | XGBoost feature importance explainability (TreeExplainer) |
| **SciPy** | ≥ 1.11 | Supporting scientific computations |

## 1.3 Visualization
| Library | Version (min) | Role |
|---|---|---|
| **Matplotlib** | ≥ 3.8 | Probability distribution histogram, SHAP bar chart (dark-theme, Agg backend) |
| **Seaborn** | ≥ 0.13 | Statistical plot support |

## 1.4 AI / External API
| Technology | Version (min) | Role |
|---|---|---|
| **Google Gemini API** (`google-generativeai`) | ≥ 0.5 | AI-powered personalized retention recommendation generation for High/Critical risk customers |
| **Gemini 2.0 Flash** (model) | — | LLM used for structured JSON retention strategy generation |

## 1.5 Configuration & Environment
| Technology | Version (min) | Role |
|---|---|---|
| **python-dotenv** | ≥ 1.0 | Loads `GEMINI_API_KEY` from `.env` file at runtime |
| **tqdm** | ≥ 4.65 | Progress bar for Gemini batch processing |
| **argparse** | stdlib | CLI argument parsing for `train.py`, `train_pipeline.py`, `predict.py` |

## 1.6 Project Structure
```
telco-customer-churn-prediction/
├── data/
│   └── customer_churn.csv          ← Raw Telco dataset (~7,043 customers)
├── models/                         ← Serialized model artifacts
│   ├── preprocessing_pipeline.joblib
│   ├── xgb_model.joblib
│   ├── nn_model.joblib
│   ├── meta_model.joblib
│   └── model_info.json
├── outputs/                        ← Generated reports & plots
│   ├── churn_predictions.csv
│   ├── summary_report.txt
│   ├── prob_distribution.png
│   ├── band_distribution.png
│   └── shap_importance.png
├── logs/                           ← Per-run JSON logs
│   └── run_YYYYMMDD_HHMMSS.json
├── src/                            ← All source modules
│   ├── config.py                   ← Central config (hyperparams, paths, constants)
│   ├── data_loader.py              ← CSV ingestion + feature type detection
│   ├── preprocessor.py             ← ColumnTransformer pipeline
│   ├── xgb_model.py                ← XGBoost base model
│   ├── nn_model.py                 ← MLP Neural Network base model
│   ├── stacking.py                 ← OOF stacking + meta-model
│   ├── calibration.py              ← Isotonic/Platt probability calibration
│   ├── evaluation.py               ← Multi-model metric comparison
│   ├── risk_segmentation.py        ← Churn probability → risk band assignment
│   ├── business_impact.py          ← Expected revenue loss calculation
│   ├── retention_ai.py             ← Gemini API retention recommendations
│   ├── reporting.py                ← CSV + TXT reports + visualizations
│   ├── train_pipeline.py           ← Main orchestration (14 steps)
│   ├── train.py                    ← Simplified single-model training CLI
│   └── predict.py                  ← Single-customer inference CLI
├── .env.example                    ← Template for GEMINI_API_KEY
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 2. SYSTEM ARCHITECTURE

## 2.1 High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    STACKED CHURN INTELLIGENCE SYSTEM                     │
│                         (train_pipeline.py)                              │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 1 — DATA LAYER                                                    │
│  data_loader.py                                                          │
│  ┌────────────────┐    ┌────────────────┐    ┌─────────────────────┐   │
│  │ Read CSV       │───▶│ Normalize      │───▶│ Drop ID Columns     │   │
│  │ (pandas)       │    │ Target (0/1)   │    │ (customerID etc.)   │   │
│  └────────────────┘    └────────────────┘    └─────────────────────┘   │
│                                                         │               │
│                                               ┌─────────▼───────────┐  │
│                                               │ clean_features()    │  │
│                                               │ Auto-coerce numeric │  │
│                                               │ strings (TotalCharge│  │
│                                               └─────────────────────┘  │
│                                                         │               │
│                                               ┌─────────▼───────────┐  │
│                                               │ Auto-detect cols:   │  │
│                                               │ numeric_cols        │  │
│                                               │ categorical_cols    │  │
│                                               └─────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 2 — PREPROCESSING LAYER                                           │
│  preprocessor.py                                                         │
│                                                                          │
│  ┌────────────── ColumnTransformer ─────────────────┐                  │
│  │                                                   │                  │
│  │  Numeric Pipeline          Categorical Pipeline   │                  │
│  │  ─────────────────         ───────────────────── │                  │
│  │  SimpleImputer(median)     SimpleImputer(freq)   │                  │
│  │        │                         │               │                  │
│  │  StandardScaler()          OneHotEncoder()        │                  │
│  │        │                   (handle_unknown=ignore)│                  │
│  │        └──────────┬────────────┘                 │                  │
│  │                   │                              │                  │
│  │          X_transformed (dense numpy)              │                  │
│  └───────────────────────────────────────────────────┘                  │
│                                                                          │
│  Saved → models/preprocessing_pipeline.joblib                          │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 3 — STACKING LAYER (No Data Leakage)                             │
│  stacking.py                                                             │
│                                                                          │
│  StratifiedKFold (5 folds, random_state=42)                            │
│                                                                          │
│  ┌──── For each fold k ────────────────────────────────────┐           │
│  │                                                          │           │
│  │  X_train_fold ──▶ XGBClassifier ──▶ P_xgb[val_fold]   │           │
│  │                   (scale_pos_weight, early_stopping)    │           │
│  │                                                          │           │
│  │  X_train_fold ──▶ MLPClassifier ──▶ P_nn[val_fold]    │           │
│  │                   (256→128→64, relu+adam)               │           │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  After all folds:                                                        │
│  Stacked_X_train = column_stack([P_xgb_oof, P_nn_oof])  shape (n, 2)  │
│                                                                          │
│  LogisticRegression(meta) ──trainon──▶ Stacked_X_train                 │
│  Saved → models/meta_model.joblib                                       │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 4 — FINAL TRAINING & INFERENCE                                    │
│  xgb_model.py + nn_model.py                                             │
│                                                                          │
│  Retrain both base models on FULL training set                          │
│  P_xgb_all = predict_proba_xgb(xgb_model, X_all)                      │
│  P_nn_all  = predict_proba_nn(nn_model, X_all)                         │
│  P_stack   = meta.predict_proba([P_xgb_all, P_nn_all])[:, 1]          │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 5 — CALIBRATION LAYER                                             │
│  calibration.py                                                          │
│                                                                          │
│  IsotonicRegression (default) or Platt Scaling (sigmoid)               │
│  Fitted on TEST SET probabilities → reduces systematic bias             │
│  final_probs = calibrated_fn(P_xgb_all, P_nn_all) → [0, 1]           │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 6 — BUSINESS INTELLIGENCE LAYER                                   │
│                                                                          │
│  risk_segmentation.py          business_impact.py                       │
│  ──────────────────────        ───────────────────────────────          │
│  final_probs → churn_band      Expected_Revenue_Loss =                  │
│  Low:    [0.0, 0.3)              churn_prob × monthly_charges           │
│  Medium: [0.3, 0.6)              × max(0, 24 − tenure)                 │
│  High:   [0.6, 0.8)                                                     │
│  Critical:[0.8, 1.0]           Sorted descending (worst first)          │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 7 — AI RETENTION LAYER                                            │
│  retention_ai.py (Google Gemini 2.0 Flash)                              │
│                                                                          │
│  Filter: churn_band IN ['High', 'Critical']                             │
│                                                                          │
│  For each customer (batches of 5):                                      │
│  ┌──────────────────────────────────────────┐                          │
│  │  Build JSON payload:                     │                          │
│  │  { customer_profile, churn_probability,  │                          │
│  │    risk_band }                            │                          │
│  │         │                                │                          │
│  │         ▼                                │                          │
│  │  Gemini 2.0 Flash API                    │                          │
│  │         │                                │                          │
│  │         ▼                                │                          │
│  │  JSON Response:                          │                          │
│  │  { likely_churn_reason,                  │                          │
│  │    risk_summary,                         │                          │
│  │    retention_action,                     │                          │
│  │    offer_recommendation,                 │                          │
│  │    communication_tone }                  │                          │
│  └──────────────────────────────────────────┘                          │
│  Retry logic: up to 3 attempts with exponential wait                   │
│  Rate-limit sleep: 1s between batches                                  │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 8 — REPORTING LAYER                                               │
│  reporting.py                                                            │
│                                                                          │
│  ├── outputs/churn_predictions.csv    (all customers + all fields)     │
│  ├── outputs/summary_report.txt       (band %, revenue at risk, top50) │
│  ├── outputs/prob_distribution.png    (histogram with band markers)    │
│  ├── outputs/band_distribution.png    (bar chart per band)            │
│  └── outputs/shap_importance.png      (XGBoost SHAP top-20 features)  │
└────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Module Dependency Map

```
train_pipeline.py (Orchestrator)
    ├── config.py           ← All hyperparams, paths, constants
    ├── data_loader.py      ← uses: config.py
    ├── preprocessor.py     ← uses: config.py
    ├── xgb_model.py        ← uses: config.py
    ├── nn_model.py         ← uses: config.py
    ├── stacking.py         ← uses: config.py, xgb_model.py, nn_model.py
    ├── calibration.py      ← uses: config.py
    ├── evaluation.py       ← standalone (numpy, pandas, sklearn.metrics)
    ├── risk_segmentation.py ← uses: config.py
    ├── business_impact.py  ← uses: config.py
    ├── retention_ai.py     ← uses: config.py, google.generativeai
    └── reporting.py        ← uses: config.py, shap, matplotlib, seaborn

train.py (Simplified CLI)
    └── Self-contained single-file version for quick experimentation

predict.py (Inference CLI)
    └── Loads joblib artifact → predicts single customer
```

---

# 3. ER DIAGRAM (Entity-Relationship)

> **Note:** This system is a batch ML pipeline, not a traditional RDBMS application.
> The ER diagram represents the **logical data entities and their relationships** as they flow through the pipeline.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LOGICAL DATA ENTITY MODEL                        │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│        CUSTOMER          │  ← Source: customer_churn.csv
├──────────────────────────┤
│ PK  customerID (dropped) │
│     gender               │
│     SeniorCitizen        │
│     Partner              │
│     Dependents           │
│     tenure (numeric)     │
│     PhoneService         │
│     MultipleLines        │
│     InternetService      │
│     OnlineSecurity       │
│     OnlineBackup         │
│     DeviceProtection     │
│     TechSupport          │
│     StreamingTV          │
│     StreamingMovies      │
│     Contract             │
│     PaperlessBilling     │
│     PaymentMethod        │
│     MonthlyCharges (num) │
│     TotalCharges (num*)  │   * auto-coerced from string
│     Churn (target: 0/1)  │
└──────────┬───────────────┘
           │  1
           │  (one customer produces one prediction)
           │  1
           ▼
┌──────────────────────────┐
│      CHURN PREDICTION    │  ← Produced by: stacking + calibration
├──────────────────────────┤
│ FK  customerID (index)   │
│     churn_probability    │  FLOAT [0.0, 1.0]
│     churn_band           │  ENUM { Low, Medium, High, Critical }
└──────────┬───────────────┘
           │  1
           │
           │  1
           ▼
┌──────────────────────────┐
│      BUSINESS IMPACT     │  ← Produced by: business_impact.py
├──────────────────────────┤
│ FK  customerID (index)   │
│     expected_revenue_loss│  FLOAT (USD)
│     estimated_remaining  │  INT (months = 24 - tenure)
│     monthly_charges_used │  FLOAT
└──────────┬───────────────┘
           │  1
           │  (only for High/Critical → Gemini API)
           │  0..1
           ▼
┌──────────────────────────┐
│   RETENTION STRATEGY     │  ← Generated by: retention_ai.py (Gemini)
├──────────────────────────┤
│ FK  customerID (index)   │
│     likely_churn_reason  │  TEXT
│     risk_summary         │  TEXT
│     retention_action     │  TEXT
│     offer_recommendation │  TEXT
│     communication_tone   │  TEXT
└──────────────────────────┘

          RELATIONSHIPS (non-RDBMS, pipeline context):
          ┌──────────────┐
          │ CUSTOMER     │──has_many──▶│ FEATURE_COLUMN │
          └──────────────┘             (20 raw features)

          ┌──────────────┐
          │ CUSTOMER     │──produces──▶│ CHURN_PREDICTION │──extends──▶│ BUSINESS_IMPACT │──may_have──▶│ RETENTION_STRATEGY │
          └──────────────┘
```

## 3.1 Model Artifact Entity Map

```
┌──────────────────────────────────────────────────────────────┐
│                     MODEL ARTIFACTS                          │
├─────────────────────────┬────────────────────────────────────┤
│ Artifact                │ Produced By / Used By              │
├─────────────────────────┼────────────────────────────────────┤
│ preprocessing_pipeline  │ preprocessor.py → transform()      │
│ xgb_model.joblib        │ xgb_model.py → predict_proba_xgb()│
│ nn_model.joblib         │ nn_model.py  → predict_proba_nn()  │
│ meta_model.joblib       │ stacking.py  → stack_predict()     │
│ model_info.json         │ train.py     → metadata store      │
└─────────────────────────┴────────────────────────────────────┘
```

---

# 4. SOFTWARE REQUIREMENTS SPECIFICATION (SRS)

**System Name:** Stacked Churn Intelligence System (SCIS)
**Domain:** Telecommunications — Customer Retention Analytics
**Type:** AI/ML Batch Intelligence Pipeline
**SRS Standard:** IEEE 830-compatible

---

## 4.1 FUNCTIONAL REQUIREMENTS

### FR-01 — Data Ingestion
| ID | Requirement |
|---|---|
| FR-01.1 | The system SHALL accept a CSV file path as a CLI argument (`--data`) |
| FR-01.2 | The system SHALL read the file using `pandas.read_csv()` |
| FR-01.3 | The system SHALL support a configurable target column name (default: `Churn`) |
| FR-01.4 | The system SHALL raise a descriptive `FileNotFoundError` if path does not exist |
| FR-01.5 | The system SHALL raise a descriptive `ValueError` if target column is not found |

### FR-02 — Target Normalization
| ID | Requirement |
|---|---|
| FR-02.1 | The system SHALL map target column values to binary integers: `{Yes, y, true, 1, churn} → 1` and `{No, n, false, 0, stay} → 0` |
| FR-02.2 | Mapping SHALL be case-insensitive |
| FR-02.3 | The system SHALL raise a `ValueError` listing any unrecognized label values |

### FR-03 — Feature Cleaning
| ID | Requirement |
|---|---|
| FR-03.1 | The system SHALL drop all ID-like columns automatically (`customerID`, `customer_id`, `id`, `ID`, `CustomerID`) |
| FR-03.2 | The system SHALL strip leading/trailing whitespace from all string-typed columns |
| FR-03.3 | The system SHALL auto-coerce string columns to numeric if ≥ 90% of non-null values parse as float (handles `TotalCharges` edge case) |
| FR-03.4 | The system SHALL automatically classify remaining columns as numeric or categorical |

### FR-04 — Preprocessing Pipeline
| ID | Requirement |
|---|---|
| FR-04.1 | The system SHALL apply `SimpleImputer(strategy="median")` to all numeric columns |
| FR-04.2 | The system SHALL apply `StandardScaler()` to all numeric columns |
| FR-04.3 | The system SHALL apply `SimpleImputer(strategy="most_frequent")` to all categorical columns |
| FR-04.4 | The system SHALL apply `OneHotEncoder(handle_unknown="ignore")` to all categorical columns |
| FR-04.5 | The system SHALL combine both pipelines in a single `ColumnTransformer` |
| FR-04.6 | The fitted preprocessor SHALL be serialized to `models/preprocessing_pipeline.joblib` |

### FR-05 — XGBoost Base Model
| ID | Requirement |
|---|---|
| FR-05.1 | The system SHALL train an `XGBClassifier` with: `max_depth=4`, `learning_rate=0.05`, `n_estimators=300`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric="auc"`, `tree_method="hist"` |
| FR-05.2 | The system SHALL compute and apply `scale_pos_weight = neg_count / pos_count` to handle class imbalance |
| FR-05.3 | The system SHALL apply early stopping with `early_stopping_rounds=50` on validation set |
| FR-05.4 | The trained model SHALL output `predict_proba(X)[:, 1]` — P(churn) |
| FR-05.5 | The model SHALL be serialized to `models/xgb_model.joblib` |

### FR-06 — Neural Network Base Model
| ID | Requirement |
|---|---|
| FR-06.1 | The system SHALL train an `MLPClassifier` with architecture: `hidden_layer_sizes=(256, 128, 64)`, `activation="relu"`, `solver="adam"` |
| FR-06.2 | The model SHALL use L2 regularization: `alpha=0.001` |
| FR-06.3 | The model SHALL use adaptive learning rate: `learning_rate="adaptive"` |
| FR-06.4 | The model SHALL enable early stopping: `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=20` |
| FR-06.5 | Maximum iterations: `max_iter=500` |
| FR-06.6 | The model SHALL be serialized to `models/nn_model.joblib` |

### FR-07 — Stacking Ensemble (No Data Leakage)
| ID | Requirement |
|---|---|
| FR-07.1 | The system SHALL use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| FR-07.2 | For each fold, the system SHALL: (a) train XGBoost on the fold's training portion, (b) generate OOF probabilities on the validation portion |
| FR-07.3 | For each fold, the system SHALL: (a) train Neural Network on the fold's training portion, (b) generate OOF probabilities on the validation portion |
| FR-07.4 | After all 5 folds, the system SHALL construct `Stacked_X = np.column_stack([oof_xgb, oof_nn])` of shape `(n_train, 2)` |
| FR-07.5 | The system SHALL train a `LogisticRegression` meta-model on `Stacked_X` and `y_train` |
| FR-07.6 | The meta-model SHALL be serialized to `models/meta_model.joblib` |

### FR-08 — Model Evaluation
| ID | Requirement |
|---|---|
| FR-08.1 | The system SHALL evaluate XGBoost, Neural Network, and Stacked Ensemble independently |
| FR-08.2 | Metrics computed SHALL include: ROC-AUC, Precision, Recall, F1-Score, Brier Score |
| FR-08.3 | The system SHALL print a formatted comparative evaluation table to stdout |
| FR-08.4 | Classification threshold for class-based metrics SHALL default to 0.5 |
| FR-08.5 | The system SHALL identify and print the best model by ROC-AUC |

### FR-09 — Probability Calibration
| ID | Requirement |
|---|---|
| FR-09.1 | The system SHALL support probability calibration using Isotonic Regression (default) |
| FR-09.2 | The system SHALL support Platt Scaling as an alternative calibration method |
| FR-09.3 | The calibrator SHALL be fitted on the HELD-OUT test set (not training data) |
| FR-09.4 | The system SHALL clip calibrated probabilities to `[0.0, 1.0]` |
| FR-09.5 | Calibration SHALL be skippable via `--no-calibration` CLI flag |

### FR-10 — Risk Band Segmentation
| ID | Requirement |
|---|---|
| FR-10.1 | The system SHALL classify each customer into one of four risk bands based on `churn_probability` |
| FR-10.2 | Bands: `Low [0.0, 0.3)`, `Medium [0.3, 0.6)`, `High [0.6, 0.8)`, `Critical [0.8, 1.0]` |
| FR-10.3 | The system SHALL add a `churn_band` column to the result DataFrame |
| FR-10.4 | The system SHALL print band distribution counts and percentages |

### FR-11 — Business Impact Calculation
| ID | Requirement |
|---|---|
| FR-11.1 | The system SHALL compute: `expected_revenue_loss = churn_probability × MonthlyCharges × max(0, 24 − tenure)` |
| FR-11.2 | If `MonthlyCharges` is missing, fallback value of $65.00/month SHALL be used |
| FR-11.3 | If `tenure` is missing, fallback value of 12 months SHALL be used |
| FR-11.4 | The result DataFrame SHALL be sorted by `expected_revenue_loss` descending |
| FR-11.5 | The system SHALL print total expected revenue at risk |

### FR-12 — AI Retention Recommendations (Gemini)
| ID | Requirement |
|---|---|
| FR-12.1 | The system SHALL process ONLY customers with `churn_band IN ['High', 'Critical']` |
| FR-12.2 | For each eligible customer, the system SHALL construct a JSON payload with: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, `OnlineSecurity`, `StreamingTV`, `StreamingMovies`, `MultipleLines`, `SeniorCitizen` |
| FR-12.3 | The payload SHALL be sent to `gemini-2.0-flash` via `google.generativeai` client |
| FR-12.4 | The API response SHALL be parsed for JSON keys: `likely_churn_reason`, `risk_summary`, `retention_action`, `offer_recommendation`, `communication_tone` |
| FR-12.5 | Customers SHALL be processed in batches of 5 |
| FR-12.6 | Retry logic SHALL attempt up to 3 times with increasing wait on rate-limit errors (429 / quota) |
| FR-12.7 | Rate-limiting sleep of 1 second SHALL be applied between batches |
| FR-12.8 | If `GEMINI_API_KEY` is not set, the system SHALL gracefully skip and set `retention_recommendation = "N/A (no API key configured)"` |
| FR-12.9 | AI recommendations SHALL be skippable via `--no-ai` CLI flag |
| FR-12.10 | Low/Medium risk customers SHALL receive: `"Standard retention — low/medium risk customer."` |

### FR-13 — Reporting & Outputs
| ID | Requirement |
|---|---|
| FR-13.1 | The system SHALL generate `outputs/churn_predictions.csv` with columns: `churn_probability`, `churn_band`, `expected_revenue_loss`, `retention_recommendation` (first), followed by all feature columns |
| FR-13.2 | The system SHALL generate `outputs/summary_report.txt` containing: total customers, total revenue-at-risk ($), band distribution (% per band), top-50 high-risk customers by revenue loss |
| FR-13.3 | The system SHALL generate `outputs/prob_distribution.png` — a histogram of churn probabilities with vertical band threshold markers |
| FR-13.4 | The system SHALL generate `outputs/band_distribution.png` — a bar chart of customer counts per risk band |
| FR-13.5 | The system SHALL generate `outputs/shap_importance.png` — XGBoost SHAP feature importance (TreeExplainer, top 20 features, sampled up to 500 rows) |

### FR-14 — Run Logging
| ID | Requirement |
|---|---|
| FR-14.1 | Each run SHALL generate a timestamped JSON log file at `logs/run_YYYYMMDD_HHMMSS.json` |
| FR-14.2 | The log SHALL contain: run_id, timestamp, total_time_s, dataset path, target column, n_samples, n_train, n_test, n_features_raw, n_features_after_preprocessing, all hyperparameters, n_folds, random_state, calibration flag, and evaluation metrics for each model |

### FR-15 — Single Customer Inference (CLI)
| ID | Requirement |
|---|---|
| FR-15.1 | The system SHALL provide `predict.py` for single customer inference via CLI |
| FR-15.2 | Input SHALL be accepted as `key=value` pairs (e.g., `tenure=12 MonthlyCharges=65.0`) |
| FR-15.3 | The system SHALL load a serialized pipeline from `--model` argument |
| FR-15.4 | Output SHALL display: predicted class label, churn probability, decision threshold |
| FR-15.5 | Custom threshold SHALL be configurable via `--threshold` argument (default: 0.5) |

---

## 4.2 NON-FUNCTIONAL REQUIREMENTS

### A. Design and Implementation Constraints

| ID | Constraint |
|---|---|
| DC-01 | **Language:** The system MUST be implemented entirely in Python 3.10+ |
| DC-02 | **Reproducibility:** `random_state=42` MUST be set consistently across all ML operations (data split, XGBoost, MLP, LogisticRegression, StratifiedKFold) |
| DC-03 | **No Data Leakage:** Out-of-fold stacking MUST use `StratifiedKFold` — training data from the test set is NEVER used to train or evaluate OOF base models in the same fold |
| DC-04 | **Modular Design:** Each pipeline step MUST be implemented in a dedicated Python module under `src/`. No monolithic scripts |
| DC-05 | **Central Configuration:** ALL hyperparameters, file paths, and constants MUST be defined in `src/config.py` only — no magic numbers in module code |
| DC-06 | **Model Serialization:** All trained model artifacts MUST be saved as `.joblib` files for cross-session reuse |
| DC-07 | **Preprocessing Encapsulation:** The preprocessing pipeline (fitted `ColumnTransformer`) MUST be saved as a separate artifact so new data can be transformed identically without re-fitting |
| DC-08 | **Class Imbalance:** XGBoost MUST use `scale_pos_weight = neg/pos` to handle the Telco dataset's natural churn imbalance (~26% churn rate) |
| DC-09 | **Dense Matrix Output:** OneHotEncoder MUST use `sparse_output=False` (sklearn ≥ 1.2) for compatibility with downstream numpy array operations |
| DC-10 | **UTF-8 Encoding:** All file I/O (reports, logs) MUST use `encoding="utf-8"` to ensure cross-platform compatibility (especially Windows stdout) |
| DC-11 | **Headless Plotting:** Matplotlib MUST use the `Agg` backend (`matplotlib.use("Agg")`) to prevent GUI pop-ups in server/batch environments |
| DC-12 | **Environment Variables:** The Gemini API key MUST be loaded from a `.env` file using `python-dotenv` — it MUST NEVER be hardcoded in source code |
| DC-13 | **Calibration Data Isolation:** Probability calibration MUST be fitted on the test set (held out from all training) — it MUST NOT use training data |
| DC-14 | **Fallback Handling:** All external dependencies (SHAP, google-generativeai) MUST be wrapped in `try/except ImportError` with graceful degradation |

### B. External Interfaces Required

#### B.1 Input Interfaces
| Interface | Type | Description |
|---|---|---|
| **CSV Dataset** | File Input | Tab-separated or comma-separated customer data file; path passed via `--data` CLI argument |
| **CLI Arguments** | Command Line | `--data`, `--target`, `--api-key`, `--no-calibration`, `--no-ai` for `train_pipeline.py`; `--model`, `--input`, `--threshold` for `predict.py` |
| **`.env` File** | Environment Config | `GEMINI_API_KEY=<key>` loaded via `python-dotenv` at runtime |

#### B.2 Output Interfaces
| Interface | Type | Description |
|---|---|---|
| **`outputs/churn_predictions.csv`** | CSV File | Full dataset with predicted churn probability, risk band, revenue loss, AI recommendation per customer |
| **`outputs/summary_report.txt`** | Text File | Human-readable summary: band distribution, revenue at risk, top-50 table |
| **`outputs/prob_distribution.png`** | PNG Image | Dark-theme histogram with risk band boundary markers |
| **`outputs/band_distribution.png`** | PNG Image | Dark-theme bar chart of customer counts per band |
| **`outputs/shap_importance.png`** | PNG Image | SHAP feature importance for XGBoost (TreeExplainer) |
| **`logs/run_<timestamp>.json`** | JSON File | Per-run metadata, hyperparameters, timing, evaluation metrics |
| **`models/`** | Directory | Four `.joblib` artifacts: preprocessing_pipeline, xgb_model, nn_model, meta_model |
| **`stdout`** | Console Output | Step-by-step progress reports, metric tables, run summary banner |

#### B.3 External API Interface
| Interface | Protocol | Description |
|---|---|---|
| **Google Gemini API** | HTTPS REST (via SDK) | `google.generativeai` Python SDK; `gemini-2.0-flash` model; request: structured text prompt with JSON customer profile; response: JSON with 5 keys; rate-limit retry with exponential backoff |

### C. Other Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | **Performance** | The full pipeline (7,043 customers, 5-fold stacking) SHOULD complete within 10 minutes on a standard CPU (Intel i5/Ryzen 5, 8GB RAM) |
| NFR-02 | **Performance** | SHAP computation MUST use a sample of max 500 rows to limit computation time |
| NFR-03 | **Scalability** | The pipeline SHOULD support datasets up to ~100,000 customers without code changes; the `tree_method="hist"` in XGBoost ensures this |
| NFR-04 | **Reliability** | Gemini API calls MUST have retry logic (3 attempts, exponential backoff) to handle transient network failures |
| NFR-05 | **Reliability** | The pipeline MUST NOT crash if SHAP or google-generativeai is not installed — graceful degradation with informative warnings |
| NFR-06 | **Maintainability** | All modules MUST include Google-style docstrings describing inputs, outputs, and behavior |
| NFR-07 | **Maintainability** | All hyperparameters MUST be centralized in `config.py` so they can be tuned without modifying model modules |
| NFR-08 | **Security** | API keys MUST NOT appear in source code, commit history, or log files |
| NFR-09 | **Security** | `.env` MUST be listed in `.gitignore` |
| NFR-10 | **Portability** | The system MUST run on Windows, Linux, and macOS with no OS-specific code |
| NFR-11 | **Usability** | The CLI MUST provide `--help` documentation for all arguments |
| NFR-12 | **Auditability** | Each run MUST produce a unique timestamped log file (`run_YYYYMMDD_HHMMSS.json`) for experiment tracking |
| NFR-13 | **Explainability** | The system MUST generate SHAP feature importance plots to explain model decisions (regulatory/business audit compliance) |
| NFR-14 | **Data Integrity** | Output CSV column ordering MUST place prediction fields first (`churn_probability`, `churn_band`, `expected_revenue_loss`, `retention_recommendation`) for business user readability |
| NFR-15 | **Availability** | The pipeline MUST operate fully offline when `--no-ai` flag is set (no external API dependency) |

---

## 4.3 GOAL OF IMPLEMENTATION

### Primary Goal
> **To build a production-grade, AI-augmented customer churn prediction pipeline for a telecommunications company that enables proactive, data-driven customer retention at scale.**

### Specific Objectives

| # | Objective | How Achieved |
|---|---|---|
| 1 | **Accurate Churn Probability** | Stacked ensemble (XGBoost + MLP + LogisticRegression meta-model) with 5-fold OOF to eliminate data leakage |
| 2 | **Reliable Probability Scores** | Isotonic Regression calibration fitted on held-out test set — critical for business decisions based on probability thresholds |
| 3 | **Actionable Risk Segments** | 4-band risk classification (Low/Medium/High/Critical) translates model output into business-actionable customer groups |
| 4 | **Revenue Impact Quantification** | `expected_revenue_loss` formula ties each customer's churn risk directly to financial impact, enabling ROI-based prioritization |
| 5 | **Personalized AI Retention** | Gemini 2.0 Flash generates individualized 5-component retention strategies for High/Critical customers — not generic scripts |
| 6 | **Full Reproducibility** | `random_state=42` everywhere + serialized preprocessing + per-run JSON logs ensure experiments are fully reproducible |
| 7 | **Explainability** | SHAP feature importance ensures model is interpretable for business stakeholders and regulatory review |
| 8 | **Operational Readiness** | CLI interface, `.env` API key handling, modular design, graceful degradation — ready for cron-job or pipeline integration |

### Business Value Statement
> The system transforms raw customer data into a prioritized, actionable retention campaign list — ranked by financial risk, segmented by probability, and enriched with AI-generated personalized offers — enabling the telecom company's retention team to focus interventions on the highest-value at-risk customers first.

---

# 5. HALF-IMPLEMENTATION ANALYSIS

This section documents **what is fully implemented**, **what is partially implemented**, and **what is missing** based on direct code analysis.

## 5.1 ✅ Fully Implemented

| Component | Module | Status |
|---|---|---|
| CSV ingestion with error handling | `data_loader.py` | ✅ Complete |
| Target normalization (multi-label support) | `data_loader.py` | ✅ Complete |
| ID column auto-dropping | `data_loader.py` | ✅ Complete |
| `TotalCharges` numeric coercion (90% rule) | `data_loader.py` | ✅ Complete |
| ColumnTransformer preprocessing | `preprocessor.py` | ✅ Complete |
| XGBoost base model with imbalance handling | `xgb_model.py` | ✅ Complete |
| Neural Network (MLP) base model | `nn_model.py` | ✅ Complete |
| OOF Stacking (5-fold StratifiedKFold) | `stacking.py` | ✅ Complete |
| Logistic Regression meta-model | `stacking.py` | ✅ Complete |
| All four `.joblib` model serialization | all model modules | ✅ Complete |
| Isotonic Regression calibration | `calibration.py` | ✅ Complete |
| Platt Scaling calibration | `calibration.py` | ✅ Complete |
| ROC-AUC, Precision, Recall, F1, Brier evaluation | `evaluation.py` | ✅ Complete |
| classification_report printing | `evaluation.py` | ✅ Complete |
| 4-band risk segmentation | `risk_segmentation.py` | ✅ Complete |
| Expected revenue loss calculation | `business_impact.py` | ✅ Complete |
| Gemini API integration with retry | `retention_ai.py` | ✅ Complete |
| Batch processing with rate-limit sleep | `retention_ai.py` | ✅ Complete |
| Graceful fallback (no API key) | `retention_ai.py` | ✅ Complete |
| CSV output generation | `reporting.py` | ✅ Complete |
| Text summary report | `reporting.py` | ✅ Complete |
| Probability distribution plot | `reporting.py` | ✅ Complete |
| Band distribution bar chart | `reporting.py` | ✅ Complete |
| SHAP feature importance plot | `reporting.py` | ✅ Complete |
| Per-run JSON logging | `train_pipeline.py` | ✅ Complete |
| 14-step orchestration pipeline | `train_pipeline.py` | ✅ Complete |
| Single-model CLI (`train.py`) | `train.py` | ✅ Complete |
| Single customer inference CLI | `predict.py` | ✅ Complete |
| Central configuration | `config.py` | ✅ Complete |
| Windows UTF-8 stdout fix | `train_pipeline.py` | ✅ Complete |

## 5.2 ⚠️ Partially Implemented

| Component | Gap | Recommendation |
|---|---|---|
| **Model versioning** | Models are overwritten each run (no version tag in filename) | Append `run_id` to model filenames: `xgb_model_20260225_221746.joblib` |
| **Hyperparameter logging in model files** | `model_info.json` only exists in `train.py` (simple), not in `train_pipeline.py` | Save a `pipeline_info.json` with all hyperparams after `train_pipeline.py` runs |
| **Platt Scaling calibration** | Code exists in `calibration.py` but `config.py` is hardcoded to `"isotonic"` | Expose `--calibration-method` CLI argument in `train_pipeline.py` |
| **SHAP for NN / meta-model** | SHAP is only computed for XGBoost; Neural Network has no explainability output | Add SHAP KernelExplainer for NN (computationally expensive but possible) |
| **Gemini response validation** | `_parse_response()` returns raw string on JSON decode failure (silent failure) | Add schema validation — check all 5 required keys exist before storing |
| **predict.py integration** | `predict.py` uses the simple `train.py` pipeline format; not compatible with the advanced stacked pipeline in `train_pipeline.py` | Create `predict_stacked.py` that loads all four `.joblib` artifacts and runs the full stacked inference |

## 5.3 ❌ Not Yet Implemented (Future Work)

| Component | Priority | Details |
|---|---|---|
| **Model retraining pipeline** | HIGH | Automated periodic retraining when data drift is detected |
| **Data drift detection** | HIGH | Monitor feature distributions over time (PSI / KL divergence) |
| **REST API endpoint** | MEDIUM | Flask/FastAPI wrapper to serve predictions as HTTP endpoint (instead of CLI only) |
| **Dashboard / UI** | MEDIUM | Streamlit or Dash-based visualization dashboard for non-technical business users |
| **Automated hyperparameter tuning** | MEDIUM | Optuna or scikit-learn GridSearchCV for XGBoost and NN |
| **Cross-validation for calibration** | LOW | `CalibratedClassifierCV` with CV folds instead of single test-set fitting |
| **Unit tests** | LOW | `pytest` test suite for `data_loader`, `preprocessor`, `risk_segmentation`, `business_impact` |
| **CI/CD pipeline** | LOW | GitHub Actions for automated test + lint on push |
| **Docker containerization** | LOW | `Dockerfile` for reproducible deployment environment |
| **Database output** | LOW | Write predictions to PostgreSQL/SQLite instead of CSV only |

---

# 6. PIPELINE STEP-BY-STEP WALKTHROUGH

Below is a sequential trace of exactly what happens when you run:

```bash
python src/train_pipeline.py --data data/customer_churn.csv --target Churn
```

| Step | Module Called | Function | Action |
|---|---|---|---|
| 1 | `data_loader.py` | `load_data()` | Read CSV → normalize Churn → drop customerID → clean_features (coerce TotalCharges) → detect numeric(3)/categorical(16) columns |
| 2 | `train_pipeline.py` | `train_test_split()` | Split 80/20 stratified: ~5,634 train / ~1,409 test |
| 3 | `preprocessor.py` | `build_preprocessor()` + `fit_preprocessor()` | Fit ColumnTransformer on train → transform train, test, all → save `.joblib` |
| 4 | `stacking.py` | `generate_oof_predictions()` | 5-fold OOF: train XGBClassifier + MLPClassifier per fold → collect `oof_xgb[n_train]`, `oof_nn[n_train]` |
| 5 | `stacking.py` | `train_meta_model()` | Fit `LogisticRegression` on `[oof_xgb, oof_nn]` → save `meta_model.joblib` |
| 6 | `xgb_model.py` | `train_xgb()` | Retrain full XGBoost on all train data (with early stopping on test set) → save `xgb_model.joblib` |
| 6 | `nn_model.py` | `train_nn()` | Retrain full MLP on all train data → save `nn_model.joblib` |
| 7 | Both | `predict_proba_*()` | Generate `P_xgb_test`, `P_nn_test`, then `P_stack_test` via meta-model |
| 8 | `evaluation.py` | `compare_models()` | Compare XGBoost vs NN vs Stacked on test metrics → print table → identify best by ROC-AUC |
| 9 | `calibration.py` | `calibrate_probabilities()` | Fit IsotonicRegression on TEST SET → generate `final_probs` for full dataset (calibrated) |
| 10 | `risk_segmentation.py` | `add_risk_band()` | Map `final_probs` → `churn_band` for every customer → print distribution |
| 11 | `business_impact.py` | `compute_business_impact()` | Compute `expected_revenue_loss` → sort descending → print total at risk |
| 12 | `retention_ai.py` | `generate_retention_recommendations()` | Filter High/Critical → batch Gemini API calls → store JSON recommendations |
| 13 | `reporting.py` | `generate_all_reports()` | Save CSV, TXT summary, 3 PNGs |
| 14 | `train_pipeline.py` | Log writing | Serialize all metadata to `logs/run_<timestamp>.json` → print completion banner |

---

*End of Project Documentation — Stacked Churn Intelligence System v1.0*
*Generated from codebase analysis: February 25, 2026*
