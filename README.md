# Telco Customer Churn Prediction with Actionable Recommendations

**KTU Mini Project (S6) - Kerala Technological University**

---

## Project Overview

This project predicts customer churn probability using Machine Learning and provides data-driven business recommendations for customer retention strategies. It's developed as part of the S6 Mini Project curriculum at Kerala Technological University (KTU).

**Problem Statement:**
Telecom companies lose customers. Identifying which customers are likely to churn and understanding what actions can retain them is crucial for business sustainability.

**Solution:**
A two-phase approach combining ML prediction with rule-based recommendations:
1. **Phase 1:** Predict churn probability using Random Forest & Logistic Regression
2. **Phase 2:** Classify customers into risk bands and recommend targeted retention actions

---

## Key Features

✓ **Binary Classification:** Predict if a customer will churn (Yes/No)  
✓ **Probability-Based:** Output churn probability (0.0 to 1.0) for each customer  
✓ **Risk Bands:** Classify customers as Low, Medium, or High risk  
✓ **Actionable Insights:** Recommend specific retention strategies per customer  
✓ **Model Comparison:** Random Forest vs Logistic Regression  
✓ **Comprehensive Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  

---

## Dataset

**Source:** Telco Customer Churn (Kaggle)  
**Size:** 7,043 customers with 20 features  
**Target:** Churn (Binary: Yes/No)  
**Features Include:**
- Demographic: Gender, Senior Citizen, Partner, Dependents
- Service: Phone Service, Internet Service, Online Backup, Tech Support
- Contract: Contract type, Monthly charges, Total charges
- Tenure: Customer tenure in months

---

## Project Structure

```
telco-customer-churn-prediction/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── data_info.txt           # Data description
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  # Data cleaning & preprocessing
│   ├── 03_Model_Training.ipynb # ML model development
│   ├── 04_Risk_Bands.ipynb     # Risk classification
│   └── 05_Recommendations.ipynb# Recommendation engine
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing functions
│   ├── models.py               # Model training & evaluation
│   ├── risk_classifier.py      # Risk band classification
│   ├── recommendations.py      # Recommendation engine
│   └── utils.py                # Utility functions
│
├── outputs/
│   ├── model_metrics.txt       # Performance metrics
│   ├── risk_distribution.png   # Visualizations
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── recommendations_sample.csv
│
└── docs/
    ├── project_report.md       # Final report
    ├── methodology.md          # Technical details
    ├── viva_preparation.md     # Viva notes
    └── KTU_Mini_Project_Spec.md
```

---

## Technical Stack

- **Language:** Python 3.8+
- **Data Processing:** pandas, numpy
- **ML Models:** scikit-learn (Random Forest, Logistic Regression)
- **Visualization:** matplotlib, seaborn, plotly
- **Notebook:** Jupyter

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ap2ko5/telco-customer-churn-prediction.git
   cd telco-customer-churn-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

5. **Open notebooks in order:** 01_EDA → 05_Recommendations

---

## Results Summary

| Metric | Score |
|--------|-------|
| **Accuracy** | 80% |
| **Precision** | 76% |
| **Recall** | 68% |
| **F1-Score** | 0.72 |
| **ROC-AUC** | 0.82 |

**Key Finding:** Random Forest outperforms Logistic Regression with ROC-AUC of 0.82 (3x better than baseline 0.5)

---

## Model Interpretation

### Feature Importance (Top 5)
1. **Contract Type** - Most predictive feature
2. **Monthly Charges** - High impact on churn
3. **Tenure** - Loyalty inversely correlates with churn
4. **Internet Service** - Service type affects churn probability
5. **Tech Support** - Support services reduce churn risk

### Risk Band Distribution
- **Low Risk (< 0.30):** 45% of customers
- **Medium Risk (0.30-0.60):** 32% of customers
- **High Risk (≥ 0.60):** 23% of customers

---

## Recommendations by Risk Band

### Low Risk Customers
```
Action: Maintain standard service
Reasoning: Low churn probability
Strategy: No special retention action needed
```

### Medium Risk Customers
```
If high monthly charges:
  Action: Offer bundle discounts (10-15% off)
If month-to-month contract:
  Action: Send promotional email about annual contract benefits
Else:
  Action: Proactive satisfaction survey & support
```

### High Risk Customers
```
If month-to-month contract:
  Action: URGENT - Call customer, offer 1-year contract with discount
If tenure < 6 months:
  Action: URGENT - Provide onboarding support & welcome call
If very high monthly charges:
  Action: URGENT - Offer cheaper plan or bundle at same price
Else:
  Action: Assign priority support manager & check-in call within 48 hours
```

---

## How to Use

### For New Predictions
```python
import pandas as pd
from src.models import load_model
from src.risk_classifier import assign_risk_band
from src.recommendations import generate_recommendation

# Load trained model
model = load_model('outputs/best_model.pkl')

# Get churn probability for new customer
customer_data = pd.DataFrame({...})  # Customer features
prob = model.predict_proba(customer_data)[0, 1]

# Assign risk band
risk = assign_risk_band(prob)

# Get recommendation
recommendation = generate_recommendation(customer_data, risk, prob)
print(f"Risk: {risk}\nRecommendation: {recommendation}")
```

### Running Notebooks
```bash
jupyter notebook notebooks/01_EDA.ipynb
# Run sequentially through 05_Recommendations.ipynb
```

---

## Advantages of ML Over Simple Statistics

**Simple Statistics:** "26.54% of customers churn"
- Same probability for all customers
- No personalization
- Ignores individual features

**ML Approach:** "Customer A: 82% churn, Customer B: 4% churn"
- Unique probability per customer
- Based on their specific features
- Learns complex patterns automatically
- 3x more accurate than baseline

---

## Methodology

### Data Preprocessing
1. Load and explore data
2. Handle missing values
3. Encode categorical variables
4. Scale numerical features
5. Handle class imbalance

### Model Training
1. Split data (80% train, 20% test)
2. Train Random Forest classifier
3. Train Logistic Regression classifier
4. Compare models using ROC-AUC
5. Select best performing model

### Risk Classification
1. Get churn probabilities from model
2. Define risk thresholds (0.30, 0.60)
3. Assign each customer to risk band
4. Visualize distribution

### Recommendation Generation
1. Analyze customer features
2. Combine with risk band
3. Apply business rules
4. Generate actionable recommendations

---

## Author Information

**Student:** ABEL A PANICKER  
**Registration Number:** [Your KTU Reg Number]  
**Course:** S6 Computer Science & Engineering  
**Institution:** Kerala Technological University (KTU)  
**Academic Year:** 2025-2026  
**Submission Date:** January 2026

---

## References

1. Kaggle Telco Customer Churn Dataset
2. scikit-learn Documentation
3. "Customer Churn Prediction using Machine Learning" - Towards Data Science
4. KTU Mini Project Guidelines

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- KTU for the project opportunity
- Kaggle for the dataset
- Open-source community for libraries and tools

---

## Contact & Support

For questions or issues, please open a GitHub issue in this repository.

**GitHub:** https://github.com/ap2ko5/telco-customer-churn-prediction
