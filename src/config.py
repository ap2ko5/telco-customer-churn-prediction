"""
Central configuration for the Stacked Churn Intelligence System.
All hyperparameters, paths, and constants live here.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
LOGS_DIR     = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Dataset ────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Churn"
TEST_SIZE     = 0.2

ID_COLUMNS = ["customerID", "customer_id", "id", "ID", "CustomerID"]  # dropped automatically

# ── XGBoost Hyperparameters ────────────────────────────────────────────────────
XGB_PARAMS = {
    "max_depth":        4,
    "learning_rate":    0.05,
    "n_estimators":     300,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "tree_method":      "hist",
}
XGB_EARLY_STOPPING_ROUNDS = 50

# ── Neural Network Hyperparameters ─────────────────────────────────────────────
NN_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation":         "relu",
    "solver":             "adam",
    "alpha":              0.001,
    "batch_size":         "auto",
    "learning_rate":      "adaptive",
    "max_iter":           500,
    "early_stopping":     True,
    "validation_fraction": 0.1,
    "n_iter_no_change":   20,
    "random_state":       RANDOM_STATE,
}

# ── Stacking ───────────────────────────────────────────────────────────────────
N_FOLDS = 5

# ── Meta-model ─────────────────────────────────────────────────────────────────
META_LR_PARAMS = {
    "C":            1.0,
    "max_iter":     1000,
    "random_state": RANDOM_STATE,
}

# ── Calibration ────────────────────────────────────────────────────────────────
CALIBRATION_METHOD = "isotonic"   # "isotonic" or "sigmoid" (Platt Scaling)
CALIBRATION_CV     = 3

# ── Risk Band Thresholds ───────────────────────────────────────────────────────
RISK_BANDS = {
    "Low":      (0.0, 0.3),
    "Medium":   (0.3, 0.6),
    "High":     (0.6, 0.8),
    "Critical": (0.8, 1.01),
}

# ── Business Impact ────────────────────────────────────────────────────────────
MONTHLY_CHARGES_COL         = "MonthlyCharges"
TENURE_COL                  = "tenure"
ESTIMATED_CONTRACT_MONTHS   = 24       # heuristic: typical contract length
FALLBACK_MONTHLY_CHARGES    = 65.0     # median-ish if column missing
FALLBACK_TENURE             = 12

# ── Gemini API ─────────────────────────────────────────────────────────────────
GEMINI_MODEL      = "gemini-2.0-flash"
GEMINI_BATCH_SIZE = 5
GEMINI_RETRY_MAX  = 3
GEMINI_RETRY_WAIT = 5   # seconds between retries
GEMINI_RATE_LIMIT_SLEEP = 1  # seconds between batches

# ── Outputs ────────────────────────────────────────────────────────────────────
PREDICTIONS_CSV   = OUTPUTS_DIR / "churn_predictions.csv"
SUMMARY_TXT       = OUTPUTS_DIR / "summary_report.txt"
PROB_PLOT         = OUTPUTS_DIR / "prob_distribution.png"
BAND_PLOT         = OUTPUTS_DIR / "band_distribution.png"
SHAP_PLOT         = OUTPUTS_DIR / "shap_importance.png"
