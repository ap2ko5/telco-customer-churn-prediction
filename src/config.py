"""
config.py
=========
Single source of truth for every hyperparameter, path, and constant
in the Stacked Churn Intelligence System.

Rule: nothing in src/ or app.py imports a magic number directly —
      everything comes from here.
"""
from __future__ import annotations

import logging
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
LOGS_DIR     = PROJECT_ROOT / "logs"

for _dir in (MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Dataset ────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "Churn"
TEST_SIZE     = 0.2

# Columns that are identifiers — dropped automatically before training
ID_COLUMNS = {"customerID", "customer_id", "id", "ID", "CustomerID"}

# ── XGBoost Hyperparameters ────────────────────────────────────────────────────
XGB_PARAMS: dict = {
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
NN_PARAMS: dict = {
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
META_LR_PARAMS: dict = {
    "C":            1.0,
    "max_iter":     1000,
    "random_state": RANDOM_STATE,
}

# ── Calibration ────────────────────────────────────────────────────────────────
CALIBRATION_METHOD = "isotonic"   # "isotonic" | "sigmoid" (Platt Scaling)
CALIBRATION_CV     = 3

# ── Risk Band Thresholds ───────────────────────────────────────────────────────
# Each band is a half-open interval [lo, hi).  Exactly 1.0 is handled as Critical.
RISK_BANDS: dict[str, tuple[float, float]] = {
    "Low":      (0.0,  0.3),
    "Medium":   (0.3,  0.6),
    "High":     (0.6,  0.8),
    "Critical": (0.8,  1.01),   # 1.01 so P=1.0 is caught by the range check
}

# ── Business Impact ────────────────────────────────────────────────────────────
MONTHLY_CHARGES_COL       = "MonthlyCharges"
TENURE_COL                = "tenure"
ESTIMATED_CONTRACT_MONTHS = 24    # heuristic: typical contract length (months)
FALLBACK_MONTHLY_CHARGES  = 65.0  # median-ish fallback when column is absent
FALLBACK_TENURE           = 12    # fallback tenure when column is absent

# ── Gemini AI ──────────────────────────────────────────────────────────────────
GEMINI_MODEL            = "gemini-2.0-flash"
GEMINI_BATCH_SIZE       = 5
GEMINI_RETRY_MAX        = 3
GEMINI_RETRY_WAIT       = 5   # seconds between retries
GEMINI_RATE_LIMIT_SLEEP = 1   # seconds between batches

# ── Output file paths ──────────────────────────────────────────────────────────
PREDICTIONS_CSV = OUTPUTS_DIR / "churn_predictions.csv"
SUMMARY_TXT     = OUTPUTS_DIR / "summary_report.txt"
PROB_PLOT       = OUTPUTS_DIR / "prob_distribution.png"
BAND_PLOT       = OUTPUTS_DIR / "band_distribution.png"
SHAP_PLOT       = OUTPUTS_DIR / "shap_importance.png"
