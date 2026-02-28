"""
xgb_model.py
============
XGBoost base model for churn prediction.

Design choices:
  - scale_pos_weight handles class imbalance automatically
  - early stopping on a validation split prevents overfitting
  - model is saved with a run-id suffix for traceability
"""
from __future__ import annotations

import logging
import time

import joblib
import numpy as np
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_STATE, XGB_EARLY_STOPPING_ROUNDS, XGB_PARAMS

logger = logging.getLogger(__name__)


def build_xgb(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Construct (but not fit) the XGBClassifier with project defaults."""
    return XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": scale_pos_weight})


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:  np.ndarray | None = None,
    y_val:  np.ndarray | None = None,
    *,
    save: bool = True,
    model_name: str = "xgb_model",
) -> XGBClassifier:
    """
    Train XGBoost on (X_train, y_train).

    If (X_val, y_val) are provided, early stopping is applied against
    the AUC on the validation set to prevent overfitting.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val     : optional validation data for early stopping
    save             : persist model to disk when True
    model_name       : filename prefix (without .joblib)
    """
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    model = build_xgb(scale_pos_weight=neg / max(pos, 1))

    t0 = time.perf_counter()
    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    elapsed = time.perf_counter() - t0
    best_iter = getattr(model, "best_iteration", "N/A")
    logger.info("XGBoost trained in %.1fs | best_iteration=%s", elapsed, best_iter)

    if save:
        out = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, out)
        logger.info("XGBoost saved → %s", out)

    return model


def predict_proba_xgb(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return P(churn=1) for each row in X."""
    return model.predict_proba(X)[:, 1]
