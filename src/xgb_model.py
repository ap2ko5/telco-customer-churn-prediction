"""
xgb_model.py — XGBoost base model for churn prediction.

Uses scale_pos_weight to handle class imbalance.
Supports early stopping on a validation split.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_STATE, XGB_EARLY_STOPPING_ROUNDS, XGB_PARAMS


def build_xgb(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Construct (but not fit) the XGBClassifier."""
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    return XGBClassifier(**params)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    save: bool = True,
    model_name: str = "xgb_model",
) -> XGBClassifier:
    """
    Train XGBoost on (X_train, y_train).
    If (X_val, y_val) provided, early stopping is applied.
    """
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    model = build_xgb(scale_pos_weight=spw)

    t0 = time.time()
    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    elapsed = time.time() - t0
    print(f"[xgb_model] Trained in {elapsed:.1f}s | best_iteration={model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

    if save:
        out = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, out)
        print(f"[xgb_model] Saved to {out}")

    return model


def predict_proba_xgb(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
