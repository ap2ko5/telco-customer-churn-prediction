"""
stacking.py
===========
Proper stacking ensemble using StratifiedKFold to avoid data leakage.

Protocol (N_FOLDS folds, default = 5):
  For each fold k:
    1. Fit XGBoost and Neural Network on the k-1 training folds.
    2. Predict probabilities on fold k (held-out) → OOF predictions.
  After all folds:
    Stacked_X_train = column_stack([P_xgb_oof, P_nn_oof])   shape (n, 2)
    Fit LogisticRegression meta-model on Stacked_X_train → y_train.

Why OOF? If we used the same data to train base models AND the meta-model,
the meta-model would overfit to perfectly-leaked predictions.
"""
from __future__ import annotations

import logging
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import META_LR_PARAMS, MODELS_DIR, N_FOLDS, RANDOM_STATE
from nn_model import build_nn, predict_proba_nn
from xgb_model import build_xgb, predict_proba_xgb

logger = logging.getLogger(__name__)


def generate_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    neg_pos_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-fold (OOF) predictions for XGBoost and Neural Network.

    Parameters
    ----------
    X_train       : preprocessed training features
    y_train       : binary training labels
    neg_pos_ratio : scale_pos_weight for XGBoost (neg / pos count)

    Returns
    -------
    oof_xgb, oof_nn : probability arrays of shape (n_train,)
    """
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_xgb = np.zeros(len(y_train), dtype=np.float64)
    oof_nn  = np.zeros(len(y_train), dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_trn, X_val = X_train[trn_idx], X_train[val_idx]
        y_trn, y_val = y_train[trn_idx], y_train[val_idx]

        # ── XGBoost fold ──────────────────────────────────────────────────────
        neg, pos = int((y_trn == 0).sum()), int((y_trn == 1).sum())
        xgb = build_xgb(scale_pos_weight=neg / max(pos, 1))
        xgb.set_params(early_stopping_rounds=50)
        xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = predict_proba_xgb(xgb, X_val)

        # ── Neural Network fold ───────────────────────────────────────────────
        nn = build_nn()
        nn.fit(X_trn, y_trn)
        oof_nn[val_idx] = predict_proba_nn(nn, X_val)

        logger.info("Stacking fold %d/%d complete", fold, N_FOLDS)

    return oof_xgb, oof_nn


def train_meta_model(
    oof_xgb: np.ndarray,
    oof_nn:  np.ndarray,
    y_train: np.ndarray,
    *,
    save: bool = True,
    model_name: str = "meta_model",
) -> LogisticRegression:
    """
    Train Logistic Regression meta-model on OOF stacked features.

    Input  : [P_xgb_oof, P_nn_oof]  shape (n_train, 2)
    Target : y_train
    """
    Stacked_X = np.column_stack([oof_xgb, oof_nn])

    meta = LogisticRegression(**META_LR_PARAMS)
    t0   = time.perf_counter()
    meta.fit(Stacked_X, y_train)
    logger.info("Meta-model trained in %.1fs", time.perf_counter() - t0)

    if save:
        out = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(meta, out)
        logger.info("Meta-model saved → %s", out)

    return meta


def stack_predict(
    meta:  LogisticRegression,
    p_xgb: np.ndarray,
    p_nn:  np.ndarray,
) -> np.ndarray:
    """Return stacked churn probabilities from base-model outputs."""
    return meta.predict_proba(np.column_stack([p_xgb, p_nn]))[:, 1]
