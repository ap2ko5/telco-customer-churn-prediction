"""
stacking.py — Proper stacking ensemble to avoid data leakage.

Protocol (StratifiedKFold, 5 folds):
  For each fold k:
    1. Fit XGB and NN on train-fold (preprocessed).
    2. Predict probabilities on validation-fold → OOF probabilities.
  After all folds:
    Stacked_X_train = [P_xgb_oof, P_nn_oof]
    Fit LogisticRegression meta-model on Stacked_X_train → y_train.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import META_LR_PARAMS, MODELS_DIR, N_FOLDS, RANDOM_STATE
from nn_model import build_nn, predict_proba_nn
from xgb_model import build_xgb, predict_proba_xgb


def generate_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    neg_pos_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-fold (OOF) predictions for XGBoost and Neural Network
    using StratifiedKFold to prevent data leakage.

    Returns
    -------
    oof_xgb : np.ndarray  shape (n_train,)
    oof_nn  : np.ndarray  shape (n_train,)
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_xgb = np.zeros(len(y_train))
    oof_nn  = np.zeros(len(y_train))

    for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_trn, X_val = X_train[trn_idx], X_train[val_idx]
        y_trn, y_val = y_train[trn_idx], y_train[val_idx]

        # ── XGBoost ───────────────────────────────────────────────────────────
        neg, pos = int((y_trn == 0).sum()), int((y_trn == 1).sum())
        spw = neg / max(pos, 1)
        xgb = build_xgb(scale_pos_weight=spw)
        xgb.set_params(early_stopping_rounds=50)
        xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = predict_proba_xgb(xgb, X_val)

        # ── Neural Network ────────────────────────────────────────────────────
        nn = build_nn()
        nn.fit(X_trn, y_trn)
        oof_nn[val_idx] = predict_proba_nn(nn, X_val)

        print(f"[stacking] Fold {fold_idx}/{N_FOLDS} complete")

    return oof_xgb, oof_nn


def train_meta_model(
    oof_xgb: np.ndarray,
    oof_nn: np.ndarray,
    y_train: np.ndarray,
    save: bool = True,
    model_name: str = "meta_model",
) -> LogisticRegression:
    """
    Train Logistic Regression meta-model on OOF stacked features.

    Input features : [P_xgb_oof, P_nn_oof]  shape (n_train, 2)
    Target         : y_train
    """
    Stacked_X = np.column_stack([oof_xgb, oof_nn])

    meta = LogisticRegression(**META_LR_PARAMS)
    t0 = time.time()
    meta.fit(Stacked_X, y_train)
    elapsed = time.time() - t0
    print(f"[stacking] Meta-model trained in {elapsed:.1f}s")

    if save:
        out = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(meta, out)
        print(f"[stacking] Saved meta-model to {out}")

    return meta


def stack_predict(
    meta: LogisticRegression,
    p_xgb: np.ndarray,
    p_nn: np.ndarray,
) -> np.ndarray:
    """Generate stacked probability from base model probabilities."""
    Stacked_X = np.column_stack([p_xgb, p_nn])
    return meta.predict_proba(Stacked_X)[:, 1]
