"""
nn_model.py — Neural Network (MLP) base model for churn prediction.

Architecture: Dense(256) → Dense(128) → Dense(64) → output
Uses early stopping and L2 regularization (alpha).
"""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from config import MODELS_DIR, NN_PARAMS


def build_nn() -> MLPClassifier:
    """Construct (but not fit) the MLPClassifier."""
    return MLPClassifier(**NN_PARAMS)


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save: bool = True,
    model_name: str = "nn_model",
) -> MLPClassifier:
    """Train MLPClassifier on (X_train, y_train)."""
    model = build_nn()

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(
        f"[nn_model] Trained in {elapsed:.1f}s | "
        f"n_iter={model.n_iter_} | loss={model.loss_:.4f}"
    )

    if save:
        out = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, out)
        print(f"[nn_model] Saved to {out}")

    return model


def predict_proba_nn(model: MLPClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
