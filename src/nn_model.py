"""
nn_model.py
===========
Neural Network (MLP) base model for churn prediction.

Architecture : Dense(256) → Dense(128) → Dense(64) → sigmoid output
Regularisation : L2 weight decay (alpha) + early stopping on validation loss
"""
from __future__ import annotations

import logging
import time

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from config import MODELS_DIR, NN_PARAMS

logger = logging.getLogger(__name__)


def build_nn() -> MLPClassifier:
    """Construct (but not fit) the MLPClassifier with project defaults."""
    return MLPClassifier(**NN_PARAMS)


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    save: bool = True,
    model_name: str = "nn_model",
) -> MLPClassifier:
    """
    Train MLPClassifier on (X_train, y_train).

    Parameters
    ----------
    X_train, y_train : training data
    save             : persist model to disk when True
    model_name       : filename prefix (without .joblib)
    """
    model = build_nn()

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    logger.info(
        "Neural Network trained in %.1fs | n_iter=%d | loss=%.4f",
        elapsed, model.n_iter_, model.loss_,
    )

    if save:
        out = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, out)
        logger.info("Neural Network saved → %s", out)

    return model


def predict_proba_nn(model: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """Return P(churn=1) for each row in X."""
    return model.predict_proba(X)[:, 1]
