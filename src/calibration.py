# -*- coding: utf-8 -*-
"""
calibration.py -- Probability calibration for the stacked ensemble.

Fits an Isotonic Regression or Platt Scaling (logistic) calibrator on
held-out (test-set) probabilities to produce reliable probability estimates.

Why calibrate?
  A model might output 0.8 for a customer but only 60% of such customers
  actually churn.  Calibration corrects that systematic bias.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config import CALIBRATION_METHOD


def calibrate_probabilities(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    prob_fn,
    method: str = CALIBRATION_METHOD,
):
    """
    Fit a calibrator on held-out stacked probabilities and return a
    callable that maps new stacked probabilities to calibrated ones.

    Parameters
    ----------
    X_cal    : np.ndarray, shape (n, 2)  -- [P_xgb, P_nn] on the held-out set
    y_cal    : np.ndarray, shape (n,)    -- true binary labels
    prob_fn  : callable(X) -> ndarray   -- raw stacked prob function
    method   : "isotonic" or "sigmoid"

    Returns
    -------
    calibrated_fn : callable(X) -> np.ndarray of calibrated probabilities
    """
    # Get raw stacked probs on the calibration (test) set
    raw_probs = prob_fn(X_cal)   # shape (n,)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_cal)

        def calibrated_fn(X):
            p = prob_fn(X)
            return np.array(calibrator.predict(p)).clip(0.0, 1.0)

    else:   # sigmoid / Platt Scaling
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(raw_probs.reshape(-1, 1), y_cal)

        def calibrated_fn(X):
            p = prob_fn(X)
            return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]

    calibrated_fn._calibrator = calibrator   # type: ignore[attr-defined]
    print(f"[calibration] Calibrated using method='{method}'")
    return calibrated_fn
