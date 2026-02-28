"""
calibration.py
==============
Probability calibration for the stacked ensemble.

Fits an Isotonic Regression or Platt Scaling (logistic) calibrator on
held-out (test-set) probabilities to produce reliable probability estimates.

Why calibrate?
  A model might output P=0.80 for a group of customers, but only 60% of
  them actually churn.  Calibration corrects that systematic bias, making
  the probabilities suitable as business-facing risk scores.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config import CALIBRATION_METHOD

logger = logging.getLogger(__name__)

# Type alias for the calibrated probability function
ProbFn = Callable[[np.ndarray], np.ndarray]


def calibrate_probabilities(
    X_cal:   np.ndarray,
    y_cal:   np.ndarray,
    prob_fn: ProbFn,
    method:  str = CALIBRATION_METHOD,
) -> ProbFn:
    """
    Fit a calibrator on held-out stacked probabilities.

    Parameters
    ----------
    X_cal    : shape (n, 2) — [P_xgb, P_nn] on the held-out (test) set
    y_cal    : shape (n,)   — true binary labels
    prob_fn  : callable X → ndarray — raw stacked probability function
    method   : "isotonic" (default) or "sigmoid" (Platt Scaling)

    Returns
    -------
    calibrated_fn : callable that maps new [P_xgb, P_nn] → calibrated probs
    """
    raw_probs = prob_fn(X_cal)  # shape (n,)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_cal)

        def calibrated_fn(X: np.ndarray) -> np.ndarray:
            return np.asarray(calibrator.predict(prob_fn(X))).clip(0.0, 1.0)

    elif method == "sigmoid":
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(raw_probs.reshape(-1, 1), y_cal)

        def calibrated_fn(X: np.ndarray) -> np.ndarray:
            return calibrator.predict_proba(prob_fn(X).reshape(-1, 1))[:, 1]

    else:
        raise ValueError(f"Unknown calibration method '{method}'. Use 'isotonic' or 'sigmoid'.")

    calibrated_fn._calibrator = calibrator  # type: ignore[attr-defined]
    logger.info("Probability calibration complete | method='%s'", method)
    return calibrated_fn
