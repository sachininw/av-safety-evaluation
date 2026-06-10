"""
Conformal Prediction for AV safety metric bounds.

Standard EVT gives parametric tail bounds that rely on distributional
assumptions.  Conformal prediction provides *distribution-free* coverage
guarantees: with probability ≥ 1 - α, the true value will fall inside
the prediction set — with no assumptions on the data distribution.

This is directly applicable to deployment readiness decisions:
  "With 99% marginal coverage, the minimum TTC in the next segment
   will not fall below X seconds."

Three variants are implemented:
  1. Split Conformal Prediction — simple, valid marginal coverage guarantee.
  2. Mondrian (conditional) Conformal — separate calibration per stratum
     (e.g., urban vs. highway, by object density).
  3. Conformal Risk Control (CRC) — controls a user-defined risk function
     (e.g., expected near-miss rate) at a specified level, per Angelopoulos
     et al. (2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConformalInterval:
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    n_calibration: int
    quantile: float
    method: str

    @property
    def coverage_target(self) -> float:
        return 1.0 - self.alpha

    def empirical_coverage(self, y_true: np.ndarray) -> float:
        return float(np.mean((y_true >= self.lower) & (y_true <= self.upper)))

    def width(self) -> np.ndarray:
        return self.upper - self.lower


# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------

def absolute_residual_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """|y - ŷ| — standard nonconformity score for regression."""
    return np.abs(y_true - y_pred)


def scaled_residual_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """(|y - ŷ| / σ) — normalised score; σ is a local variance estimate."""
    return np.abs(y_true - y_pred) / (sigma + 1e-8)


# ---------------------------------------------------------------------------
# Split Conformal Prediction
# ---------------------------------------------------------------------------

class SplitConformalPredictor:
    """
    Inductive (split) conformal predictor for a regression target.

    Steps:
      1. Fit a base model on a proper training set.
      2. Compute nonconformity scores on a held-out calibration set.
      3. At test time, add q̂ (the (1−α)(1+1/n) quantile of calibration scores)
         symmetrically around the point prediction.

    Finite-sample marginal coverage guarantee:
      P(Y_test ∈ Ĉ(X_test)) ≥ 1 − α
    """

    def __init__(
        self,
        base_model=None,
        alpha: float = 0.05,
        score_fn: Callable = absolute_residual_score,
    ):
        self.base_model = base_model or Ridge(alpha=1.0)
        self.alpha = alpha
        self.score_fn = score_fn
        self._scaler = StandardScaler()
        self._q_hat: Optional[float] = None
        self._calibration_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "SplitConformalPredictor":
        X_train_s = self._scaler.fit_transform(X_train)
        self.base_model.fit(X_train_s, y_train)

        X_cal_s = self._scaler.transform(X_cal)
        y_cal_pred = self.base_model.predict(X_cal_s)
        scores = self.score_fn(y_cal, y_cal_pred)
        self._calibration_scores = scores

        n = len(scores)
        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        return self

    def predict(self, X_test: np.ndarray) -> ConformalInterval:
        if self._q_hat is None:
            raise RuntimeError("Call fit() before predict().")
        X_s = self._scaler.transform(X_test)
        y_pred = self.base_model.predict(X_s)
        return ConformalInterval(
            lower=y_pred - self._q_hat,
            upper=y_pred + self._q_hat,
            alpha=self.alpha,
            n_calibration=len(self._calibration_scores),
            quantile=self._q_hat,
            method="split_conformal",
        )

    def calibration_score_distribution(self) -> np.ndarray:
        return self._calibration_scores


# ---------------------------------------------------------------------------
# Mondrian (conditional) Conformal Prediction
# ---------------------------------------------------------------------------

class MondrianConformalPredictor:
    """
    Conditional conformal predictor with separate calibration per stratum.

    Provides approximate *conditional* coverage P(Y ∈ Ĉ | stratum) ≥ 1−α
    by computing a stratum-specific quantile.

    Useful when safety margins differ by scenario type:
      - Urban (high density, low speed)
      - Highway (low density, high speed)
      - Intersection
    """

    def __init__(self, base_model=None, alpha: float = 0.05):
        self.base_model = base_model or Ridge(alpha=1.0)
        self.alpha = alpha
        self._scaler = StandardScaler()
        self._stratum_quantiles: dict[str, float] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        strata_cal: np.ndarray,
    ) -> "MondrianConformalPredictor":
        X_train_s = self._scaler.fit_transform(X_train)
        self.base_model.fit(X_train_s, y_train)

        X_cal_s = self._scaler.transform(X_cal)
        y_cal_pred = self.base_model.predict(X_cal_s)
        scores = np.abs(y_cal - y_cal_pred)

        for stratum in np.unique(strata_cal):
            mask = strata_cal == stratum
            s = scores[mask]
            n = len(s)
            level = min(np.ceil((1 - self.alpha) * (n + 1)) / n, 1.0)
            self._stratum_quantiles[str(stratum)] = float(np.quantile(s, level))

        return self

    def predict(
        self,
        X_test: np.ndarray,
        strata_test: np.ndarray,
    ) -> ConformalInterval:
        X_s = self._scaler.transform(X_test)
        y_pred = self.base_model.predict(X_s)
        q_hats = np.array([
            self._stratum_quantiles.get(str(s), max(self._stratum_quantiles.values()))
            for s in strata_test
        ])
        return ConformalInterval(
            lower=y_pred - q_hats,
            upper=y_pred + q_hats,
            alpha=self.alpha,
            n_calibration=len(self._stratum_quantiles),
            quantile=float(q_hats.mean()),
            method="mondrian_conformal",
        )


# ---------------------------------------------------------------------------
# Conformal Risk Control (CRC)
# ---------------------------------------------------------------------------

def conformal_risk_control(
    risk_fn: Callable[[np.ndarray, float], float],
    calibration_labels: np.ndarray,
    lambda_grid: np.ndarray,
    delta: float = 0.05,
    alpha: float = 0.1,
) -> float:
    """
    Find the smallest threshold λ such that the expected risk is ≤ α.

    Based on Angelopoulos et al. (2022) "Conformal Risk Control".
    The guarantee is:
      E[risk(ŷ(λ*), y)] ≤ α   with probability ≥ 1 − δ

    Parameters
    ----------
    risk_fn         : function(labels, lambda) → risk value in [0, 1]
    calibration_labels : array of calibration ground-truth labels
    lambda_grid     : candidate threshold values (sorted ascending)
    delta           : confidence level (default 0.05)
    alpha           : target risk level

    Returns the selected λ*.
    """
    n = len(calibration_labels)
    # Bonferroni correction for multiple testing over the grid
    alpha_corrected = alpha - np.sqrt(np.log(len(lambda_grid) / delta) / (2 * n))
    alpha_corrected = max(alpha_corrected, 0.0)

    for lam in sorted(lambda_grid):
        r = risk_fn(calibration_labels, lam)
        if r <= alpha_corrected:
            return float(lam)
    return float(lambda_grid[-1])


# ---------------------------------------------------------------------------
# Coverage plots
# ---------------------------------------------------------------------------

def coverage_width_tradeoff(
    predictor: SplitConformalPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluate empirical coverage and average interval width across alpha values.
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.01, 0.20, 20)

    original_alpha = predictor.alpha
    records = []
    for alpha in alpha_grid:
        predictor.alpha = alpha
        n = len(predictor._calibration_scores)
        level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        q = float(np.quantile(predictor._calibration_scores, level))
        predictor._q_hat = q

        interval = predictor.predict(X_test)
        records.append({
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "empirical_coverage": interval.empirical_coverage(y_test),
            "mean_width": float(interval.width().mean()),
            "q_hat": q,
        })

    predictor.alpha = original_alpha
    return pd.DataFrame(records)
