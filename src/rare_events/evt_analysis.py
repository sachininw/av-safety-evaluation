"""
Rare-event estimation using Extreme Value Theory (EVT).

Autonomous vehicle safety evaluation requires estimating the probability of
rare but critical events (near-misses, hard braking) from limited observations.
EVT provides a principled statistical framework for extrapolating into the
tail of a distribution far beyond observed data.

Two approaches are implemented:
  1. Block Maxima / GEV fitting — fits a Generalised Extreme Value distribution
     to block maxima (e.g., minimum TTC per segment).
  2. Peaks-Over-Threshold (POT) / GPD fitting — fits a Generalised Pareto
     Distribution to exceedances above a high threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# GEV (Block Maxima) approach
# ---------------------------------------------------------------------------

@dataclass
class GEVFit:
    xi: float      # shape parameter (tail index)
    mu: float      # location parameter
    sigma: float   # scale parameter
    log_likelihood: float
    aic: float
    bic: float
    n_blocks: int

    def return_level(self, return_period: float) -> float:
        """
        Compute the T-period return level.

        For minima (e.g., minimum TTC), we negate to convert to maxima,
        fit GEV, then negate back. Caller is responsible for this transform.
        """
        p = 1.0 - 1.0 / return_period
        if abs(self.xi) < 1e-8:
            return self.mu - self.sigma * np.log(-np.log(p))
        return self.mu + self.sigma * ((-np.log(p)) ** (-self.xi) - 1) / self.xi

    def return_level_ci(
        self,
        return_period: float,
        alpha: float = 0.95,
        n_bootstrap: int = 500,
        block_data: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for a return level."""
        if block_data is None:
            return (np.nan, np.nan)
        rng = np.random.default_rng(42)
        levels = []
        for _ in range(n_bootstrap):
            sample = rng.choice(block_data, size=len(block_data), replace=True)
            try:
                fit = fit_gev(sample)
                levels.append(fit.return_level(return_period))
            except Exception:
                continue
        if not levels:
            return (np.nan, np.nan)
        lower = float(np.percentile(levels, (1 - alpha) / 2 * 100))
        upper = float(np.percentile(levels, (1 + alpha) / 2 * 100))
        return lower, upper


def fit_gev(block_maxima: np.ndarray) -> GEVFit:
    """Fit a GEV distribution to an array of block maxima using MLE."""
    block_maxima = np.asarray(block_maxima, dtype=float)
    block_maxima = block_maxima[np.isfinite(block_maxima)]

    if len(block_maxima) < 5:
        raise ValueError(f"Need at least 5 block maxima, got {len(block_maxima)}")

    # scipy.stats.genextreme: shape=-xi in scipy convention
    xi_neg, loc, scale = stats.genextreme.fit(block_maxima)
    xi = -xi_neg

    log_lik = float(stats.genextreme.logpdf(block_maxima, xi_neg, loc, scale).sum())
    k = 3
    n = len(block_maxima)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik

    return GEVFit(
        xi=xi,
        mu=float(loc),
        sigma=float(scale),
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
        n_blocks=n,
    )


def extract_block_minima(
    ttc_series: pd.Series,
    block_size: int = 200,
) -> np.ndarray:
    """
    Split a TTC time-series into non-overlapping blocks and take the minimum.

    We model the distribution of minimum TTC as the safety-critical quantity.
    Finite values only — frames with no agent interaction are excluded.
    """
    finite = ttc_series[np.isfinite(ttc_series)].values
    n_blocks = len(finite) // block_size
    if n_blocks < 5:
        raise ValueError(
            f"Not enough data for block maxima: {n_blocks} blocks of size {block_size}. "
            "Try a smaller block_size or more data."
        )
    blocks = finite[: n_blocks * block_size].reshape(n_blocks, block_size)
    return blocks.min(axis=1)


# ---------------------------------------------------------------------------
# GPD (Peaks-Over-Threshold) approach
# ---------------------------------------------------------------------------

@dataclass
class GPDFit:
    xi: float      # shape parameter
    sigma: float   # scale parameter
    threshold: float
    n_exceedances: int
    n_total: int
    log_likelihood: float
    aic: float
    bic: float

    @property
    def exceedance_probability(self) -> float:
        return self.n_exceedances / self.n_total

    def tail_probability(self, level: float) -> float:
        """P(X > level) for level > threshold."""
        excess = level - self.threshold
        if excess < 0:
            raise ValueError("level must exceed the threshold")
        if abs(self.xi) < 1e-8:
            return self.exceedance_probability * np.exp(-excess / self.sigma)
        return self.exceedance_probability * (
            1 + self.xi * excess / self.sigma
        ) ** (-1 / self.xi)

    def return_level(self, return_period: float, n_obs_per_year: float = 1.0) -> float:
        """Return level for a given return period (in years)."""
        p_exceed = 1.0 / (return_period * n_obs_per_year)
        p_excess = p_exceed / self.exceedance_probability
        if abs(self.xi) < 1e-8:
            return self.threshold - self.sigma * np.log(p_excess)
        return self.threshold + self.sigma * ((p_excess ** (-self.xi) - 1) / self.xi)


def select_threshold_mean_excess(
    data: np.ndarray,
    quantile_range: tuple[float, float] = (0.85, 0.99),
    n_points: int = 30,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Mean Excess Plot: returns (thresholds, mean_excesses, suggested_threshold).

    The GPD is appropriate above a threshold where mean excess is approximately
    linear in the threshold.  We suggest the smallest threshold where the
    residual coefficient of variation stabilises.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    thresholds = np.quantile(data, np.linspace(*quantile_range, n_points))
    mean_excesses = np.array([
        (data[data > u] - u).mean() for u in thresholds
    ])
    # Suggest threshold at the elbow: where the gradient first becomes roughly constant
    grads = np.abs(np.gradient(mean_excesses))
    suggested_idx = max(0, int(np.argmin(grads[: n_points // 2])))
    return thresholds, mean_excesses, float(thresholds[suggested_idx])


def fit_gpd(data: np.ndarray, threshold: Optional[float] = None) -> GPDFit:
    """Fit a Generalised Pareto Distribution above a threshold using MLE."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if threshold is None:
        threshold = float(np.quantile(data, 0.90))

    exceedances = data[data > threshold] - threshold
    n_exceed = len(exceedances)
    if n_exceed < 10:
        raise ValueError(f"Too few exceedances ({n_exceed}) above threshold {threshold:.3f}")

    xi_neg, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    xi = -xi_neg

    log_lik = float(stats.genpareto.logpdf(exceedances, xi_neg, loc, scale).sum())
    k = 2
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n_exceed) - 2 * log_lik

    return GPDFit(
        xi=xi,
        sigma=float(scale),
        threshold=float(threshold),
        n_exceedances=n_exceed,
        n_total=len(data),
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
    )


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------

def rare_event_summary(
    ttc_series: pd.Series,
    return_periods: list[float] = [10, 100, 1000],
    block_size: int = 200,
) -> pd.DataFrame:
    """
    Produce a summary table of estimated rare-event TTC levels.

    For safety, we work with *negative* TTC (so small TTC → large negative
    value → a GEV maximum over negatives gives worst-case minima).
    """
    finite = ttc_series[np.isfinite(ttc_series)].values

    rows = []

    # GEV block-minima approach (negate to convert minima to maxima)
    try:
        block_minima = extract_block_minima(ttc_series, block_size=block_size)
        gev = fit_gev(-block_minima)  # negate: minima become maxima
        for rp in return_periods:
            level = -gev.return_level(rp)  # negate back to TTC
            rows.append({"method": "GEV", "return_period": rp, "ttc_return_level_s": level})
    except Exception as e:
        rows.append({"method": "GEV", "return_period": np.nan, "ttc_return_level_s": np.nan, "error": str(e)})

    # GPD approach on the lower tail of TTC (low TTC = dangerous)
    # Reflect the low tail: model (u - TTC) for TTC < u, where u is 15th percentile
    try:
        pos_finite = finite[finite > 0.05]  # exclude zero-TTC (agents already in collision box)
        if len(pos_finite) < 20:
            raise ValueError(f"Too few positive TTC values ({len(pos_finite)})")
        # Reciprocal: high 1/TTC = low TTC = dangerous; exclude near-zero TTC first
        inv_ttc = 1.0 / pos_finite
        gpd_threshold = float(np.quantile(inv_ttc, 0.85))
        gpd = fit_gpd(inv_ttc, threshold=gpd_threshold)
        for rp in return_periods:
            inv_level = gpd.return_level(rp, n_obs_per_year=len(pos_finite))
            level = 1.0 / inv_level if inv_level > 0 else 0.0
            rows.append({"method": "GPD", "return_period": rp, "ttc_return_level_s": level})
    except Exception as e:
        rows.append({"method": "GPD", "return_period": np.nan, "ttc_return_level_s": np.nan, "error": str(e)})

    return pd.DataFrame(rows)
