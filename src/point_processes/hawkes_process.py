"""
Hawkes Process for modelling rare-event clustering in AV data.

Observation: safety-critical events (near-misses, hard braking) tend to
cluster in time — a near-miss in heavy traffic is followed by more.
A Poisson process cannot capture this self-exciting behaviour; a Hawkes
process can.

The Hawkes process has intensity:
  λ(t) = μ + Σ_{tᵢ < t} α · exp(−β · (t − tᵢ))

where:
  μ    = background (exogenous) event rate
  α    = excitation amplitude — how much each event raises the rate
  β    = decay rate — how quickly the excitation fades
  α/β  = branching ratio — fraction of events triggered by past events
         (must be < 1 for stationarity)

The model is fitted by maximum log-likelihood using scipy.optimize.

Applications:
  - Estimate the background danger rate on a route
  - Quantify contagion: do near-misses trigger more near-misses?
  - Predict elevated-risk periods following an event
  - Compare urban vs. highway event clustering
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


# ---------------------------------------------------------------------------
# Model parameters and intensity
# ---------------------------------------------------------------------------

@dataclass
class HawkesParams:
    mu: float     # background rate (events/second)
    alpha: float  # excitation amplitude
    beta: float   # decay rate (1/second)

    @property
    def branching_ratio(self) -> float:
        """α/β — fraction of events triggered by prior events (must be < 1)."""
        return self.alpha / (self.beta + 1e-12)

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    @property
    def unconditional_rate(self) -> float:
        """E[λ(t)] = μ / (1 − α/β) in the stationary regime."""
        br = self.branching_ratio
        if br >= 1.0:
            return np.inf
        return self.mu / (1.0 - br)

    def __str__(self) -> str:
        return (
            f"HawkesParams(μ={self.mu:.4f}, α={self.alpha:.4f}, β={self.beta:.4f}) "
            f"| branching={self.branching_ratio:.4f} "
            f"| stationary={self.is_stationary}"
        )


def hawkes_intensity(
    t: float,
    event_times: np.ndarray,
    params: HawkesParams,
) -> float:
    """Evaluate the conditional intensity λ(t | history) at time t."""
    past = event_times[event_times < t]
    return params.mu + params.alpha * np.sum(np.exp(-params.beta * (t - past)))


def intensity_series(
    t_grid: np.ndarray,
    event_times: np.ndarray,
    params: HawkesParams,
) -> np.ndarray:
    """Vectorised intensity over a time grid."""
    return np.array([hawkes_intensity(t, event_times, params) for t in t_grid])


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def hawkes_log_likelihood(
    params_vec: np.ndarray,
    event_times: np.ndarray,
    T: float,
) -> float:
    """
    Negative log-likelihood of a Hawkes process with exponential kernel.

    Uses the recursive formula for efficiency:
      R(i) = exp(−β(tᵢ − tᵢ₋₁)) · (1 + R(i−1))

    Log-likelihood = Σᵢ log(μ + α·R(i)) − μ·T − α/β · Σᵢ (1 − exp(−β(T−tᵢ)))
    """
    mu, alpha, beta = params_vec
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10

    n = len(event_times)
    R = np.zeros(n)
    for i in range(1, n):
        R[i] = np.exp(-beta * (event_times[i] - event_times[i - 1])) * (1 + R[i - 1])

    intensities = mu + alpha * R
    if np.any(intensities <= 0):
        return 1e10

    log_lik = (
        np.sum(np.log(intensities))
        - mu * T
        - (alpha / beta) * np.sum(1 - np.exp(-beta * (T - event_times)))
    )
    return float(-log_lik)


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------

def fit_hawkes(
    event_times: np.ndarray,
    T: Optional[float] = None,
    n_restarts: int = 10,
    seed: int = 42,
) -> tuple[HawkesParams, float]:
    """
    Fit a Hawkes process to event times by maximum likelihood.

    Parameters
    ----------
    event_times : sorted 1-D array of event timestamps (seconds)
    T           : observation window end time (default: max event time)
    n_restarts  : number of random restarts to avoid local optima

    Returns (best_params, log_likelihood).
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    event_times = event_times - event_times[0]  # normalize to start at 0
    if T is None:
        T = float(event_times.max())

    if len(event_times) < 5:
        raise ValueError(f"Need at least 5 events, got {len(event_times)}")

    rng = np.random.default_rng(seed)
    best_nll = np.inf
    best_params = None

    rate_est = len(event_times) / T
    mu_lo = max(rate_est * 0.001, 1e-12)
    mu_hi = max(rate_est * 10.0, mu_lo * 100)

    for _ in range(n_restarts):
        # Adaptive initial bounds based on observed event rate
        mu0 = rng.uniform(mu_lo, mu_hi)
        alpha0 = rng.uniform(0.01, 0.8)
        beta_lo = alpha0 + 0.01
        beta_hi = max(beta_lo * 10, rate_est * 20, 1e-8)
        beta0 = rng.uniform(beta_lo, beta_hi)

        result = minimize(
            hawkes_log_likelihood,
            x0=[mu0, alpha0, beta0],
            args=(event_times, T),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
        )

        if result.fun < best_nll and result.success:
            best_nll = result.fun
            best_params = result.x

    if best_params is None:
        raise RuntimeError("Optimisation failed across all restarts.")

    mu, alpha, beta = best_params
    return (
        HawkesParams(mu=float(mu), alpha=float(alpha), beta=float(beta)),
        float(-best_nll),
    )


# ---------------------------------------------------------------------------
# Goodness-of-fit: time-rescaling test
# ---------------------------------------------------------------------------

def time_rescaling_test(
    event_times: np.ndarray,
    params: HawkesParams,
    T: Optional[float] = None,
) -> tuple[float, float]:
    """
    Test goodness of fit via the time-rescaling theorem.

    If the Hawkes process is correctly specified, the rescaled inter-arrival
    times Λ(tᵢ) − Λ(tᵢ₋₁) should be i.i.d. Exp(1).

    Returns (KS statistic, p-value).
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    T = T or float(event_times.max())

    # Compute integrated intensity between consecutive events
    R = np.zeros(len(event_times))
    for i in range(1, len(event_times)):
        R[i] = np.exp(-params.beta * (event_times[i] - event_times[i - 1])) * (1 + R[i - 1])

    delta_t = np.diff(event_times)
    # Approximate integrated intensity in each interval
    lambda_integral = (
        params.mu * delta_t
        + (params.alpha / params.beta) * R[:-1] * (1 - np.exp(-params.beta * delta_t))
    )
    lambda_integral = lambda_integral[lambda_integral > 0]

    ks_stat, p_val = stats.kstest(lambda_integral, "expon")
    return float(ks_stat), float(p_val)


# ---------------------------------------------------------------------------
# Multi-segment analysis
# ---------------------------------------------------------------------------

def fit_hawkes_by_group(
    events_df: pd.DataFrame,
    group_col: str = "segment_id",
    event_col: str = "timestamp_s",
    min_events: int = 10,
) -> pd.DataFrame:
    """
    Fit a Hawkes process per group and return a comparison DataFrame.
    """
    records = []
    for grp, sub in events_df.groupby(group_col):
        times = sub[event_col].sort_values().values.astype(float)
        if len(times) < min_events:
            continue
        T = float(times.max() - times.min())
        times = times - times.min()
        try:
            params, log_lik = fit_hawkes(times, T=T)
            ks_stat, ks_p = time_rescaling_test(times, params, T=T)
            records.append({
                "group": str(grp),
                "n_events": len(times),
                "mu": params.mu,
                "alpha": params.alpha,
                "beta": params.beta,
                "branching_ratio": params.branching_ratio,
                "unconditional_rate": params.unconditional_rate,
                "log_likelihood": log_lik,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "good_fit": ks_p > 0.05,
            })
        except Exception as e:
            records.append({"group": str(grp), "n_events": len(times), "error": str(e)})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Simulation (for testing and scenario generation)
# ---------------------------------------------------------------------------

def simulate_hawkes(
    params: HawkesParams,
    T: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate a Hawkes process via Ogata's thinning algorithm.

    Returns sorted array of event times in [0, T].
    """
    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    lambda_bar = params.mu  # initial upper bound

    while t < T:
        dt = rng.exponential(1.0 / lambda_bar)
        t += dt
        if t >= T:
            break

        # Compute actual intensity
        lam = params.mu + params.alpha * sum(
            np.exp(-params.beta * (t - ti)) for ti in events
        )
        # Accept with probability lam / lambda_bar
        if rng.uniform() < lam / lambda_bar:
            events.append(t)
            lambda_bar = lam + params.alpha  # update upper bound after event

    return np.array(events)
