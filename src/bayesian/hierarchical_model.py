"""
Bayesian hierarchical model for AV safety metrics.

Motivation
----------
Safety metrics vary across segments, drivers, and road conditions.  A naive
pooled estimate ignores this variation; a fully unpooled estimate is noisy
for short segments.  A hierarchical (partial-pooling) model finds the optimal
compromise: each segment has its own parameter, but those parameters are
drawn from a shared population distribution.

Model specification (for minimum TTC per frame):

  Population level:
    μ_pop  ~ Normal(μ₀, σ₀²)      # prior on mean TTC
    σ_pop  ~ HalfNormal(τ)        # prior on between-segment SD

  Segment level (j = 1…J):
    μⱼ ~ Normal(μ_pop, σ_pop²)    # partial pooling

  Observation level (i in segment j):
    y_ij ~ Normal(μⱼ, σ_obs²)     # likelihood

This module implements the model using a Metropolis-Hastings MCMC sampler
(pure numpy/scipy — no external probabilistic programming library needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalPosterior:
    mu_pop_samples: np.ndarray          # (n_samples,) population mean
    sigma_pop_samples: np.ndarray       # (n_samples,) population SD
    mu_segment_samples: np.ndarray      # (n_samples, n_segments) segment means
    segment_ids: list[str]
    n_samples: int
    acceptance_rate: float

    def segment_posterior_mean(self) -> pd.Series:
        means = self.mu_segment_samples.mean(axis=0)
        return pd.Series(means, index=self.segment_ids)

    def segment_posterior_std(self) -> pd.Series:
        stds = self.mu_segment_samples.std(axis=0)
        return pd.Series(stds, index=self.segment_ids)

    def population_summary(self) -> dict:
        return {
            "mu_pop_mean": float(self.mu_pop_samples.mean()),
            "mu_pop_std": float(self.mu_pop_samples.std()),
            "mu_pop_95ci": (
                float(np.percentile(self.mu_pop_samples, 2.5)),
                float(np.percentile(self.mu_pop_samples, 97.5)),
            ),
            "sigma_pop_mean": float(self.sigma_pop_samples.mean()),
            "shrinkage_factor": float(
                1 - self.mu_segment_samples.std(axis=0).mean() /
                (self.mu_segment_samples.mean(axis=0).std() + 1e-8)
            ),
        }

    def risk_probability(self, threshold: float) -> pd.Series:
        """P(μⱼ < threshold) for each segment — probability of unsafe mean TTC."""
        probs = (self.mu_segment_samples < threshold).mean(axis=0)
        return pd.Series(probs, index=self.segment_ids, name="risk_probability")


# ---------------------------------------------------------------------------
# Log-posterior computation
# ---------------------------------------------------------------------------

def _log_posterior(
    params: np.ndarray,
    observations: list[np.ndarray],
    n_segments: int,
    mu0: float = 5.0,
    sigma0: float = 5.0,
    tau: float = 2.0,
    sigma_obs: float = 1.0,
) -> float:
    """
    Unnormalised log-posterior for the hierarchical model.

    params layout: [mu_pop, log_sigma_pop, mu_1, ..., mu_J]
    """
    mu_pop = params[0]
    log_sigma_pop = params[1]
    mu_segs = params[2: 2 + n_segments]

    sigma_pop = np.exp(log_sigma_pop)

    # Population-level priors
    lp = stats.norm.logpdf(mu_pop, mu0, sigma0)
    lp += stats.halfnorm.logpdf(sigma_pop, scale=tau) + log_sigma_pop  # Jacobian

    # Segment-level priors (partial pooling)
    lp += stats.norm.logpdf(mu_segs, mu_pop, sigma_pop).sum()

    # Likelihood
    for j, obs in enumerate(observations):
        if len(obs) == 0:
            continue
        lp += stats.norm.logpdf(obs, mu_segs[j], sigma_obs).sum()

    return float(lp)


# ---------------------------------------------------------------------------
# Metropolis-Hastings sampler
# ---------------------------------------------------------------------------

def fit_hierarchical_model(
    safety_df: pd.DataFrame,
    metric_col: str = "min_ttc",
    segment_col: str = "segment_id",
    n_samples: int = 3000,
    warmup: int = 1000,
    step_size: float = 0.15,
    seed: int = 42,
    sigma_obs: float = 1.0,
) -> HierarchicalPosterior:
    """
    Fit the hierarchical Bayesian model via Metropolis-Hastings MCMC.

    Parameters
    ----------
    safety_df   : DataFrame with one row per (segment, frame) observation
    metric_col  : safety metric to model (should be roughly normally distributed)
    segment_col : column identifying the segment
    n_samples   : number of post-warmup MCMC draws
    warmup      : number of warmup (burn-in) draws
    step_size   : Gaussian proposal SD (tune if acceptance rate is outside 20–50%)
    sigma_obs   : assumed observational noise SD (can be estimated separately)
    """
    rng = np.random.default_rng(seed)

    # Prepare data
    df = safety_df[[segment_col, metric_col]].dropna().copy()
    finite_mask = np.isfinite(df[metric_col])
    df = df[finite_mask]

    segments = df[segment_col].unique().tolist()
    J = len(segments)
    observations = [
        df.loc[df[segment_col] == seg, metric_col].values
        for seg in segments
    ]

    # Initialise params: [mu_pop, log_sigma_pop, mu_1..mu_J]
    obs_means = np.array([o.mean() if len(o) > 0 else 0.0 for o in observations])
    params = np.concatenate([[obs_means.mean(), np.log(max(obs_means.std(), 0.1))], obs_means])
    n_params = len(params)

    current_lp = _log_posterior(params, observations, J, sigma_obs=sigma_obs)

    # Storage
    all_params = np.zeros((n_samples + warmup, n_params))
    n_accepted = 0

    for i in range(n_samples + warmup):
        proposal = params + rng.normal(0, step_size, n_params)
        proposal[1] = max(proposal[1], -4.0)  # keep sigma_pop from collapsing

        proposed_lp = _log_posterior(proposal, observations, J, sigma_obs=sigma_obs)
        log_accept = proposed_lp - current_lp

        if np.log(rng.uniform()) < log_accept:
            params = proposal
            current_lp = proposed_lp
            if i >= warmup:
                n_accepted += 1

        all_params[i] = params

    post_warmup = all_params[warmup:]
    acceptance_rate = n_accepted / n_samples

    return HierarchicalPosterior(
        mu_pop_samples=post_warmup[:, 0],
        sigma_pop_samples=np.exp(post_warmup[:, 1]),
        mu_segment_samples=post_warmup[:, 2: 2 + J],
        segment_ids=segments,
        n_samples=n_samples,
        acceptance_rate=acceptance_rate,
    )


# ---------------------------------------------------------------------------
# Posterior predictive checks
# ---------------------------------------------------------------------------

def posterior_predictive_check(
    posterior: HierarchicalPosterior,
    observations: list[np.ndarray],
    n_ppc_draws: int = 200,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Generate posterior predictive samples to check model fit.

    Returns dict with 'ppc_means' and 'ppc_stds' for each segment.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(posterior.n_samples, size=n_ppc_draws, replace=False)

    ppc_means = posterior.mu_segment_samples[idx].mean(axis=0)
    ppc_stds = posterior.mu_segment_samples[idx].std(axis=0)

    observed_means = np.array([o.mean() if len(o) > 0 else np.nan for o in observations])

    return {
        "ppc_means": ppc_means,
        "ppc_stds": ppc_stds,
        "observed_means": observed_means,
        "coverage_80": float(np.mean(
            (observed_means >= posterior.mu_segment_samples.mean(axis=0) - 1.28 * ppc_stds) &
            (observed_means <= posterior.mu_segment_samples.mean(axis=0) + 1.28 * ppc_stds)
        )),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_hierarchical_posteriors(posterior: HierarchicalPosterior):
    """Plot population-level and per-segment posterior distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_seg = len(posterior.segment_ids)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Population mean posterior
    ax = axes[0]
    ax.hist(posterior.mu_pop_samples, bins=60, color="steelblue",
            edgecolor="white", density=True, alpha=0.8)
    lo, hi = np.percentile(posterior.mu_pop_samples, [2.5, 97.5])
    ax.axvline(posterior.mu_pop_samples.mean(), color="red", lw=2, label="Posterior mean")
    ax.axvspan(lo, hi, alpha=0.2, color="red", label="95% CI")
    ax.set_title("Population Mean TTC — Posterior")
    ax.set_xlabel("μ_pop (s)")
    ax.legend()

    # Per-segment shrinkage plot
    ax2 = axes[1]
    seg_means = posterior.segment_posterior_mean()
    seg_stds = posterior.segment_posterior_std()
    y_pos = np.arange(len(posterior.segment_ids))
    ax2.errorbar(
        seg_means.values, y_pos,
        xerr=1.96 * seg_stds.values,
        fmt="o", color="steelblue", capsize=4, markersize=6,
    )
    ax2.axvline(posterior.mu_pop_samples.mean(), color="red",
                linestyle="--", label="Population mean")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([s[:20] for s in posterior.segment_ids], fontsize=8)
    ax2.set_xlabel("Posterior Mean TTC (s)")
    ax2.set_title("Per-Segment Posteriors (Partial Pooling)")
    ax2.legend()

    fig.tight_layout()
    return fig
