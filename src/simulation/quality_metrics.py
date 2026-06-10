"""
Simulation quality assessment: quantifies how well a driving simulator
reproduces the statistical properties of real Waymo on-road data.

A high-fidelity simulator is essential for AV safety evaluation — scenarios
that are too rare in the real world can be generated at scale in simulation.
This module measures the gap between simulated and real distributions across
speed, acceleration, proximity, and agent density features.

Metrics implemented:
  - Kolmogorov-Smirnov (KS) test — marginal distribution comparison
  - Jensen-Shannon Divergence (JSD) — symmetric, bounded divergence [0, 1]
  - Wasserstein-1 distance — Earth-mover's distance (interpretable units)
  - Maximum Mean Discrepancy (MMD) — kernel-based joint distribution test
  - Coverage / Diversity score — checks that rare scenarios are represented
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist


COMPARISON_FEATURES = [
    "speed",
    "acceleration",
    "jerk",
    "heading_rate",
    "nearest_agent_dist",
]


@dataclass
class FeatureComparisonResult:
    feature: str
    ks_statistic: float
    ks_pvalue: float
    jsd: float
    wasserstein: float
    real_mean: float
    sim_mean: float
    real_std: float
    sim_std: float

    @property
    def mean_bias(self) -> float:
        return self.sim_mean - self.real_mean

    @property
    def std_ratio(self) -> float:
        return self.sim_std / self.real_std if self.real_std > 0 else np.nan

    @property
    def distribution_match_score(self) -> float:
        """Heuristic score in [0, 1]: higher is better match."""
        return float(1.0 - self.jsd)


@dataclass
class SimQualityReport:
    feature_results: list[FeatureComparisonResult]
    mmd: float
    coverage_score: float
    overall_score: float

    def summary_df(self) -> pd.DataFrame:
        rows = [
            {
                "feature": r.feature,
                "ks_statistic": r.ks_statistic,
                "ks_pvalue": r.ks_pvalue,
                "jsd": r.jsd,
                "wasserstein_dist": r.wasserstein,
                "mean_bias": r.mean_bias,
                "std_ratio": r.std_ratio,
                "match_score": r.distribution_match_score,
            }
            for r in self.feature_results
        ]
        df = pd.DataFrame(rows)
        df.loc[len(df)] = {
            "feature": "OVERALL",
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "jsd": np.nan,
            "wasserstein_dist": np.nan,
            "mean_bias": np.nan,
            "std_ratio": np.nan,
            "match_score": self.overall_score,
        }
        return df


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def _clean(series: pd.Series, max_n: int = 50_000) -> np.ndarray:
    arr = series.replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)
    if len(arr) > max_n:
        rng = np.random.default_rng(0)
        arr = rng.choice(arr, size=max_n, replace=False)
    return arr


def ks_test(real: np.ndarray, sim: np.ndarray) -> tuple[float, float]:
    result = stats.ks_2samp(real, sim)
    return float(result.statistic), float(result.pvalue)


def jensen_shannon_divergence(
    real: np.ndarray,
    sim: np.ndarray,
    n_bins: int = 50,
) -> float:
    """Compute JSD using equal-width histograms over the combined range."""
    lo = min(real.min(), sim.min())
    hi = max(real.max(), sim.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(real, bins=bins, density=True)
    q, _ = np.histogram(sim, bins=bins, density=True)
    # Smooth to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.clip(jsd, 0, 1))


def wasserstein_distance(real: np.ndarray, sim: np.ndarray) -> float:
    return float(stats.wasserstein_distance(real, sim))


def maximum_mean_discrepancy(
    real: np.ndarray,
    sim: np.ndarray,
    gamma: Optional[float] = None,
    max_n: int = 2000,
) -> float:
    """
    Compute unbiased MMD² with an RBF kernel between two multivariate samples.

    real / sim : shape (n, d) arrays
    gamma      : RBF bandwidth; defaults to the median heuristic
    """
    if real.ndim == 1:
        real = real.reshape(-1, 1)
    if sim.ndim == 1:
        sim = sim.reshape(-1, 1)

    # Subsample for performance
    rng = np.random.default_rng(0)
    if len(real) > max_n:
        real = real[rng.choice(len(real), max_n, replace=False)]
    if len(sim) > max_n:
        sim = sim[rng.choice(len(sim), max_n, replace=False)]

    if gamma is None:
        combined = np.vstack([real, sim])
        pairwise = cdist(combined, combined, "sqeuclidean")
        gamma = 1.0 / (2.0 * np.median(pairwise[pairwise > 0]) + 1e-8)

    def rbf_kernel(X, Y):
        return np.exp(-gamma * cdist(X, Y, "sqeuclidean"))

    Kxx = rbf_kernel(real, real)
    Kyy = rbf_kernel(sim, sim)
    Kxy = rbf_kernel(real, sim)

    n, m = len(real), len(sim)
    mmd2 = (
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2.0 * Kxy.mean()
    )
    return float(max(mmd2, 0.0))


def coverage_score(
    real: pd.DataFrame,
    sim: pd.DataFrame,
    features: list[str],
    n_quantile_bins: int = 10,
) -> float:
    """
    Measure what fraction of the real distribution's tail bins are covered
    by the simulation.

    A coverage of 1.0 means the simulation generates samples throughout the
    full range of the real data (including rare extremes).
    """
    present = [f for f in features if f in real.columns and f in sim.columns]
    if not present:
        return np.nan

    coverage_per_feat = []
    for feat in present:
        r = _clean(real[feat])
        s = _clean(sim[feat])
        bins = np.quantile(r, np.linspace(0, 1, n_quantile_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            continue
        real_hist, _ = np.histogram(r, bins=bins)
        sim_hist, _ = np.histogram(s, bins=bins)
        covered = np.sum((real_hist > 0) & (sim_hist > 0))
        total = np.sum(real_hist > 0)
        coverage_per_feat.append(covered / total if total > 0 else 1.0)

    return float(np.mean(coverage_per_feat)) if coverage_per_feat else np.nan


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_distributions(
    real_trajectories: pd.DataFrame,
    sim_trajectories: pd.DataFrame,
    features: Optional[list[str]] = None,
) -> SimQualityReport:
    """
    Compare real vs simulated trajectory distributions.

    Parameters
    ----------
    real_trajectories : DataFrame from waymo_loader (on-road data)
    sim_trajectories  : DataFrame with same schema from a simulator
    features          : list of feature columns to compare

    Returns a SimQualityReport with per-feature and overall scores.
    """
    features = features or COMPARISON_FEATURES
    present = [f for f in features if f in real_trajectories.columns and f in sim_trajectories.columns]

    feature_results = []
    for feat in present:
        r = _clean(real_trajectories[feat])
        s = _clean(sim_trajectories[feat])
        if len(r) < 10 or len(s) < 10:
            continue

        ks_stat, ks_pval = ks_test(r, s)
        jsd = jensen_shannon_divergence(r, s)
        wass = wasserstein_distance(r, s)

        feature_results.append(FeatureComparisonResult(
            feature=feat,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            jsd=jsd,
            wasserstein=wass,
            real_mean=float(r.mean()),
            sim_mean=float(s.mean()),
            real_std=float(r.std()),
            sim_std=float(s.std()),
        ))

    # Multivariate MMD across all features jointly
    multi_present = [f for f in present if f in real_trajectories.columns]
    if len(multi_present) >= 2:
        R = real_trajectories[multi_present].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
        S = sim_trajectories[multi_present].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
        mmd = maximum_mean_discrepancy(R, S)
    else:
        mmd = np.nan

    cov = coverage_score(real_trajectories, sim_trajectories, present)

    if feature_results:
        overall = float(np.mean([r.distribution_match_score for r in feature_results]))
    else:
        overall = np.nan

    return SimQualityReport(
        feature_results=feature_results,
        mmd=mmd,
        coverage_score=cov,
        overall_score=overall,
    )
