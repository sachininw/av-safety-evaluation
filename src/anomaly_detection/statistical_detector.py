"""
Statistical anomaly detection for AV trajectory data.

Implements three complementary methods:
  1. Z-score detector — flags values beyond N standard deviations
  2. CUSUM (Cumulative Sum) — detects sustained shifts in a signal
  3. Multivariate Mahalanobis distance — jointly flags unusual feature combinations

These are interpretable, low-overhead methods well-suited for first-pass
anomaly flagging across large amounts of driving data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Z-score detector
# ---------------------------------------------------------------------------

@dataclass
class ZScoreConfig:
    threshold: float = 3.0
    features: list[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = ["speed", "acceleration", "jerk", "heading_rate"]


def zscore_detector(
    trajectories: pd.DataFrame,
    config: Optional[ZScoreConfig] = None,
) -> pd.DataFrame:
    """
    Flag rows whose feature values exceed `config.threshold` standard deviations
    from the per-segment mean.

    Returns the input DataFrame with added columns:
      - z_<feature>: z-score for each feature
      - zscore_anomaly: True if any feature z-score exceeds the threshold
      - zscore_max_z: the highest absolute z-score for that row
    """
    cfg = config or ZScoreConfig()
    df = trajectories.copy()
    present = [f for f in cfg.features if f in df.columns]

    for feat in present:
        group_mean = df.groupby("segment_id")[feat].transform("mean")
        group_std = df.groupby("segment_id")[feat].transform("std").replace(0, np.nan)
        df[f"z_{feat}"] = (df[feat] - group_mean) / group_std

    z_cols = [f"z_{f}" for f in present]
    df["zscore_max_z"] = df[z_cols].abs().max(axis=1)
    df["zscore_anomaly"] = df["zscore_max_z"] > cfg.threshold
    return df


# ---------------------------------------------------------------------------
# CUSUM detector
# ---------------------------------------------------------------------------

def cusum_detector(
    series: np.ndarray,
    target: Optional[float] = None,
    slack: float = 0.5,
    threshold: float = 5.0,
) -> np.ndarray:
    """
    Upper CUSUM for detecting upward shifts in a univariate series.

    Parameters
    ----------
    series    : 1-D array of observations
    target    : in-control mean (default: series mean)
    slack     : allowance parameter k (in units of series std)
    threshold : decision limit h (in units of series std)

    Returns a boolean array — True marks frames where CUSUM exceeds the limit.
    """
    x = np.asarray(series, dtype=float)
    mu = target if target is not None else float(np.nanmean(x))
    sigma = float(np.nanstd(x))
    if sigma < 1e-10:
        return np.zeros(len(x), dtype=bool)

    k = slack * sigma
    h = threshold * sigma
    cusum_pos = np.zeros(len(x))
    for i in range(1, len(x)):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + (x[i] - mu) - k)

    return cusum_pos > h


def apply_cusum_to_trajectories(
    trajectories: pd.DataFrame,
    feature: str = "acceleration",
    slack: float = 0.5,
    threshold: float = 5.0,
) -> pd.DataFrame:
    """Apply the CUSUM detector track-by-track and add a 'cusum_anomaly' column."""
    df = trajectories.copy().sort_values(["segment_id", "object_id", "timestamp_us"])
    flags = np.zeros(len(df), dtype=bool)

    for (seg, obj), idx in df.groupby(["segment_id", "object_id"]).groups.items():
        if feature not in df.columns:
            break
        series = df.loc[idx, feature].values
        flags[df.index.get_indexer(idx)] = cusum_detector(series, slack=slack, threshold=threshold)

    df["cusum_anomaly"] = flags
    return df


# ---------------------------------------------------------------------------
# Mahalanobis distance detector
# ---------------------------------------------------------------------------

def mahalanobis_detector(
    trajectories: pd.DataFrame,
    features: Optional[list[str]] = None,
    threshold_percentile: float = 99.0,
) -> pd.DataFrame:
    """
    Flag rows with high Mahalanobis distance from the multivariate feature mean.

    The covariance matrix is estimated per segment.  Rows with distance above
    `threshold_percentile` of the chi-squared distribution (df = n_features)
    are flagged as anomalies.

    Returns input DataFrame with 'mahal_distance' and 'mahal_anomaly' columns.
    """
    if features is None:
        features = ["speed", "acceleration", "heading_rate"]

    present = [f for f in features if f in trajectories.columns]
    if len(present) < 2:
        trajectories["mahal_distance"] = np.nan
        trajectories["mahal_anomaly"] = False
        return trajectories

    df = trajectories.copy()
    df["mahal_distance"] = np.nan
    df["mahal_anomaly"] = False

    chi2_threshold = stats.chi2.ppf(threshold_percentile / 100.0, df=len(present))

    for seg_id, group in df.groupby("segment_id"):
        X = group[present].dropna()
        if len(X) < len(present) + 2:
            continue
        try:
            mu = X.mean().values
            cov = np.cov(X.values.T)
            inv_cov = np.linalg.pinv(cov)
            diff = X.values - mu
            dist_sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
            dist = np.sqrt(np.abs(dist_sq))
            df.loc[X.index, "mahal_distance"] = dist
            df.loc[X.index, "mahal_anomaly"] = dist_sq > chi2_threshold
        except np.linalg.LinAlgError:
            continue

    return df


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------

def combined_statistical_anomalies(
    trajectories: pd.DataFrame,
    zscore_threshold: float = 3.5,
    cusum_feature: str = "acceleration",
) -> pd.DataFrame:
    """Run all statistical detectors and add a 'statistical_anomaly' summary flag."""
    df = zscore_detector(trajectories, ZScoreConfig(threshold=zscore_threshold))
    df = apply_cusum_to_trajectories(df, feature=cusum_feature)
    df = mahalanobis_detector(df)
    df["statistical_anomaly"] = (
        df["zscore_anomaly"] | df["cusum_anomaly"] | df.get("mahal_anomaly", False)
    )
    return df
