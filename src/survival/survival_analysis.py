"""
Survival analysis for AV rare-event timing.

Frames rare events as a time-to-event problem:
  "How long until the next near-miss, and which driving conditions
   are associated with earlier occurrence?"

This mirrors the methodology used in clinical safety trials and reliability
engineering, applied here to autonomous vehicle safety evaluation.

Methods implemented:
  1. Kaplan-Meier estimator — non-parametric survival curve with
     Greenwood confidence bands.
  2. Log-rank test — compares survival curves between groups (e.g.,
     urban vs. highway).
  3. Cox Proportional Hazards — semi-parametric model that estimates
     the hazard ratio for each covariate while leaving the baseline
     hazard unspecified.
  4. Cumulative incidence function (competing risks) — when multiple
     event types (near-miss, hard brake, swerve) compete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_survival_dataset(
    trajectories: pd.DataFrame,
    events_df: pd.DataFrame,
    event_type: str = "near_miss",
    covariates: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build a survival dataset: one row per (segment, object) track.

    For each track, the outcome is:
      - duration: time from track start to first event occurrence (or end of track)
      - event_observed: 1 if the event occurred, 0 if censored (track ended first)

    Covariates are aggregated as track-level means.
    """
    if covariates is None:
        covariates = ["mean_speed", "mean_nearest_dist", "mean_acceleration"]

    records = []
    for (seg, obj), track in trajectories.groupby(["segment_id", "object_id"]):
        track = track.sort_values("timestamp_s")
        t_start = float(track["timestamp_s"].min())
        t_end = float(track["timestamp_s"].max())
        duration = t_end - t_start

        # Check if an event of the specified type occurred in this track's segment
        seg_events = events_df[
            (events_df["segment_id"] == seg) &
            (events_df["event_type"] == event_type)
        ]
        # Take earliest event after track start
        valid_events = seg_events[seg_events["timestamp_s"] >= t_start]
        if not valid_events.empty:
            first_event_t = float(valid_events["timestamp_s"].min())
            event_duration = first_event_t - t_start
            event_observed = 1
        else:
            event_duration = duration
            event_observed = 0

        row = {
            "segment_id": seg,
            "object_id": obj,
            "duration": max(event_duration, 0.01),
            "event_observed": event_observed,
        }
        # Aggregate covariates
        for cov in covariates:
            col = cov.replace("mean_", "")
            if col in track.columns:
                row[cov] = float(track[col].mean())
            elif cov in track.columns:
                row[cov] = float(track[cov].mean())

        records.append(row)

    return pd.DataFrame(records).dropna()


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

@dataclass
class KMResult:
    timeline: np.ndarray
    survival_prob: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    median_survival: float
    n_events: int
    n_censored: int
    group_label: str


def kaplan_meier(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event_observed",
    group_label: str = "all",
) -> KMResult:
    """Fit a Kaplan-Meier estimator and return structured result."""
    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], df[event_col], label=group_label)

    ci = kmf.confidence_interval_survival_function_
    sf_col = kmf.survival_function_.columns[0]
    return KMResult(
        timeline=kmf.timeline,
        survival_prob=kmf.survival_function_[sf_col].values,
        ci_lower=ci.iloc[:, 0].values,
        ci_upper=ci.iloc[:, 1].values,
        median_survival=float(kmf.median_survival_time_),
        n_events=int(df[event_col].sum()),
        n_censored=int((1 - df[event_col]).sum()),
        group_label=group_label,
    )


def compare_survival_curves(
    df: pd.DataFrame,
    group_col: str,
    duration_col: str = "duration",
    event_col: str = "event_observed",
) -> dict:
    """
    Fit per-group KM curves and run a log-rank test.

    Returns dict with 'km_results' (list of KMResult) and 'logrank' stats.
    """
    groups = df[group_col].unique()
    km_results = []
    group_data = {}

    for grp in groups:
        sub = df[df[group_col] == grp]
        km_results.append(kaplan_meier(sub, duration_col, event_col, group_label=str(grp)))
        group_data[grp] = sub

    # Pairwise log-rank tests for all group pairs
    logrank_results = []
    grp_list = list(groups)
    for i in range(len(grp_list)):
        for j in range(i + 1, len(grp_list)):
            a, b = grp_list[i], grp_list[j]
            da, db = group_data[a], group_data[b]
            result = logrank_test(
                da[duration_col], db[duration_col],
                da[event_col], db[event_col],
            )
            logrank_results.append({
                "group_a": str(a), "group_b": str(b),
                "test_statistic": float(result.test_statistic),
                "p_value": float(result.p_value),
                "significant": result.p_value < 0.05,
            })

    return {
        "km_results": km_results,
        "logrank": pd.DataFrame(logrank_results),
    }


# ---------------------------------------------------------------------------
# Cox Proportional Hazards Model
# ---------------------------------------------------------------------------

@dataclass
class CoxResult:
    summary: pd.DataFrame      # HR, CI, p-value per covariate
    concordance: float         # C-index (discrimination)
    log_likelihood: float
    aic: float
    n_events: int


def cox_proportional_hazards(
    survival_df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event_observed",
    covariates: Optional[list[str]] = None,
    penalizer: float = 0.1,
) -> CoxResult:
    """
    Fit a Cox PH model and return interpretable hazard ratios.

    The hazard ratio HR > 1 means the covariate increases the instantaneous
    risk of an event (bad for safety covariates like acceleration).
    """
    if covariates is None:
        covariates = [c for c in survival_df.columns
                      if c not in (duration_col, event_col, "segment_id", "object_id")]

    present = [c for c in covariates if c in survival_df.columns]
    cols = [duration_col, event_col] + present
    df = survival_df[cols].dropna()

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%",
                             "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"]
    summary["significant"] = summary["p_value"] < 0.05

    return CoxResult(
        summary=summary.reset_index(),
        concordance=float(cph.concordance_index_),
        log_likelihood=float(cph.log_likelihood_),
        aic=float(cph.AIC_partial_),
        n_events=int(df[event_col].sum()),
    )


# ---------------------------------------------------------------------------
# Cumulative Incidence (competing risks)
# ---------------------------------------------------------------------------

def competing_risks_incidence(
    events_df: pd.DataFrame,
    trajectories: pd.DataFrame,
    event_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Estimate the cause-specific cumulative incidence for each event type.

    In an AV context, multiple event types (near_miss, hard_braking, swerve)
    compete — once one occurs it "uses up" the observation.  The Aalen-Johansen
    estimator provides a consistent estimate of P(event type k by time t).

    This implementation uses a simplified cause-specific hazard approach:
    for each event type k, fit KM on the cause-specific event indicator
    (treating other event types as censored).
    """
    if event_types is None:
        event_types = events_df["event_type"].unique().tolist()

    all_segs = trajectories["segment_id"].unique()
    records = []
    for seg in all_segs:
        track = trajectories[trajectories["segment_id"] == seg].sort_values("timestamp_s")
        t_start = float(track["timestamp_s"].min())
        t_end = float(track["timestamp_s"].max())
        seg_events = events_df[events_df["segment_id"] == seg]

        for et in event_types:
            this_type = seg_events[seg_events["event_type"] == et]
            if not this_type.empty:
                first_t = float(this_type["timestamp_s"].min())
                duration = first_t - t_start
                event_observed = 1
            else:
                duration = t_end - t_start
                event_observed = 0
            records.append({
                "segment_id": seg,
                "event_type": et,
                "duration": max(duration, 0.01),
                "event_observed": event_observed,
            })

    result_df = pd.DataFrame(records)
    ci_rows = []
    for et in event_types:
        sub = result_df[result_df["event_type"] == et]
        if sub["event_observed"].sum() < 2:
            continue
        km = kaplan_meier(sub, group_label=et)
        ci_rows.append({
            "event_type": et,
            "median_survival_s": km.median_survival,
            "n_events": km.n_events,
            "incidence_at_60s": float(1 - km.survival_prob[
                np.searchsorted(km.timeline, 60, side="right") - 1
            ]) if km.timeline.max() >= 60 else np.nan,
        })

    return pd.DataFrame(ci_rows)
