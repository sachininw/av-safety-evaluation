"""
Causal inference for AV safety analysis.

Standard correlation analysis cannot distinguish between:
  - "Dense traffic causes more near-misses" (causation)
  - "Certain road segments attract both dense traffic and near-misses" (confounding)

This module estimates causal effects using three complementary frameworks:

  1. Propensity Score Matching (PSM) — matches treated/control units on
     observable confounders to estimate Average Treatment Effect (ATE).
  2. Inverse Probability Weighting (IPW) — reweights observations to create
     a pseudo-population in which treatment is independent of confounders.
  3. Granger Causality — tests whether past values of X improve prediction
     of Y beyond Y's own past (time-series causality within a track).
  4. Structural Counterfactual Analysis — estimates what safety metric Y
     would have been under a hypothetical intervention do(X = x').
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests


# ---------------------------------------------------------------------------
# Propensity Score Matching
# ---------------------------------------------------------------------------

@dataclass
class ATEResult:
    ate: float               # Average Treatment Effect
    ate_se: float            # Standard error
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    n_matched: int
    p_value: float
    treatment: str
    outcome: str

    def __str__(self) -> str:
        sig = "✓ significant" if self.p_value < 0.05 else "✗ not significant"
        return (
            f"ATE({self.treatment} → {self.outcome}): {self.ate:+.4f} "
            f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  "
            f"p={self.p_value:.4f}  {sig}"
        )


def estimate_propensity_scores(
    df: pd.DataFrame,
    treatment_col: str,
    confounder_cols: list[str],
    max_iter: int = 500,
) -> np.ndarray:
    """
    Fit a logistic regression to estimate P(treatment=1 | confounders).

    Returns an array of propensity scores in [0, 1].
    """
    X = df[confounder_cols].fillna(0).values.astype(float)
    y = df[treatment_col].values.astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(X_scaled, y)
    return clf.predict_proba(X_scaled)[:, 1]


def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: list[str],
    caliper: float = 0.05,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> ATEResult:
    """
    Estimate the Average Treatment Effect (ATE) via 1:1 nearest-neighbour
    propensity score matching with a caliper constraint.

    Parameters
    ----------
    df              : DataFrame with one row per unit of analysis
    treatment_col   : binary column (0/1) indicating treatment
    outcome_col     : continuous safety outcome (e.g., min_ttc)
    confounder_cols : features to balance between treated and control
    caliper         : maximum allowed propensity score difference for a match

    Returns ATEResult with point estimate, SE, and 95% CI (bootstrap).
    """
    df = df[[treatment_col, outcome_col] + confounder_cols].dropna().copy()
    ps = estimate_propensity_scores(df, treatment_col, confounder_cols)
    df["_ps"] = ps

    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    # Greedy 1:1 nearest-neighbour matching without replacement
    matched_pairs: list[tuple[int, int]] = []
    used_control = set()
    for t_idx in treated.index:
        t_ps = df.loc[t_idx, "_ps"]
        candidates = control[~control.index.isin(used_control)]
        if candidates.empty:
            continue
        diffs = (candidates["_ps"] - t_ps).abs()
        best_idx = diffs.idxmin()
        if diffs[best_idx] <= caliper:
            matched_pairs.append((t_idx, best_idx))
            used_control.add(best_idx)

    if len(matched_pairs) < 10:
        raise ValueError(
            f"Too few matched pairs ({len(matched_pairs)}). "
            "Try a larger caliper or more data."
        )

    t_outcomes = df.loc[[p[0] for p in matched_pairs], outcome_col].values
    c_outcomes = df.loc[[p[1] for p in matched_pairs], outcome_col].values
    ate_point = float((t_outcomes - c_outcomes).mean())

    # Bootstrap SE
    rng = np.random.default_rng(seed)
    boot_ates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(matched_pairs), len(matched_pairs))
        bt = t_outcomes[idx]
        bc = c_outcomes[idx]
        boot_ates.append((bt - bc).mean())
    ate_se = float(np.std(boot_ates))
    ci_lower = float(np.percentile(boot_ates, 2.5))
    ci_upper = float(np.percentile(boot_ates, 97.5))

    t_stat = ate_point / (ate_se + 1e-12)
    p_value = float(2 * stats.norm.sf(abs(t_stat)))

    return ATEResult(
        ate=ate_point,
        ate_se=ate_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_treated=len(treated),
        n_control=len(control),
        n_matched=len(matched_pairs),
        p_value=p_value,
        treatment=treatment_col,
        outcome=outcome_col,
    )


# ---------------------------------------------------------------------------
# Inverse Probability Weighting (IPW)
# ---------------------------------------------------------------------------

def ipw_ate(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: list[str],
    trim_quantile: float = 0.01,
) -> ATEResult:
    """
    Estimate ATE via Horvitz-Thompson IPW estimator.

    Extreme propensity scores (near 0 or 1) are trimmed to reduce variance.
    """
    df = df[[treatment_col, outcome_col] + confounder_cols].dropna().copy()
    ps = estimate_propensity_scores(df, treatment_col, confounder_cols)

    lo, hi = np.quantile(ps, [trim_quantile, 1 - trim_quantile])
    mask = (ps > lo) & (ps < hi)
    ps = ps[mask]
    df = df.iloc[mask].copy()

    T = df[treatment_col].values.astype(float)
    Y = df[outcome_col].values.astype(float)

    # Normalised IPW (Hajek estimator)
    w_treated = T / ps
    w_control = (1 - T) / (1 - ps)
    mu1 = np.average(Y, weights=w_treated)
    mu0 = np.average(Y, weights=w_control)
    ate = float(mu1 - mu0)

    # Sandwich SE via delta method approximation
    n = len(Y)
    pseudo_y1 = T * Y / ps - (T - ps) / ps * mu1
    pseudo_y0 = (1 - T) * Y / (1 - ps) + (T - ps) / (1 - ps) * mu0
    se = float(np.std(pseudo_y1 - pseudo_y0) / np.sqrt(n))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    p_value = float(2 * stats.norm.sf(abs(ate / (se + 1e-12))))

    return ATEResult(
        ate=ate, ate_se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        n_treated=int(T.sum()), n_control=int((1 - T).sum()),
        n_matched=len(df), p_value=p_value,
        treatment=treatment_col, outcome=outcome_col,
    )


# ---------------------------------------------------------------------------
# Granger Causality
# ---------------------------------------------------------------------------

@dataclass
class GrangerResult:
    cause: str
    effect: str
    max_lag: int
    f_statistic: float
    p_value: float
    significant: bool

    def __str__(self) -> str:
        sig = "✓" if self.significant else "✗"
        return (
            f"{sig} Granger: {self.cause} → {self.effect}  "
            f"F={self.f_statistic:.3f}  p={self.p_value:.4f}  lag={self.max_lag}"
        )


def granger_causality(
    track: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> GrangerResult:
    """
    Test whether `cause_col` Granger-causes `effect_col` within a single track.

    Uses the VAR-based F-test from statsmodels.  A significant result means
    that past values of the cause improve prediction of the effect beyond the
    effect's own past.

    track: DataFrame for a single (segment_id, object_id) sorted by time.
    """
    data = track[[effect_col, cause_col]].dropna().values
    if len(data) < max_lag * 3 + 10:
        raise ValueError(f"Track too short ({len(data)} rows) for lag {max_lag}")

    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    # Pick the lag with the smallest p-value
    best_lag = min(results, key=lambda k: results[k][0]["ssr_ftest"][1])
    f_stat = float(results[best_lag][0]["ssr_ftest"][0])
    p_val = float(results[best_lag][0]["ssr_ftest"][1])

    return GrangerResult(
        cause=cause_col,
        effect=effect_col,
        max_lag=best_lag,
        f_statistic=f_stat,
        p_value=p_val,
        significant=p_val < alpha,
    )


def batch_granger(
    trajectories: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 5,
    min_track_len: int = 50,
) -> pd.DataFrame:
    """
    Run Granger causality tests across all tracks and aggregate results.

    Returns a DataFrame with one row per (segment_id, object_id) track.
    """
    records = []
    for (seg, obj), track in trajectories.groupby(["segment_id", "object_id"]):
        if len(track) < min_track_len:
            continue
        try:
            result = granger_causality(
                track.sort_values("timestamp_s"),
                cause_col, effect_col, max_lag,
            )
            records.append({
                "segment_id": seg, "object_id": obj,
                "f_statistic": result.f_statistic,
                "p_value": result.p_value,
                "best_lag": result.max_lag,
                "significant": result.significant,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Structural Counterfactual Analysis
# ---------------------------------------------------------------------------

def counterfactual_ttc(
    trajectories: pd.DataFrame,
    intervention_col: str,
    intervention_value: float,
    outcome_col: str = "speed",
    confounder_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Estimate the counterfactual outcome under do(intervention_col = value).

    Uses a linear structural equation model fitted on the observational data.
    The counterfactual is computed by:
      1. Fitting Y ~ X + confounders using OLS.
      2. Predicting Y with intervention_col set to intervention_value for all rows.

    Returns the original DataFrame with two new columns:
      - cf_<outcome_col>:   predicted counterfactual outcome
      - cf_delta_<outcome>: counterfactual − factual difference
    """
    from sklearn.linear_model import LinearRegression

    if confounder_cols is None:
        confounder_cols = []

    feature_cols = [intervention_col] + confounder_cols
    present = [c for c in feature_cols if c in trajectories.columns]

    df = trajectories.copy()
    X = df[present].fillna(0).values.astype(float)
    Y = df[outcome_col].fillna(0).values.astype(float)

    model = LinearRegression()
    model.fit(X, Y)

    # Counterfactual: override the intervention column
    X_cf = X.copy()
    int_idx = present.index(intervention_col)
    X_cf[:, int_idx] = intervention_value

    cf_label = f"cf_{outcome_col}"
    df[cf_label] = model.predict(X_cf)
    df[f"cf_delta_{outcome_col}"] = df[cf_label] - Y
    return df


# ---------------------------------------------------------------------------
# Convenience: run all causal analyses on a dataset
# ---------------------------------------------------------------------------

def causal_safety_report(
    segment_df: pd.DataFrame,
    treatment_col: str = "high_density",
    outcome_col: str = "min_ttc",
    confounder_cols: Optional[list[str]] = None,
) -> dict:
    """
    Run PSM, IPW, and batch Granger causality and return a summary dict.

    `segment_df` should be a per-(segment, frame) DataFrame where each row
    has been aggregated to a single unit of analysis (e.g., mean metrics
    per segment, with a binary treatment flag).
    """
    if confounder_cols is None:
        confounder_cols = ["mean_speed", "n_agents", "mean_nearest_dist"]

    present_conf = [c for c in confounder_cols if c in segment_df.columns]
    report: dict = {}

    try:
        report["psm"] = propensity_score_matching(
            segment_df, treatment_col, outcome_col, present_conf
        )
    except Exception as e:
        report["psm_error"] = str(e)

    try:
        report["ipw"] = ipw_ate(
            segment_df, treatment_col, outcome_col, present_conf
        )
    except Exception as e:
        report["ipw_error"] = str(e)

    return report
