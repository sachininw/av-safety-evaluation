"""
Visualization utilities for the AV safety evaluation framework.

All functions return matplotlib Figure objects so callers can save or display
them independently. Seaborn theme is applied at module import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and notebooks

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

FIGURE_DIR = Path("outputs/plots")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Safety metric plots
# ---------------------------------------------------------------------------

def plot_ttc_distribution(
    ttc_series: pd.Series,
    near_miss_threshold: float = 3.0,
    title: str = "Time-to-Collision Distribution",
) -> plt.Figure:
    finite_ttc = ttc_series[np.isfinite(ttc_series)].clip(upper=15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(finite_ttc, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(near_miss_threshold, color="tomato", linestyle="--", linewidth=2,
               label=f"Near-miss threshold ({near_miss_threshold} s)")
    ax.set_xlabel("Minimum TTC per frame (s)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_safety_heatmap(
    dataset_report,
    title: str = "Per-Segment Safety Heatmap",
) -> plt.Figure:
    df = dataset_report.summary_df()
    metrics = ["near_miss_rate_pct", "jerk_violation_rate_pct",
               "speed_excess_rate_pct", "proximity_warning_rate_pct"]
    present = [m for m in metrics if m in df.columns]
    heat_data = df.set_index("segment_id")[present]

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 2), max(6, len(df) * 0.4 + 1)))
    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Rate (%)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Segment")
    fig.tight_layout()
    return fig


def plot_speed_profile(
    trajectories: pd.DataFrame,
    object_id: Optional[str] = None,
    segment_id: Optional[str] = None,
    speed_limit_mps: float = 15.0,
) -> plt.Figure:
    df = trajectories.copy()
    if segment_id:
        df = df[df["segment_id"] == segment_id]
    if object_id:
        df = df[df["object_id"] == object_id]
    df = df.sort_values("timestamp_s")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["timestamp_s"], df["speed"], lw=1.5, color="steelblue")
    axes[0].axhline(speed_limit_mps, color="tomato", linestyle="--", label="Speed limit")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Speed, Acceleration, and Jerk Profile")

    axes[1].plot(df["timestamp_s"], df["acceleration"], lw=1.2, color="darkorange")
    axes[1].axhline(4.0, color="red", linestyle=":", label="4 m/s² threshold")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].legend(loc="upper right")

    axes[2].plot(df["timestamp_s"], df["jerk"].clip(-20, 20), lw=1.0, color="mediumpurple")
    axes[2].set_ylabel("Jerk (m/s³)")
    axes[2].set_xlabel("Time (s)")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rare-event / EVT plots
# ---------------------------------------------------------------------------

def plot_evt_fit(
    ttc_series: pd.Series,
    gev_fit=None,
    return_periods: Optional[list[float]] = None,
    title: str = "Extreme Value Theory — TTC Tail Analysis",
) -> plt.Figure:
    finite = ttc_series[np.isfinite(ttc_series)].values
    return_periods = return_periods or [10, 100, 1000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: empirical CDF vs fitted GEV
    ax = axes[0]
    sorted_ttc = np.sort(finite)
    n = len(sorted_ttc)
    emp_cdf = np.arange(1, n + 1) / n
    ax.plot(sorted_ttc, emp_cdf, "o", markersize=2, alpha=0.4, label="Empirical CDF")

    if gev_fit is not None:
        x_fit = np.linspace(sorted_ttc.min(), sorted_ttc.max(), 300)
        from scipy.stats import genextreme
        gev_cdf = genextreme.cdf(x_fit, -gev_fit.xi, gev_fit.mu, gev_fit.sigma)
        ax.plot(x_fit, gev_cdf, "r-", lw=2, label="GEV fit")

    ax.set_xlabel("TTC (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical vs GEV CDF")
    ax.legend()

    # Right: return level plot
    ax2 = axes[1]
    if gev_fit is not None:
        rp_range = np.logspace(0, 4, 200)
        levels = [-gev_fit.return_level(rp) for rp in rp_range]
        ax2.semilogx(rp_range, levels, "b-", lw=2)
        for rp in return_periods:
            level = -gev_fit.return_level(rp)
            ax2.axhline(level, linestyle="--", alpha=0.6, label=f"T={rp} → {level:.2f} s")
    ax2.set_xlabel("Return Period (frames)")
    ax2.set_ylabel("Return Level TTC (s)")
    ax2.set_title("Return Level Plot")
    ax2.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_mean_excess(
    thresholds: np.ndarray,
    mean_excesses: np.ndarray,
    suggested: float,
    title: str = "Mean Excess Plot (Threshold Selection)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, mean_excesses, "bo-", markersize=4)
    ax.axvline(suggested, color="tomato", linestyle="--", lw=2,
               label=f"Suggested threshold: {suggested:.2f}")
    ax.set_xlabel("Threshold u")
    ax.set_ylabel("Mean Excess E[X - u | X > u]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rare-event timeline
# ---------------------------------------------------------------------------

def plot_rare_events_timeline(
    events_df: pd.DataFrame,
    title: str = "Rare Safety Events Timeline",
) -> plt.Figure:
    if events_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No rare events detected", ha="center", va="center")
        return fig

    event_types = events_df["event_type"].unique()
    colors = dict(zip(event_types, sns.color_palette("tab10", len(event_types))))
    severity_markers = {"warning": "^", "critical": "X"}

    fig, ax = plt.subplots(figsize=(14, 5))
    for _, row in events_df.iterrows():
        ax.scatter(
            row["timestamp_s"],
            row["event_type"],
            color=colors.get(row["event_type"], "gray"),
            marker=severity_markers.get(row.get("severity", "warning"), "o"),
            s=80, zorder=3,
        )

    legend_handles = [
        mpatches.Patch(color=colors[t], label=t) for t in event_types
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Anomaly detection plots
# ---------------------------------------------------------------------------

def plot_anomaly_scores(
    trajectories: pd.DataFrame,
    score_col: str = "if_score",
    anomaly_col: str = "if_anomaly",
    title: str = "Isolation Forest Anomaly Scores",
) -> plt.Figure:
    df = trajectories.dropna(subset=[score_col])
    fig, ax = plt.subplots(figsize=(10, 5))
    normal = df[~df[anomaly_col]]
    anomalies = df[df[anomaly_col]]

    ax.scatter(range(len(normal)), normal[score_col], s=5, alpha=0.3,
               color="steelblue", label="Normal")
    if not anomalies.empty:
        ax.scatter(
            [df.index.get_loc(i) for i in anomalies.index],
            anomalies[score_col], s=20, color="tomato", label="Anomaly", zorder=4
        )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Anomaly score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: pd.Series,
    title: str = "Feature Importance for Anomaly Detection",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Simulation quality plots
# ---------------------------------------------------------------------------

def plot_distribution_comparison(
    real: pd.Series,
    sim: pd.Series,
    feature_name: str,
    title: Optional[str] = None,
) -> plt.Figure:
    title = title or f"Real vs Simulated: {feature_name}"
    r = real.replace([np.inf, -np.inf], np.nan).dropna()
    s = sim.replace([np.inf, -np.inf], np.nan).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram overlay
    lo = min(r.quantile(0.01), s.quantile(0.01))
    hi = max(r.quantile(0.99), s.quantile(0.99))
    bins = np.linspace(lo, hi, 50)
    axes[0].hist(r, bins=bins, alpha=0.6, density=True, label="Real", color="steelblue")
    axes[0].hist(s, bins=bins, alpha=0.6, density=True, label="Simulated", color="tomato")
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution Overlay")
    axes[0].legend()

    # Q-Q plot
    quantiles = np.linspace(0.01, 0.99, 100)
    r_q = np.quantile(r, quantiles)
    s_q = np.quantile(s, quantiles)
    axes[1].scatter(r_q, s_q, s=15, alpha=0.7, color="steelblue")
    lim = [min(r_q.min(), s_q.min()), max(r_q.max(), s_q.max())]
    axes[1].plot(lim, lim, "r--", lw=1.5, label="Perfect match")
    axes[1].set_xlabel("Real quantiles")
    axes[1].set_ylabel("Simulated quantiles")
    axes[1].set_title("Q-Q Plot")
    axes[1].legend()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_sim_quality_summary(report) -> plt.Figure:
    df = report.summary_df()
    df = df[df["feature"] != "OVERALL"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # JSD bar chart
    axes[0].barh(df["feature"], df["jsd"], color="steelblue", edgecolor="white")
    axes[0].axvline(0.1, color="orange", linestyle="--", label="Acceptable JSD < 0.1")
    axes[0].set_xlabel("Jensen-Shannon Divergence")
    axes[0].set_title("JSD by Feature (lower = better)")
    axes[0].legend()

    # Wasserstein
    axes[1].barh(df["feature"], df["wasserstein_dist"], color="darkorange", edgecolor="white")
    axes[1].set_xlabel("Wasserstein Distance")
    axes[1].set_title("Wasserstein Distance (lower = better)")

    # Match score
    axes[2].barh(df["feature"], df["match_score"], color="mediumseagreen", edgecolor="white")
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Distribution Match Score")
    axes[2].set_title("Match Score by Feature (higher = better)")

    fig.suptitle(
        f"Simulation Quality Summary  |  Overall: {report.overall_score:.2%}  |  MMD: {report.mmd:.4f}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig
