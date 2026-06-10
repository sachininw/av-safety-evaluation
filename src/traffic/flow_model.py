"""
Traffic flow modelling from Waymo trajectory data.

Maps extracted agent trajectories onto macroscopic traffic flow theory.
This directly addresses the "traffic modeling" preferred qualification.

Models implemented:
  1. Greenshields linear speed-density model — the simplest fundamental
     diagram: v = v_f * (1 - k/k_j)
  2. Greenberg logarithmic model — better fit in congested regime
  3. Fundamental diagram (flow-density) with regime classification
  4. Traffic state machine — free flow, transitional, congested
  5. Space-mean speed and density estimation from floating car data

The "floating car" approach uses agent trajectories as probe vehicles
to estimate macroscopic flow conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats


# ---------------------------------------------------------------------------
# Macroscopic flow variables from trajectory data
# ---------------------------------------------------------------------------

def estimate_flow_variables(
    trajectories: pd.DataFrame,
    cell_length_m: float = 100.0,
    time_window_s: float = 5.0,
) -> pd.DataFrame:
    """
    Estimate macroscopic traffic variables per (segment, time_window) cell.

    Uses Edie's definitions which are valid for floating car data:
      - Flow q = total distance / (cell_length × time_window)
      - Density k = total time / (cell_length × time_window)
      - Space-mean speed v = q / k

    Returns a DataFrame with columns: segment_id, window_start_s,
    density_veh_per_m, flow_veh_per_s, space_mean_speed_mps, n_vehicles.
    """
    vehicles = trajectories[trajectories["object_type_name"] == "vehicle"].copy()
    vehicles["time_window"] = (vehicles["timestamp_s"] // time_window_s).astype(int)

    records = []
    for (seg, win), group in vehicles.groupby(["segment_id", "time_window"]):
        dt = group.groupby("object_id")["dt"].sum()
        ds = group.groupby("object_id").apply(
            lambda g: float(np.trapezoid(g["speed"].values, g["timestamp_s"].values))
        )

        total_time = float(dt.sum())
        total_dist = float(ds.sum())
        area = cell_length_m * time_window_s

        if area == 0 or total_time == 0:
            continue

        q = total_dist / area
        k = total_time / area
        v = q / k if k > 0 else 0.0

        records.append({
            "segment_id": seg,
            "window_start_s": win * time_window_s,
            "density_veh_per_m": k,
            "flow_veh_per_s": q,
            "space_mean_speed_mps": v,
            "n_vehicles": group["object_id"].nunique(),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Speed-density models
# ---------------------------------------------------------------------------

def greenshields_model(k: np.ndarray, v_f: float, k_j: float) -> np.ndarray:
    """Greenshields linear model: v = v_f * (1 - k/k_j)"""
    return v_f * (1.0 - k / k_j)


def greenberg_model(k: np.ndarray, v_opt: float, k_j: float) -> np.ndarray:
    """Greenberg log model: v = v_opt * ln(k_j / k)"""
    k_safe = np.clip(k, 1e-6, k_j - 1e-6)
    return v_opt * np.log(k_j / k_safe)


@dataclass
class FundamentalDiagramFit:
    model_name: str
    v_free_flow: float        # free-flow speed (m/s)
    k_jam: float              # jam density (veh/m)
    q_capacity: float         # capacity flow (veh/s)
    k_critical: float         # density at capacity
    rmse: float
    r2: float
    fitted_params: dict

    def predict_speed(self, k: np.ndarray) -> np.ndarray:
        if self.model_name == "greenshields":
            return greenshields_model(k, self.v_free_flow, self.k_jam)
        elif self.model_name == "greenberg":
            return greenberg_model(k, self.fitted_params["v_opt"], self.k_jam)
        return np.full_like(k, self.v_free_flow)

    def predict_flow(self, k: np.ndarray) -> np.ndarray:
        return k * self.predict_speed(k)


def fit_fundamental_diagram(
    flow_df: pd.DataFrame,
    model: str = "greenshields",
) -> FundamentalDiagramFit:
    """
    Fit a speed-density model to observed macroscopic flow data.

    flow_df: output of estimate_flow_variables (must have density and speed cols).
    """
    k = flow_df["density_veh_per_m"].values
    v = flow_df["space_mean_speed_mps"].values

    # Filter physically meaningful observations
    mask = (k > 0) & (v >= 0) & np.isfinite(k) & np.isfinite(v)
    k, v = k[mask], v[mask]

    if len(k) < 5:
        raise ValueError(f"Too few observations ({len(k)}) to fit fundamental diagram")

    # Initial guesses
    v_f_init = float(np.percentile(v, 90))
    k_j_init = float(k.max() * 2)

    if model == "greenshields":
        popt, _ = curve_fit(
            greenshields_model, k, v,
            p0=[v_f_init, k_j_init],
            bounds=([0, k.max()], [50, 1.0]),
            maxfev=5000,
        )
        v_f, k_j = popt
        v_pred = greenshields_model(k, v_f, k_j)
        k_crit = k_j / 2.0
        q_cap = v_f * k_j / 4.0
        params = {"v_f": v_f, "k_j": k_j}

    elif model == "greenberg":
        def _gberg(k_, v_opt, k_j_):
            return greenberg_model(k_, v_opt, k_j_)
        popt, _ = curve_fit(
            _gberg, k, v,
            p0=[v_f_init * 0.6, k_j_init],
            bounds=([0, k.max()], [50, 1.0]),
            maxfev=5000,
        )
        v_opt, k_j = popt
        v_f = v_opt * np.log(k_j / 1e-3)
        v_pred = greenberg_model(k, v_opt, k_j)
        k_crit = k_j / np.e
        q_cap = v_opt * k_crit
        params = {"v_opt": v_opt, "k_j": k_j}
    else:
        raise ValueError(f"Unknown model: {model}. Use 'greenshields' or 'greenberg'")

    rmse = float(np.sqrt(mean_squared_error_simple(v, v_pred)))
    r2 = float(r2_score_simple(v, v_pred))

    return FundamentalDiagramFit(
        model_name=model,
        v_free_flow=float(v_f),
        k_jam=float(k_j),
        q_capacity=float(q_cap),
        k_critical=float(k_crit),
        rmse=rmse,
        r2=r2,
        fitted_params=params,
    )


def mean_squared_error_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


# ---------------------------------------------------------------------------
# Traffic state classification
# ---------------------------------------------------------------------------

def classify_traffic_state(
    flow_df: pd.DataFrame,
    fd_fit: Optional[FundamentalDiagramFit] = None,
    v_free_flow: Optional[float] = None,
) -> pd.DataFrame:
    """
    Classify each observation as Free Flow, Transitional, or Congested.

    If a fitted FundamentalDiagramFit is provided, uses the critical density
    to separate regimes.  Otherwise uses speed thresholds.
    """
    df = flow_df.copy()

    if fd_fit is not None:
        k_crit = fd_fit.k_critical
        df["traffic_state"] = pd.cut(
            df["density_veh_per_m"],
            bins=[-np.inf, k_crit * 0.7, k_crit * 1.3, np.inf],
            labels=["free_flow", "transitional", "congested"],
        )
    else:
        v_f = v_free_flow or float(df["space_mean_speed_mps"].quantile(0.85))
        df["traffic_state"] = pd.cut(
            df["space_mean_speed_mps"],
            bins=[-np.inf, v_f * 0.4, v_f * 0.75, np.inf],
            labels=["congested", "transitional", "free_flow"],
        )

    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_fundamental_diagram(
    flow_df: pd.DataFrame,
    fd_fit: Optional[FundamentalDiagramFit] = None,
    title: str = "Fundamental Traffic Flow Diagram",
):
    """Plot speed-density and flow-density diagrams side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    k_obs = flow_df["density_veh_per_m"].values
    v_obs = flow_df["space_mean_speed_mps"].values
    q_obs = flow_df["flow_veh_per_s"].values

    # Speed-density
    axes[0].scatter(k_obs, v_obs, s=20, alpha=0.5, color="steelblue", label="Observed")
    if fd_fit is not None:
        k_grid = np.linspace(0, k_obs.max() * 1.1, 200)
        v_fit = fd_fit.predict_speed(k_grid)
        axes[0].plot(k_grid, v_fit, "r-", lw=2,
                     label=f"{fd_fit.model_name} fit (R²={fd_fit.r2:.3f})")
        axes[0].axvline(fd_fit.k_critical, color="orange", linestyle="--",
                        label=f"k_crit={fd_fit.k_critical:.4f}")
    axes[0].set_xlabel("Density k (veh/m)")
    axes[0].set_ylabel("Space-mean speed v (m/s)")
    axes[0].set_title("Speed-Density Diagram")
    axes[0].legend()

    # Flow-density
    axes[1].scatter(k_obs, q_obs, s=20, alpha=0.5, color="darkorange", label="Observed")
    if fd_fit is not None:
        q_fit = fd_fit.predict_flow(k_grid)
        axes[1].plot(k_grid, q_fit, "r-", lw=2, label="Fitted flow")
        axes[1].axvline(fd_fit.k_critical, color="orange", linestyle="--",
                        label=f"Capacity={fd_fit.q_capacity:.4f} veh/s")
    axes[1].set_xlabel("Density k (veh/m)")
    axes[1].set_ylabel("Flow q (veh/s)")
    axes[1].set_title("Flow-Density Diagram")
    axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
