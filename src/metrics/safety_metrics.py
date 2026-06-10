"""
Safety metrics for autonomous vehicle performance evaluation.

Implements the core metrics used to quantify AV safety and comfort:
  - Time to Collision (TTC)
  - Post-Encroachment Time (PET)
  - Deceleration Rate to Avoid a Crash (DRAC)
  - Proximity Score (distance to nearest agent)
  - Jerk and lateral acceleration (ride comfort proxies)
  - Speed deviation from expected flow
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# --- Time-to-Collision -------------------------------------------------------

def time_to_collision(
    ego_x: float, ego_y: float, ego_vx: float, ego_vy: float,
    agent_x: float, agent_y: float, agent_vx: float, agent_vy: float,
    agent_length: float = 4.5, agent_width: float = 2.0,
) -> float:
    """
    Compute straight-line TTC between ego vehicle and one agent.

    Returns np.inf when the gap is not closing (vehicles moving apart).
    Uses a simplified 1-D model along the line connecting both objects.
    """
    dx = agent_x - ego_x
    dy = agent_y - ego_y
    dist = np.sqrt(dx**2 + dy**2)

    # Safety buffer accounts for approximate agent footprint
    safety_margin = np.sqrt(agent_length**2 + agent_width**2) / 2.0
    gap = max(dist - safety_margin, 0.0)

    # Relative velocity component along the separation vector
    if dist < 1e-6:
        return 0.0
    ux, uy = dx / dist, dy / dist
    rel_vx = ego_vx - agent_vx
    rel_vy = ego_vy - agent_vy
    closing_speed = rel_vx * ux + rel_vy * uy  # positive => closing

    if closing_speed <= 0:
        return np.inf
    return gap / closing_speed


def compute_ttc_frame(frame_df: pd.DataFrame) -> pd.Series:
    """
    Compute minimum TTC for every agent in a single timestep frame.

    frame_df must have columns: object_id, x, y, vx, vy, length, width,
    plus a single row where object_id == 'ego' (or the first row is treated
    as the ego vehicle).

    Returns a Series indexed by object_id with TTC values.
    """
    if frame_df.empty:
        return pd.Series(dtype=float)

    ego = frame_df.iloc[0]
    ttc_values = {}
    for _, agent in frame_df.iloc[1:].iterrows():
        ttc_values[agent["object_id"]] = time_to_collision(
            ego_x=ego["x"], ego_y=ego["y"],
            ego_vx=ego.get("vx", 0.0), ego_vy=ego.get("vy", 0.0),
            agent_x=agent["x"], agent_y=agent["y"],
            agent_vx=agent.get("vx", 0.0), agent_vy=agent.get("vy", 0.0),
            agent_length=agent.get("length", 4.5),
            agent_width=agent.get("width", 2.0),
        )
    return pd.Series(ttc_values)


def _vectorized_frame_ttc(
    x: np.ndarray, y: np.ndarray,
    vx: np.ndarray, vy: np.ndarray,
    safety_margin: float = 3.0,
) -> float:
    """
    Vectorised minimum TTC across all pairs in a single frame.

    Uses numpy broadcasting — O(n²) in memory but avoids Python-level loops,
    running ~100× faster than the row-iteration approach for typical frame sizes.
    """
    n = len(x)
    if n < 2:
        return np.inf

    # Pairwise separation vectors (i, j): agent i looking at agent j
    dx = x[None, :] - x[:, None]   # (n, n)
    dy = y[None, :] - y[:, None]
    dist = np.sqrt(dx**2 + dy**2)

    # Relative velocity of i relative to j
    dvx = vx[:, None] - vx[None, :]
    dvy = vy[:, None] - vy[None, :]

    # Unit vector along separation; avoid div-by-zero on diagonal
    eps = 1e-8
    safe_dist = np.where(dist > eps, dist, eps)
    ux = dx / safe_dist
    uy = dy / safe_dist

    # Closing speed: positive means approaching
    closing = dvx * ux + dvy * uy

    gap = np.maximum(dist - safety_margin, 0.0)
    ttc = np.where(closing > 0, gap / closing, np.inf)
    np.fill_diagonal(ttc, np.inf)  # exclude self-pairs

    return float(ttc.min())


def batch_ttc(trajectories: pd.DataFrame) -> pd.DataFrame:
    """
    Compute minimum TTC per (segment_id, timestamp_us) across all vehicle pairs.

    Uses vectorised numpy broadcasting — typically 50–100× faster than
    the naive row-iteration approach.

    Returns a DataFrame with columns: segment_id, timestamp_us, min_ttc,
    near_miss (bool: TTC < 3 s).
    """
    vehicles = trajectories[trajectories["object_type_name"] == "vehicle"]
    records = []

    for (seg, ts), frame in vehicles.groupby(["segment_id", "timestamp_us"]):
        if len(frame) < 2:
            continue
        x  = frame["x"].values.astype(float)
        y  = frame["y"].values.astype(float)
        vx = frame["vx"].values.astype(float) if "vx" in frame.columns else np.zeros(len(frame))
        vy = frame["vy"].values.astype(float) if "vy" in frame.columns else np.zeros(len(frame))

        min_ttc = _vectorized_frame_ttc(x, y, vx, vy)
        records.append({
            "segment_id": seg,
            "timestamp_us": ts,
            "min_ttc": min_ttc,
            "near_miss": min_ttc < 3.0,
        })

    return pd.DataFrame(records)


# --- DRAC (Deceleration Rate to Avoid Crash) --------------------------------

def drac(ego_speed: float, agent_speed: float, distance: float) -> float:
    """
    Required deceleration for ego to avoid rear-end collision with agent ahead.

    DRAC = (v_ego - v_lead)^2 / (2 * gap)   when ego is faster than lead.
    Returns 0.0 when ego is slower or distance is non-positive.
    """
    rel_speed = ego_speed - agent_speed
    if rel_speed <= 0 or distance <= 0:
        return 0.0
    return (rel_speed**2) / (2.0 * distance)


# --- Proximity score --------------------------------------------------------

def nearest_agent_distances(trajectories: pd.DataFrame) -> pd.DataFrame:
    """
    For each agent at each frame, find the distance to its nearest neighbour.

    Uses a per-frame KD-tree for efficiency.  Returns the input DataFrame
    with an added 'nearest_agent_dist' column.
    """
    result_rows = []
    for (seg, ts), frame in trajectories.groupby(["segment_id", "timestamp_us"]):
        coords = frame[["x", "y"]].values
        if len(coords) < 2:
            frame = frame.copy()
            frame["nearest_agent_dist"] = np.inf
            result_rows.append(frame)
            continue

        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)  # k=2: self + nearest neighbour
        nearest = dists[:, 1]

        frame = frame.copy()
        frame["nearest_agent_dist"] = nearest
        result_rows.append(frame)

    return pd.concat(result_rows, ignore_index=True)


# --- Jerk and comfort metrics -----------------------------------------------

def comfort_metrics(trajectories: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate comfort statistics per (segment_id, object_id) track.

    Returns a summary DataFrame with:
      - mean/max longitudinal acceleration
      - mean/max jerk
      - mean/max heading rate (proxy for lateral discomfort)
      - distance covered
    """
    grp = trajectories.groupby(["segment_id", "object_id"])

    summary = grp.agg(
        mean_acceleration=("acceleration", "mean"),
        max_acceleration=("acceleration", "max"),
        mean_jerk=("jerk", lambda s: s.abs().mean()),
        max_jerk=("jerk", lambda s: s.abs().max()),
        mean_heading_rate=("heading_rate", lambda s: s.abs().mean()),
        max_heading_rate=("heading_rate", lambda s: s.abs().max()),
        mean_speed=("speed", "mean"),
        max_speed=("speed", "max"),
        track_duration_s=("timestamp_s", lambda s: s.max() - s.min()),
    ).reset_index()

    # Distance is integral of speed over time (trapezoid rule)
    def total_distance(group: pd.DataFrame) -> float:
        g = group.sort_values("timestamp_s")
        return float(np.trapezoid(g["speed"], g["timestamp_s"]))

    dist_df = (
        trajectories.groupby(["segment_id", "object_id"])
        .apply(total_distance)
        .reset_index()
        .rename(columns={0: "distance_m"})
    )
    return summary.merge(dist_df, on=["segment_id", "object_id"])


# --- Speed deviation ---------------------------------------------------------

def speed_deviation(
    trajectories: pd.DataFrame,
    speed_limit_mps: float = 15.0,
) -> pd.DataFrame:
    """
    Compute per-frame speed deviation from a reference speed limit.

    speed_limit_mps: road speed limit in m/s (default 15 m/s ≈ 54 km/h).
    Adds columns: speed_excess (positive when over limit),
                  speed_deficit (positive when under limit by >30%).
    """
    df = trajectories.copy()
    df["speed_excess"] = (df["speed"] - speed_limit_mps).clip(lower=0)
    df["speed_deficit"] = (speed_limit_mps * 0.7 - df["speed"]).clip(lower=0)
    return df
