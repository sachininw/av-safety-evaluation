"""
Detects and classifies rare safety-critical events from AV trajectories.

Events classified:
  - Near-miss: min TTC < 3 s between any two agents
  - Hard braking: deceleration > 4 m/s²
  - Emergency stop: deceleration > 6 m/s²
  - Sudden swerve: heading rate > 0.5 rad/s
  - Pedestrian encroachment: pedestrian within 3 m of any vehicle
  - Cluster: consecutive frames of the same event type (de-duplication)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


NEAR_MISS_TTC_S = 3.0
HARD_BRAKE_THRESHOLD_MPS2 = 4.0
EMERGENCY_STOP_THRESHOLD_MPS2 = 6.0
SWERVE_THRESHOLD_RADS = 0.5
PEDESTRIAN_PROXIMITY_M = 3.0


@dataclass
class RareEvent:
    event_type: str
    segment_id: str
    timestamp_us: int
    object_id: Optional[str]
    severity: str         # "warning" | "critical"
    value: float          # metric value that triggered the event
    threshold: float      # threshold that was exceeded


def detect_hard_braking(trajectories: pd.DataFrame) -> list[RareEvent]:
    events = []
    mask = trajectories["acceleration"] > HARD_BRAKE_THRESHOLD_MPS2
    for _, row in trajectories[mask].iterrows():
        severity = "critical" if row["acceleration"] > EMERGENCY_STOP_THRESHOLD_MPS2 else "warning"
        events.append(RareEvent(
            event_type="hard_braking" if severity == "warning" else "emergency_stop",
            segment_id=row["segment_id"],
            timestamp_us=int(row["timestamp_us"]),
            object_id=str(row.get("object_id", "")),
            severity=severity,
            value=float(row["acceleration"]),
            threshold=HARD_BRAKE_THRESHOLD_MPS2,
        ))
    return events


def detect_sudden_swerve(trajectories: pd.DataFrame) -> list[RareEvent]:
    events = []
    mask = trajectories["heading_rate"].abs() > SWERVE_THRESHOLD_RADS
    for _, row in trajectories[mask].iterrows():
        events.append(RareEvent(
            event_type="sudden_swerve",
            segment_id=row["segment_id"],
            timestamp_us=int(row["timestamp_us"]),
            object_id=str(row.get("object_id", "")),
            severity="warning",
            value=float(abs(row["heading_rate"])),
            threshold=SWERVE_THRESHOLD_RADS,
        ))
    return events


def detect_pedestrian_encroachment(
    trajectories: pd.DataFrame,
) -> list[RareEvent]:
    """Flag frames where a pedestrian is within PEDESTRIAN_PROXIMITY_M of any vehicle."""
    events = []
    vehicles = trajectories[trajectories["object_type_name"] == "vehicle"]
    pedestrians = trajectories[trajectories["object_type_name"] == "pedestrian"]

    if vehicles.empty or pedestrians.empty:
        return events

    for (seg, ts), v_frame in vehicles.groupby(["segment_id", "timestamp_us"]):
        p_frame = pedestrians[
            (pedestrians["segment_id"] == seg) & (pedestrians["timestamp_us"] == ts)
        ]
        if p_frame.empty:
            continue
        for _, ped in p_frame.iterrows():
            dists = np.sqrt(
                (v_frame["x"] - ped["x"]) ** 2 + (v_frame["y"] - ped["y"]) ** 2
            )
            if dists.min() < PEDESTRIAN_PROXIMITY_M:
                events.append(RareEvent(
                    event_type="pedestrian_encroachment",
                    segment_id=seg,
                    timestamp_us=int(ts),
                    object_id=str(ped.get("object_id", "")),
                    severity="critical",
                    value=float(dists.min()),
                    threshold=PEDESTRIAN_PROXIMITY_M,
                ))
    return events


def detect_near_misses(ttc_df: pd.DataFrame) -> list[RareEvent]:
    """Convert the TTC DataFrame (from batch_ttc) into RareEvent objects."""
    events = []
    mask = ttc_df["near_miss"]
    for _, row in ttc_df[mask].iterrows():
        severity = "critical" if row["min_ttc"] < 1.5 else "warning"
        events.append(RareEvent(
            event_type="near_miss",
            segment_id=row["segment_id"],
            timestamp_us=int(row["timestamp_us"]),
            object_id=None,
            severity=severity,
            value=float(row["min_ttc"]),
            threshold=NEAR_MISS_TTC_S,
        ))
    return events


def cluster_events(events: list[RareEvent], gap_us: int = 500_000) -> list[RareEvent]:
    """
    Merge consecutive events of the same type within gap_us microseconds.

    Keeps only the most severe event in each cluster to avoid double-counting.
    """
    if not events:
        return []

    df = pd.DataFrame([vars(e) for e in events])
    df = df.sort_values(["segment_id", "event_type", "timestamp_us"])

    clustered = []
    for (seg, etype), group in df.groupby(["segment_id", "event_type"]):
        group = group.sort_values("timestamp_us").reset_index(drop=True)
        cluster_start = 0
        for i in range(1, len(group)):
            gap = group.loc[i, "timestamp_us"] - group.loc[i - 1, "timestamp_us"]
            if gap > gap_us or i == len(group) - 1:
                cluster = group.loc[cluster_start : i]
                # Keep the row with the extreme value (max for braking, min for TTC)
                if etype in ("near_miss",):
                    rep = cluster.loc[cluster["value"].idxmin()]
                else:
                    rep = cluster.loc[cluster["value"].idxmax()]
                clustered.append(RareEvent(**rep.to_dict()))
                cluster_start = i
    return clustered


def run_all_detectors(
    trajectories: pd.DataFrame,
    ttc_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run all rare-event detectors and return a tidy events DataFrame.

    Parameters
    ----------
    trajectories : output of waymo_loader.extract_trajectories
    ttc_df       : optional output of safety_metrics.batch_ttc (computed
                   externally to avoid recomputation)
    """
    events: list[RareEvent] = []
    events.extend(detect_hard_braking(trajectories))
    events.extend(detect_sudden_swerve(trajectories))
    events.extend(detect_pedestrian_encroachment(trajectories))
    if ttc_df is not None:
        events.extend(detect_near_misses(ttc_df))

    events = cluster_events(events)

    if not events:
        return pd.DataFrame(columns=["event_type", "segment_id", "timestamp_us",
                                     "object_id", "severity", "value", "threshold"])
    df = pd.DataFrame([vars(e) for e in events])
    df["timestamp_s"] = df["timestamp_us"] / 1e6
    return df.sort_values(["segment_id", "timestamp_us"]).reset_index(drop=True)
