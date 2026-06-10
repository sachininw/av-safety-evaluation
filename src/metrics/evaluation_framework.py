"""
Top-level evaluation framework: orchestrates all safety and comfort metrics
and produces a structured report for a dataset or segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.metrics.safety_metrics import (
    batch_ttc,
    comfort_metrics,
    nearest_agent_distances,
    speed_deviation,
)


@dataclass
class EvaluationConfig:
    near_miss_ttc_threshold_s: float = 3.0
    critical_ttc_threshold_s: float = 1.5
    speed_limit_mps: float = 15.0
    max_acceptable_jerk_mps3: float = 6.0
    max_acceptable_accel_mps2: float = 4.0
    proximity_warning_m: float = 5.0


@dataclass
class SegmentReport:
    segment_id: str
    n_frames: int
    n_agents: int
    object_type_counts: dict

    # TTC statistics
    min_ttc_overall: float
    mean_min_ttc: float
    near_miss_count: int
    near_miss_rate: float

    # Comfort statistics
    mean_jerk: float
    max_jerk: float
    mean_accel: float
    max_accel: float
    jerk_violation_rate: float
    accel_violation_rate: float

    # Speed statistics
    mean_speed_mps: float
    speed_excess_rate: float

    # Proximity
    mean_nearest_dist: float
    min_nearest_dist: float
    proximity_warning_rate: float

    config: EvaluationConfig = field(default_factory=EvaluationConfig, repr=False)


@dataclass
class DatasetReport:
    n_segments: int
    segment_reports: list[SegmentReport]
    config: EvaluationConfig

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for r in self.segment_reports:
            rows.append({
                "segment_id": r.segment_id,
                "n_frames": r.n_frames,
                "n_agents": r.n_agents,
                "min_ttc_overall": r.min_ttc_overall,
                "mean_min_ttc": r.mean_min_ttc,
                "near_miss_count": r.near_miss_count,
                "near_miss_rate_pct": r.near_miss_rate * 100,
                "mean_jerk": r.mean_jerk,
                "max_jerk": r.max_jerk,
                "jerk_violation_rate_pct": r.jerk_violation_rate * 100,
                "mean_speed_mps": r.mean_speed_mps,
                "speed_excess_rate_pct": r.speed_excess_rate * 100,
                "mean_nearest_dist_m": r.mean_nearest_dist,
                "min_nearest_dist_m": r.min_nearest_dist,
                "proximity_warning_rate_pct": r.proximity_warning_rate * 100,
            })
        return pd.DataFrame(rows)


class AVEvaluationFramework:
    """
    Evaluate an autonomous vehicle dataset using a suite of safety and comfort metrics.

    Usage
    -----
    framework = AVEvaluationFramework(config)
    report = framework.evaluate_dataset(trajectories_df)
    print(report.summary_df())
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def evaluate_segment(
        self,
        segment_id: str,
        trajectories: pd.DataFrame,
    ) -> SegmentReport:
        cfg = self.config
        seg = trajectories[trajectories["segment_id"] == segment_id].copy()

        n_frames = seg["timestamp_us"].nunique()
        n_agents = seg["object_id"].nunique()
        object_type_counts = seg.groupby("object_type_name")["object_id"].nunique().to_dict()

        # TTC
        ttc_df = batch_ttc(seg)
        if ttc_df.empty:
            min_ttc_overall = np.inf
            mean_min_ttc = np.inf
            near_miss_count = 0
            near_miss_rate = 0.0
        else:
            finite_ttc = ttc_df["min_ttc"].replace(np.inf, np.nan).dropna()
            min_ttc_overall = float(finite_ttc.min()) if not finite_ttc.empty else np.inf
            mean_min_ttc = float(finite_ttc.mean()) if not finite_ttc.empty else np.inf
            near_miss_count = int(ttc_df["near_miss"].sum())
            near_miss_rate = near_miss_count / max(len(ttc_df), 1)

        # Comfort
        comfort = comfort_metrics(seg)
        mean_jerk = float(comfort["mean_jerk"].mean())
        max_jerk = float(comfort["max_jerk"].max())
        mean_accel = float(comfort["mean_acceleration"].mean())
        max_accel = float(comfort["max_acceleration"].max())
        jerk_violation_rate = float((seg["jerk"].abs() > cfg.max_acceptable_jerk_mps3).mean())
        accel_violation_rate = float((seg["acceleration"] > cfg.max_acceptable_accel_mps2).mean())

        # Speed
        seg_speed = speed_deviation(seg, cfg.speed_limit_mps)
        mean_speed_mps = float(seg["speed"].mean())
        speed_excess_rate = float((seg_speed["speed_excess"] > 0).mean())

        # Proximity
        seg_prox = nearest_agent_distances(seg)
        finite_prox = seg_prox["nearest_agent_dist"].replace(np.inf, np.nan).dropna()
        mean_nearest_dist = float(finite_prox.mean()) if not finite_prox.empty else np.inf
        min_nearest_dist = float(finite_prox.min()) if not finite_prox.empty else np.inf
        proximity_warning_rate = float(
            (seg_prox["nearest_agent_dist"] < cfg.proximity_warning_m).mean()
        )

        return SegmentReport(
            segment_id=segment_id,
            n_frames=n_frames,
            n_agents=n_agents,
            object_type_counts=object_type_counts,
            min_ttc_overall=min_ttc_overall,
            mean_min_ttc=mean_min_ttc,
            near_miss_count=near_miss_count,
            near_miss_rate=near_miss_rate,
            mean_jerk=mean_jerk,
            max_jerk=max_jerk,
            mean_accel=mean_accel,
            max_accel=max_accel,
            jerk_violation_rate=jerk_violation_rate,
            accel_violation_rate=accel_violation_rate,
            mean_speed_mps=mean_speed_mps,
            speed_excess_rate=speed_excess_rate,
            mean_nearest_dist=mean_nearest_dist,
            min_nearest_dist=min_nearest_dist,
            proximity_warning_rate=proximity_warning_rate,
            config=cfg,
        )

    def evaluate_dataset(self, trajectories: pd.DataFrame) -> DatasetReport:
        segments = trajectories["segment_id"].unique()
        reports = [self.evaluate_segment(seg_id, trajectories) for seg_id in segments]
        return DatasetReport(
            n_segments=len(segments),
            segment_reports=reports,
            config=self.config,
        )
