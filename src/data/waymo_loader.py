"""
Loads Waymo Open Dataset v2 parquet files and extracts agent trajectories.

The v2 dataset stores perception data in component-based parquet files:
  - lidar_box: 3D bounding boxes for detected agents
  - vehicle_pose: ego-vehicle world pose per frame

Columns follow the naming convention: key.* for index fields and
[ComponentName.field_name] for data fields.
"""

import os
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Object type codes used by the Waymo v2 dataset
OBJECT_TYPES = {1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "sign"}

# Canonical column aliases — map from v2 parquet names to short names used
# throughout this project.  The loader renames on read so downstream code
# never has to reference raw column names.
LIDAR_BOX_COLUMN_MAP = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_us",
    "key.laser_object_id": "object_id",
    "[LiDARBoxComponent].type": "object_type",
    "[LiDARBoxComponent].box.center.x": "x",
    "[LiDARBoxComponent].box.center.y": "y",
    "[LiDARBoxComponent].box.center.z": "z",
    "[LiDARBoxComponent].box.size.x": "length",
    "[LiDARBoxComponent].box.size.y": "width",
    "[LiDARBoxComponent].box.size.z": "height",
    "[LiDARBoxComponent].box.heading": "heading",
    "[LiDARBoxComponent].speed.x": "vx",
    "[LiDARBoxComponent].speed.y": "vy",
    "[LiDARBoxComponent].num_lidar_points_in_box": "lidar_points",
}

VEHICLE_POSE_COLUMN_MAP = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_us",
    "[VehiclePoseComponent].world_from_vehicle.transform": "ego_transform",
}


def _discover_columns(df: pd.DataFrame, expected_map: dict) -> dict:
    """Return a subset of expected_map whose keys actually exist in df."""
    return {k: v for k, v in expected_map.items() if k in df.columns}


def load_lidar_box(path: str) -> pd.DataFrame:
    """Read a single lidar_box parquet file and return a tidy DataFrame."""
    df = pd.read_parquet(path)
    col_map = _discover_columns(df, LIDAR_BOX_COLUMN_MAP)
    df = df.rename(columns=col_map)[list(col_map.values())]

    df["timestamp_s"] = df["timestamp_us"] / 1e6
    df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    df["object_type_name"] = df["object_type"].map(OBJECT_TYPES).fillna("unknown")
    return df


def load_vehicle_pose(path: str) -> pd.DataFrame:
    """Read a single vehicle_pose parquet file and return a tidy DataFrame."""
    df = pd.read_parquet(path)
    col_map = _discover_columns(df, VEHICLE_POSE_COLUMN_MAP)
    df = df.rename(columns=col_map)[list(col_map.values())]
    df["timestamp_s"] = df["timestamp_us"] / 1e6
    return df


def load_segment(segment_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all available components for a single segment directory.

    Returns a dict with keys 'lidar_box' and/or 'vehicle_pose', each a
    DataFrame sorted by timestamp.
    """
    result: dict[str, pd.DataFrame] = {}
    segment_dir = Path(segment_dir)

    lb_files = sorted(segment_dir.glob("lidar_box*.parquet"))
    if lb_files:
        frames = [load_lidar_box(str(f)) for f in lb_files]
        result["lidar_box"] = pd.concat(frames, ignore_index=True).sort_values(
            ["segment_id", "timestamp_us"]
        )

    vp_files = sorted(segment_dir.glob("vehicle_pose*.parquet"))
    if vp_files:
        frames = [load_vehicle_pose(str(f)) for f in vp_files]
        result["vehicle_pose"] = pd.concat(frames, ignore_index=True).sort_values(
            ["segment_id", "timestamp_us"]
        )

    return result


def load_dataset(data_dir: str, max_segments: Optional[int] = None) -> dict[str, pd.DataFrame]:
    """
    Walk a dataset root directory and load all segments into two merged DataFrames.

    data_dir structure expected:
        data_dir/
          <split>/
            lidar_box/
              *.parquet
            vehicle_pose/
              *.parquet
    """
    data_dir = Path(data_dir)
    lb_files = sorted(data_dir.rglob("lidar_box/*.parquet")) or sorted(data_dir.rglob("lidar_box*.parquet"))
    vp_files = sorted(data_dir.rglob("vehicle_pose/*.parquet")) or sorted(data_dir.rglob("vehicle_pose*.parquet"))

    if max_segments:
        lb_files = lb_files[:max_segments]
        vp_files = vp_files[:max_segments]

    lidar_frames, pose_frames = [], []

    for f in lb_files:
        try:
            lidar_frames.append(load_lidar_box(str(f)))
        except Exception as exc:
            print(f"[WARN] Could not load {f}: {exc}")

    for f in vp_files:
        try:
            pose_frames.append(load_vehicle_pose(str(f)))
        except Exception as exc:
            print(f"[WARN] Could not load {f}: {exc}")

    result: dict[str, pd.DataFrame] = {}
    if lidar_frames:
        result["lidar_box"] = pd.concat(lidar_frames, ignore_index=True)
    if pose_frames:
        result["vehicle_pose"] = pd.concat(pose_frames, ignore_index=True)
    return result


def extract_trajectories(lidar_box_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert frame-level detections into per-object trajectory DataFrames.

    Adds acceleration and jerk columns derived from finite differences of
    velocity and speed over time within each (segment_id, object_id) track.
    """
    df = lidar_box_df.copy().sort_values(["segment_id", "object_id", "timestamp_us"])

    grp = df.groupby(["segment_id", "object_id"])

    df["dt"] = grp["timestamp_s"].diff()
    df["ax"] = grp["vx"].diff() / df["dt"]
    df["ay"] = grp["vy"].diff() / df["dt"]
    df["acceleration"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2)
    df["jerk"] = grp["acceleration"].diff() / df["dt"]

    # Heading rate (rad/s)
    df["heading_rate"] = grp["heading"].diff() / df["dt"]

    # Drop the first row of each track (no previous frame to diff against)
    df = df.dropna(subset=["dt"])
    return df.reset_index(drop=True)
