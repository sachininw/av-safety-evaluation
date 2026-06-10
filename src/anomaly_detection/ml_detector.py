"""
ML-based anomaly detection for AV trajectory data.

Implements two unsupervised detectors that identify structurally unusual
driving patterns without requiring labelled anomaly examples:

  1. Isolation Forest — fast, tree-based; effective for high-dimensional data
  2. One-Class SVM (OC-SVM) — kernel-based; better for compact, well-defined
     normal regions

Both detectors are fitted on the normal portion of the data (by excluding
known rare-event frames if a label mask is provided) and then used to score
all frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


DEFAULT_FEATURES = [
    "speed",
    "acceleration",
    "jerk",
    "heading_rate",
    "nearest_agent_dist",
]


@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    contamination: float = 0.02
    random_state: int = 42
    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))


@dataclass
class OCSVMConfig:
    kernel: str = "rbf"
    nu: float = 0.02
    gamma: str = "scale"
    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))


def _prepare_features(
    df: pd.DataFrame,
    features: list[str],
    normal_mask: Optional[pd.Series] = None,
) -> tuple[np.ndarray, pd.Index, RobustScaler]:
    present = [f for f in features if f in df.columns]
    X = df[present].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
    scaler = RobustScaler()
    train_X = X[normal_mask.values] if normal_mask is not None else X
    scaler.fit(train_X)
    return scaler.transform(X), df.index, scaler


def isolation_forest_detector(
    trajectories: pd.DataFrame,
    config: Optional[IsolationForestConfig] = None,
    normal_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Run Isolation Forest and annotate each row with anomaly score and label.

    Parameters
    ----------
    trajectories : DataFrame with per-frame agent features
    config       : IsolationForestConfig
    normal_mask  : boolean Series (same index as trajectories) where True
                   marks known-normal frames used for fitting

    Added columns:
      - if_score     : raw anomaly score (lower = more anomalous)
      - if_anomaly   : True for predicted anomalies
    """
    cfg = config or IsolationForestConfig()
    X, idx, _ = _prepare_features(trajectories, cfg.features, normal_mask)

    clf = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    clf.fit(X[normal_mask.values] if normal_mask is not None else X)

    df = trajectories.copy()
    df["if_score"] = clf.score_samples(X)
    df["if_anomaly"] = clf.predict(X) == -1
    return df


def ocsvm_detector(
    trajectories: pd.DataFrame,
    config: Optional[OCSVMConfig] = None,
    normal_mask: Optional[pd.Series] = None,
    max_train_samples: int = 50_000,
) -> pd.DataFrame:
    """
    Run One-Class SVM and annotate each row.

    OC-SVM is sensitive to dataset size — we subsample the training set if
    needed to keep fit time tractable.

    Added columns:
      - ocsvm_score   : decision function score (more negative = more anomalous)
      - ocsvm_anomaly : True for predicted anomalies
    """
    cfg = config or OCSVMConfig()
    X, idx, _ = _prepare_features(trajectories, cfg.features, normal_mask)

    train_X = X[normal_mask.values] if normal_mask is not None else X
    if len(train_X) > max_train_samples:
        rng = np.random.default_rng(42)
        train_idx = rng.choice(len(train_X), size=max_train_samples, replace=False)
        train_X = train_X[train_idx]

    clf = OneClassSVM(kernel=cfg.kernel, nu=cfg.nu, gamma=cfg.gamma)
    clf.fit(train_X)

    df = trajectories.copy()
    df["ocsvm_score"] = clf.decision_function(X)
    df["ocsvm_anomaly"] = clf.predict(X) == -1
    return df


def run_ml_detectors(
    trajectories: pd.DataFrame,
    normal_mask: Optional[pd.Series] = None,
    use_ocsvm: bool = True,
) -> pd.DataFrame:
    """
    Apply all ML detectors and produce a combined 'ml_anomaly' flag.

    OC-SVM can be slow on large datasets; set use_ocsvm=False to skip it.
    """
    df = isolation_forest_detector(trajectories, normal_mask=normal_mask)
    if use_ocsvm:
        df = ocsvm_detector(df, normal_mask=normal_mask)
        df["ml_anomaly"] = df["if_anomaly"] | df["ocsvm_anomaly"]
    else:
        df["ml_anomaly"] = df["if_anomaly"]
    return df


def anomaly_feature_importance(
    trajectories: pd.DataFrame,
    features: Optional[list[str]] = None,
    config: Optional[IsolationForestConfig] = None,
) -> pd.Series:
    """
    Estimate which features drive anomaly detection using mean depth decrease.

    Returns a Series of feature importance scores (higher = more important).
    """
    cfg = config or IsolationForestConfig()
    feature_list = features or cfg.features
    present = [f for f in feature_list if f in trajectories.columns]

    X, _, scaler = _prepare_features(trajectories, present)
    clf = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
    )
    clf.fit(X)

    importances = np.mean(
        [tree.feature_importances_ for tree in clf.estimators_], axis=0
    )
    return pd.Series(importances, index=present).sort_values(ascending=False)
