"""
ML model evaluation for AV safety prediction.

Trains a trajectory-based TTC prediction model (predicting next-frame minimum
TTC from current kinematic features) and evaluates it with a full suite of
regression and calibration metrics.

This mirrors what a Waymo Data Scientist does when evaluating large-scale ML
models: checking not just accuracy but calibration, coverage, and failure modes.

Pipeline
--------
1. Feature engineering from trajectory + TTC data
2. Train/validation/test split (by segment to prevent data leakage)
3. Baseline: Ridge regression
4. Improved: Random Forest
5. Evaluation: RMSE, MAE, R², calibration curve, reliability diagram, ECE
6. Error analysis: where does the model fail?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "mean_speed", "std_speed",
    "mean_acceleration", "max_acceleration",
    "mean_jerk", "max_jerk",
    "mean_heading_rate",
    "mean_nearest_dist", "min_nearest_dist",
    "n_agents",
    "current_min_ttc",
]


def build_prediction_dataset(
    trajectories: pd.DataFrame,
    ttc_df: pd.DataFrame,
    horizon_frames: int = 5,
) -> pd.DataFrame:
    """
    Build a supervised dataset: predict TTC `horizon_frames` ahead.

    Each row is a (segment, timestamp) unit with:
      - Features: kinematic aggregates over the past window
      - Target: min_ttc `horizon_frames` frames into the future

    Segment-level grouping is preserved so we can do proper
    group-based train/test splitting to avoid leakage.
    """
    # Per-frame kinematic aggregates across all agents
    frame_agg = (
        trajectories
        .groupby(["segment_id", "timestamp_us"])
        .agg(
            mean_speed=("speed", "mean"),
            std_speed=("speed", "std"),
            mean_acceleration=("acceleration", "mean"),
            max_acceleration=("acceleration", "max"),
            mean_jerk=("jerk", lambda s: s.abs().mean()),
            max_jerk=("jerk", lambda s: s.abs().max()),
            mean_heading_rate=("heading_rate", lambda s: s.abs().mean()),
            n_agents=("object_id", "count"),
        )
        .reset_index()
    )

    # Add proximity if available
    if "nearest_agent_dist" in trajectories.columns:
        prox = (
            trajectories
            .groupby(["segment_id", "timestamp_us"])["nearest_agent_dist"]
            .agg(mean_nearest_dist="mean", min_nearest_dist="min")
            .reset_index()
        )
        frame_agg = frame_agg.merge(prox, on=["segment_id", "timestamp_us"])
    else:
        frame_agg["mean_nearest_dist"] = np.nan
        frame_agg["min_nearest_dist"] = np.nan

    # Merge current TTC
    ttc_finite = ttc_df.copy()
    ttc_finite["min_ttc_clipped"] = ttc_finite["min_ttc"].replace(np.inf, 20.0).clip(0, 20)
    frame_agg = frame_agg.merge(
        ttc_finite[["segment_id", "timestamp_us", "min_ttc_clipped"]].rename(
            columns={"min_ttc_clipped": "current_min_ttc"}
        ),
        on=["segment_id", "timestamp_us"], how="left",
    )

    # Create future TTC target by shifting within each segment
    frame_agg = frame_agg.sort_values(["segment_id", "timestamp_us"])
    frame_agg["future_min_ttc"] = (
        frame_agg.groupby("segment_id")["current_min_ttc"]
        .shift(-horizon_frames)
    )

    return frame_agg.dropna(subset=["future_min_ttc"] + ["mean_speed"]).reset_index(drop=True)


def segment_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by segment to prevent temporal data leakage."""
    segments = df["segment_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(segments)

    n = len(segments)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))

    test_segs = segments[:n_test]
    val_segs = segments[n_test: n_test + n_val]
    train_segs = segments[n_test + n_val:]

    return (
        df[df["segment_id"].isin(train_segs)],
        df[df["segment_id"].isin(val_segs)],
        df[df["segment_id"].isin(test_segs)],
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

@dataclass
class TTCPredictorResult:
    model_name: str
    rmse: float
    mae: float
    r2: float
    predictions: np.ndarray
    targets: np.ndarray
    feature_importance: Optional[pd.Series]
    model: object

    def summary(self) -> str:
        return (
            f"{self.model_name}: RMSE={self.rmse:.3f}s  MAE={self.mae:.3f}s  R²={self.r2:.4f}"
        )


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Optional[list[str]] = None,
    target: str = "future_min_ttc",
) -> list[TTCPredictorResult]:
    """
    Train Ridge, RandomForest, and GradientBoosting models and evaluate each.
    """
    if features is None:
        features = [f for f in FEATURE_COLS if f in train_df.columns]

    X_train = train_df[features].fillna(0).values
    y_train = train_df[target].values
    X_test = test_df[features].fillna(0).values
    y_test = test_df[target].values

    models = {
        "Ridge (baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100, max_depth=8,
                random_state=42, n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42,
            )),
        ]),
    }

    results = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        # Feature importance (tree models only)
        inner = pipeline.named_steps["model"]
        if hasattr(inner, "feature_importances_"):
            fi = pd.Series(inner.feature_importances_, index=features).sort_values(ascending=False)
        else:
            fi = None

        results.append(TTCPredictorResult(
            model_name=name, rmse=rmse, mae=mae, r2=r2,
            predictions=preds, targets=y_test,
            feature_importance=fi, model=pipeline,
        ))
        print(f"  {name}: RMSE={rmse:.3f}s  MAE={mae:.3f}s  R²={r2:.4f}")

    return results


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ECE for regression using equal-width bins of predicted values.

    ECE = Σ (|Bₖ|/n) * |mean_error(Bₖ)|

    Returns (ece, bin_centers, bin_mean_error, bin_counts).
    """
    bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_errors = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for k in range(n_bins):
        mask = bin_indices == k
        bin_counts[k] = mask.sum()
        if mask.sum() > 0:
            bin_errors[k] = float(np.abs(y_true[mask] - y_pred[mask]).mean())

    ece = float(np.sum(bin_counts / len(y_true) * bin_errors))
    return ece, bin_centers, bin_errors, bin_counts


def reliability_diagram_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build data for a reliability diagram (predicted quantile vs actual quantile).
    """
    quantiles = np.linspace(0.05, 0.95, n_bins)
    pred_q = np.quantile(y_pred, quantiles)
    true_q = np.quantile(y_true, quantiles)
    return pd.DataFrame({
        "quantile": quantiles,
        "predicted_quantile_value": pred_q,
        "actual_quantile_value": true_q,
    })


def error_analysis(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    target: str = "future_min_ttc",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Identify the frames with largest prediction error.

    Returns the `top_n` worst predictions with full feature context —
    useful for understanding where the model fails.
    """
    df = test_df.copy()
    df["prediction"] = predictions
    df["error"] = df["prediction"] - df[target]
    df["abs_error"] = df["error"].abs()
    return df.nlargest(top_n, "abs_error")[
        ["segment_id", "timestamp_us", target, "prediction", "error", "abs_error"]
        + [c for c in FEATURE_COLS if c in df.columns]
    ]
