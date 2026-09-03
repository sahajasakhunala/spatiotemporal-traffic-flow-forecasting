"""
Historical Rolling Forecast Simulation (One-Step-Ahead Backtest).

Simulates deployment of the trained Random Forest model during the unseen
chronological test period (Aug 27-30, 2024) as a one-step-ahead rolling-origin
backtest.

FORECASTING SEMANTICS
---------------------
This is a ONE-STEP-AHEAD ROLLING-ORIGIN backtest, NOT a recursive multi-step
forecast.

At each forecast origin time t:
  1. Use observations available through time t (including the actual probe_count
     at t, which appears as lag_1 in the feature vector for predicting t+1).
  2. Predict traffic flow at t+1 using the pre-trained Random Forest model.
  3. Compare the prediction against the actual observed probe_count at t+1.
  4. Once the actual t+1 observation arrives in the historical record, it becomes
     available as lag_1 for the NEXT forecast (predicting t+2).
  5. Continue sequentially across the entire test period.

This means predictions are NEVER fed back into future lag features. Every lag
feature is populated with actual historical observations. This represents a
realistic operational forecasting system that receives updated traffic
observations every hour and uses them to produce the next one-hour-ahead
forecast.

ANTI-LEAKAGE GUARANTEES
-----------------------
- The Random Forest model (models/random_forest.joblib) was trained exclusively
  on Aug 11-26 data and is loaded without retraining.
- Segment historical statistics (segment_mean_traffic, segment_std_traffic,
  segment_p90_traffic, segment_zero_freq) were computed strictly on the
  Aug 11-26 training period and are NOT recomputed.
- Lag features at each test-period row use only probe_count values from time t
  or earlier. The feature engineering pipeline constructs lags via
  groupby("segment_id")["probe_count"].shift(N) on the chronologically sorted
  full dataset, so lag_1 at target hour t+1 equals probe_count at hour t.
- No future information is used at any point.

EXPECTED METRIC REPRODUCIBILITY
-------------------------------
Because this simulation reconstructs the identical feature matrix and evaluates
the same test rows used by the original benchmark, the overall metrics should
reproduce:
  MAE  = 17.8112
  RMSE = 36.2533
  R^2  = 0.9683

If they differ, the difference is investigated and documented rather than
forced to match.

Usage:
    python src/forecast_simulation.py
"""
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project imports
from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    VISUALIZATIONS_DIR,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    MORNING_RUSH_HOURS,
    EVENING_RUSH_HOURS,
    FRC_MAP,
)
from src.feature_engineering import (
    load_dataset_range,
    build_temporal_features,
    build_timestamp_aware_lags,
    compute_segment_historical_stats,
)
from src.model_training import FEATURE_COLUMNS, TARGET_COLUMN


# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
PLOT_DPI = 300
COLORS = {
    "rf": "#2196F3",
    "lag1": "#FF9800",
    "actual": "#4CAF50",
    "error": "#9C27B0",
    "morning": "#FF5722",
    "evening": "#3F51B5",
    "offpeak": "#607D8B",
}


# ---------------------------------------------------------------------------
# Helper: metrics
# ---------------------------------------------------------------------------
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    mean_actual = float(np.mean(y_true))
    # Weighted Absolute Percentage Error (WAPE) / Normalized MAE:
    # WAPE = sum(|y - y_hat|) / sum(y) = MAE / mean(y)
    # Robust against individual zero-flow records (~8.06% of observations)
    wape_pct = round((mae / mean_actual * 100.0), 2) if mean_actual > 0 else 0.0
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Mean_Bias": round(bias, 4),
        "Mean_Actual": round(mean_actual, 2),
        "WAPE_pct": wape_pct,
    }


# ---------------------------------------------------------------------------
# 1. Data preparation  (identical to run_pipeline.py)
# ---------------------------------------------------------------------------
def prepare_test_data() -> pd.DataFrame:
    """Reconstruct the feature-engineered test DataFrame identically to
    the original training pipeline."""

    print("[1/5] Loading partitioned Parquet data (Aug 11 - Aug 30)...")
    full_df = load_dataset_range(TRAIN_START_DATE, TEST_END_DATE)
    print(f"      Loaded {len(full_df):,} total rows "
          f"({full_df['segment_id'].nunique():,} segments)")

    print("[2/5] Building temporal features...")
    df_temporal = build_temporal_features(full_df)

    print("[3/5] Building timestamp-aware lag features...")
    df_featured = build_timestamp_aware_lags(df_temporal)

    # Chronological split
    train_mask = (
        (df_featured["date"] >= pd.to_datetime(TRAIN_START_DATE).date())
        & (df_featured["date"] <= pd.to_datetime(TRAIN_END_DATE).date())
    )
    test_mask = (
        (df_featured["date"] >= pd.to_datetime(TEST_START_DATE).date())
        & (df_featured["date"] <= pd.to_datetime(TEST_END_DATE).date())
    )

    train_df = df_featured[train_mask].copy()
    test_df = df_featured[test_mask].copy()

    # Load pre-computed segment stats from the saved training-only artifact
    seg_stats_path = MODELS_DIR / "segment_stats.parquet"
    if seg_stats_path.exists():
        print("[4/5] Loading training-period segment statistics from "
              "models/segment_stats.parquet...")
        seg_stats = pd.read_parquet(seg_stats_path)
        stat_cols = [
            "segment_id", "segment_mean_traffic", "segment_std_traffic",
            "segment_p90_traffic", "segment_zero_freq",
        ]
        seg_stats = seg_stats[stat_cols]
    else:
        print("[4/5] Recomputing segment statistics from training split...")
        seg_stats = compute_segment_historical_stats(train_df)

    test_df = pd.merge(test_df, seg_stats, on="segment_id", how="left")

    # Fill cold-start segments with global training defaults
    test_df["segment_mean_traffic"] = test_df["segment_mean_traffic"].fillna(
        train_df["probe_count"].mean()
    )
    test_df["segment_std_traffic"] = test_df["segment_std_traffic"].fillna(
        train_df["probe_count"].std()
    )
    test_df["segment_p90_traffic"] = test_df["segment_p90_traffic"].fillna(
        train_df["probe_count"].quantile(0.9)
    )
    test_df["segment_zero_freq"] = test_df["segment_zero_freq"].fillna(0.1)

    # Drop rows with NaN in any feature column (first few hours of test period
    # may lack lag_24 if Aug 26 data was the last training day)
    clean_test = test_df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    print(f"[5/5] Clean test rows: {len(clean_test):,} "
          f"(dropped {len(test_df) - len(clean_test):,} rows with NaN lags)")

    return clean_test


# ---------------------------------------------------------------------------
# 2. Run simulation
# ---------------------------------------------------------------------------
def run_simulation(test_df: pd.DataFrame) -> pd.DataFrame:
    """Execute the one-step-ahead rolling-origin backtest."""

    # --- Load pre-trained Random Forest ---
    rf_path = MODELS_DIR / "random_forest.joblib"
    if not rf_path.exists():
        print(f"ERROR: Model artifact not found at {rf_path}")
        sys.exit(1)

    print(f"\nLoading pre-trained Random Forest from {rf_path}...")
    rf = joblib.load(rf_path)
    assert rf.n_features_in_ == len(FEATURE_COLUMNS), (
        f"Feature count mismatch: model expects {rf.n_features_in_}, "
        f"FEATURE_COLUMNS has {len(FEATURE_COLUMNS)}"
    )

    # --- Prepare feature matrix ---
    X = test_df[FEATURE_COLUMNS].values
    y_actual = test_df[TARGET_COLUMN].values

    # --- Generate predictions ---
    print(f"Running RF predictions on {len(X):,} test observations...")
    t0 = time.time()
    y_pred_raw = rf.predict(X)
    # Clamp to non-negative (probe counts cannot be negative)
    y_pred = np.maximum(y_pred_raw, 0.0)
    pred_time = time.time() - t0
    print(f"Predictions completed in {pred_time:.1f}s")

    # --- Build results DataFrame ---
    results_df = test_df[
        ["segment_id", "street_name", "date", "hour", "datetime",
         "frc", "speed_limit", "distance",
         "probe_count", "probe_count_lag_1"]
    ].copy()

    # Forecast origin = the hour whose information was used (t)
    # Forecast target = the hour being predicted (t+1 = the row's own hour)
    # Because lag_1 at row t+1 equals probe_count at t, the forecast origin
    # is one hour before the row's datetime.
    results_df["forecast_origin_datetime"] = (
        results_df["datetime"] - pd.Timedelta(hours=1)
    )
    results_df["forecast_datetime"] = results_df["datetime"]
    results_df["actual_probe_count"] = y_actual
    results_df["rf_predicted"] = np.round(y_pred, 1)
    results_df["lag1_predicted"] = results_df["probe_count_lag_1"]

    # Error columns
    results_df["rf_error"] = results_df["rf_predicted"] - results_df["actual_probe_count"]
    results_df["rf_abs_error"] = np.abs(results_df["rf_error"])
    results_df["rf_squared_error"] = results_df["rf_error"] ** 2
    results_df["lag1_error"] = results_df["lag1_predicted"] - results_df["actual_probe_count"]
    results_df["lag1_abs_error"] = np.abs(results_df["lag1_error"])

    return results_df


# ---------------------------------------------------------------------------
# 3. Compute breakdowns
# ---------------------------------------------------------------------------
def compute_breakdowns(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all dimensional performance breakdowns."""

    y_actual = df["actual_probe_count"].values
    y_rf = df["rf_predicted"].values
    y_lag1 = df["lag1_predicted"].values

    report: Dict[str, Any] = {}

    # --- Overall ---
    report["overall"] = {
        "n_forecasts": len(df),
        "random_forest": calc_metrics(y_actual, y_rf),
        "naive_lag1": calc_metrics(y_actual, y_lag1),
    }
    rf_mae = report["overall"]["random_forest"]["MAE"]
    lag1_mae = report["overall"]["naive_lag1"]["MAE"]
    report["overall"]["mae_improvement_vs_lag1_pct"] = round(
        (1 - rf_mae / lag1_mae) * 100, 1
    )

    # --- By Hour (0-23) ---
    hourly = {}
    for h in range(24):
        mask = df["hour"] == h
        if mask.sum() == 0:
            continue
        hourly[str(h)] = {
            "n": int(mask.sum()),
            "random_forest": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "rf_predicted"].values,
            ),
            "naive_lag1": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "lag1_predicted"].values,
            ),
        }
    report["by_hour"] = hourly

    # --- By Traffic Period ---
    morning_mask = df["hour"].isin(MORNING_RUSH_HOURS)
    evening_mask = df["hour"].isin(EVENING_RUSH_HOURS)
    offpeak_mask = ~(morning_mask | evening_mask)

    report["by_period"] = {}
    for name, mask in [
        ("morning_rush", morning_mask),
        ("evening_rush", evening_mask),
        ("off_peak", offpeak_mask),
    ]:
        if mask.sum() == 0:
            continue
        report["by_period"][name] = {
            "hours": (
                MORNING_RUSH_HOURS if "morning" in name
                else EVENING_RUSH_HOURS if "evening" in name
                else "all other hours"
            ),
            "n": int(mask.sum()),
            "random_forest": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "rf_predicted"].values,
            ),
            "naive_lag1": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "lag1_predicted"].values,
            ),
        }

    # --- By FRC ---
    frc_report = {}
    for frc_val in sorted(df["frc"].unique()):
        mask = df["frc"] == frc_val
        if mask.sum() == 0:
            continue
        frc_label = FRC_MAP.get(int(frc_val), f"FRC {int(frc_val)}")
        frc_report[f"FRC_{int(frc_val)}"] = {
            "label": frc_label,
            "n": int(mask.sum()),
            "random_forest": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "rf_predicted"].values,
            ),
            "naive_lag1": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "lag1_predicted"].values,
            ),
        }
    report["by_frc"] = frc_report

    # --- By Traffic Volume Quartile ---
    # Bin by actual traffic volume into quartiles
    # Q1: 0 - 25th percentile, Q2: 25 - 50th, Q3: 50 - 75th, Q4: 75 - 100th
    q25 = float(np.percentile(y_actual, 25))
    q50 = float(np.percentile(y_actual, 50))
    q75 = float(np.percentile(y_actual, 75))
    max_val = float(np.max(y_actual))

    quartile_bins = [
        ("Q1_Low_Volume", (y_actual <= q25), f"0 to {q25:.0f}"),
        ("Q2_Medium_Low", (y_actual > q25) & (y_actual <= q50), f"{q25+1:.0f} to {q50:.0f}"),
        ("Q3_Medium_High", (y_actual > q50) & (y_actual <= q75), f"{q50+1:.0f} to {q75:.0f}"),
        ("Q4_High_Volume", (y_actual > q75), f">{q75:.0f} (max {max_val:.0f})"),
    ]

    vol_report = {}
    for q_name, mask, range_str in quartile_bins:
        if mask.sum() == 0:
            continue
        vol_report[q_name] = {
            "flow_range": range_str,
            "n": int(mask.sum()),
            "random_forest": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "rf_predicted"].values,
            ),
            "naive_lag1": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "lag1_predicted"].values,
            ),
        }
    report["by_volume_quartile"] = vol_report

    # --- By Day ---
    daily = {}
    for d in sorted(df["date"].unique()):
        mask = df["date"] == d
        if mask.sum() == 0:
            continue
        daily[str(d)] = {
            "n": int(mask.sum()),
            "random_forest": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "rf_predicted"].values,
            ),
            "naive_lag1": calc_metrics(
                df.loc[mask, "actual_probe_count"].values,
                df.loc[mask, "lag1_predicted"].values,
            ),
        }
    report["by_day"] = daily

    # --- Segment-level errors ---
    seg_errors = (
        df.groupby("segment_id")
        .agg(
            street_name=("street_name", "first"),
            frc=("frc", "first"),
            n_forecasts=("rf_abs_error", "count"),
            mean_actual=("actual_probe_count", "mean"),
            mean_predicted=("rf_predicted", "mean"),
            mae=("rf_abs_error", "mean"),
            rmse=("rf_squared_error", lambda x: float(np.sqrt(x.mean()))),
            mean_signed_error=("rf_error", "mean"),
        )
        .reset_index()
        .sort_values("mae")
    )

    best_10 = seg_errors.head(10)
    worst_10 = seg_errors.tail(10).sort_values("mae", ascending=False)

    report["segment_analysis"] = {
        "total_segments_evaluated": len(seg_errors),
        "best_10_segments": best_10.to_dict(orient="records"),
        "worst_10_segments": worst_10.to_dict(orient="records"),
    }

    return report


# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------
def select_representative_segments(df: pd.DataFrame, n: int = 5) -> List:
    """Select representative high-flow road segments for timeline plots.

    Strategy: pick segments with above-median mean traffic, spread across
    different FRC classes where possible, preferring segments with known
    street names.
    """
    seg_summary = (
        df.groupby("segment_id")
        .agg(
            mean_flow=("actual_probe_count", "mean"),
            frc=("frc", "first"),
            street_name=("street_name", "first"),
        )
        .reset_index()
    )
    # Filter to above-median flow
    median_flow = seg_summary["mean_flow"].median()
    high_flow = seg_summary[seg_summary["mean_flow"] > median_flow].copy()

    # Prefer named streets
    named = high_flow[high_flow["street_name"] != "Unknown"]
    pool = named if len(named) >= n else high_flow

    # Try to get one per FRC
    selected = []
    for frc_val in sorted(pool["frc"].unique()):
        frc_pool = pool[pool["frc"] == frc_val].sort_values(
            "mean_flow", ascending=False
        )
        if len(frc_pool) > 0 and len(selected) < n:
            selected.append(frc_pool.iloc[0]["segment_id"])

    # Fill remaining from top flow
    remaining = pool[~pool["segment_id"].isin(selected)].sort_values(
        "mean_flow", ascending=False
    )
    for _, row in remaining.iterrows():
        if len(selected) >= n:
            break
        selected.append(row["segment_id"])

    return selected


def generate_visualizations(df: pd.DataFrame, report: Dict[str, Any],
                            viz_dir: Path) -> None:
    """Generate all 7 simulation diagnostic plots."""

    viz_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Actual vs Predicted timeline for representative segments ----
    print("  Generating rolling_forecast_actual_vs_predicted.png...")
    rep_segments = select_representative_segments(df, n=4)
    fig, axes = plt.subplots(len(rep_segments), 1,
                             figsize=(16, 4 * len(rep_segments)),
                             sharex=False)
    if len(rep_segments) == 1:
        axes = [axes]

    for ax, seg_id in zip(axes, rep_segments):
        seg_data = df[df["segment_id"] == seg_id].sort_values("forecast_datetime")
        # Show first 48 hours
        seg_48h = seg_data.head(48)
        street = seg_48h["street_name"].iloc[0]
        frc_val = int(seg_48h["frc"].iloc[0])

        ax.plot(seg_48h["forecast_datetime"], seg_48h["actual_probe_count"],
                color=COLORS["actual"], linewidth=2, label="Actual", marker="o",
                markersize=3)
        ax.plot(seg_48h["forecast_datetime"], seg_48h["rf_predicted"],
                color=COLORS["rf"], linewidth=2, label="RF Predicted",
                linestyle="--", marker="s", markersize=3)
        ax.set_ylabel("Probe Flow")
        ax.set_title(f"{street} (FRC {frc_val})", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Rolling One-Step-Ahead Forecast: Actual vs Random Forest",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(viz_dir / "rolling_forecast_actual_vs_predicted.png",
                dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # ---- 2. Actual vs RF vs Lag-1 comparison ----
    print("  Generating rolling_forecast_vs_lag1.png...")
    top_seg = rep_segments[0]
    seg_data = df[df["segment_id"] == top_seg].sort_values("forecast_datetime")
    seg_48h = seg_data.head(48)
    street = seg_48h["street_name"].iloc[0]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(seg_48h["forecast_datetime"], seg_48h["actual_probe_count"],
            color=COLORS["actual"], linewidth=2.5, label="Actual", marker="o",
            markersize=4)
    ax.plot(seg_48h["forecast_datetime"], seg_48h["rf_predicted"],
            color=COLORS["rf"], linewidth=2, label="Random Forest",
            linestyle="--", marker="s", markersize=3)
    ax.plot(seg_48h["forecast_datetime"], seg_48h["lag1_predicted"],
            color=COLORS["lag1"], linewidth=1.5, label="Naive Lag-1",
            linestyle=":", marker="^", markersize=3)
    ax.set_xlabel("Forecast Datetime")
    ax.set_ylabel("Probe Flow")
    ax.set_title(f"48-Hour Forecast Comparison: {street}\n"
                 f"Actual vs Random Forest vs Naive Lag-1 Persistence",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(viz_dir / "rolling_forecast_vs_lag1.png",
                dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # ---- 3. Hourly MAE ----
    print("  Generating simulation_hourly_mae.png...")
    hourly_data = report["by_hour"]
    hours = sorted([int(h) for h in hourly_data.keys()])
    rf_mae_h = [hourly_data[str(h)]["random_forest"]["MAE"] for h in hours]
    lag1_mae_h = [hourly_data[str(h)]["naive_lag1"]["MAE"] for h in hours]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(hours))
    w = 0.35
    bars_rf = ax.bar(x - w / 2, rf_mae_h, w, label="Random Forest",
                     color=COLORS["rf"], alpha=0.85)
    bars_lag = ax.bar(x + w / 2, lag1_mae_h, w, label="Naive Lag-1",
                      color=COLORS["lag1"], alpha=0.85)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE (Probe Flow Units)")
    ax.set_title("Simulation MAE by Hour of Day: Random Forest vs Naive Lag-1",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / "simulation_hourly_mae.png",
                dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # ---- 4. Period performance ----
    print("  Generating simulation_period_performance.png...")
    period_data = report["by_period"]
    periods = list(period_data.keys())
    period_labels = [p.replace("_", " ").title() for p in periods]
    rf_mae_p = [period_data[p]["random_forest"]["MAE"] for p in periods]
    lag1_mae_p = [period_data[p]["naive_lag1"]["MAE"] for p in periods]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(periods))
    w = 0.35
    ax.bar(x - w / 2, rf_mae_p, w, label="Random Forest", color=COLORS["rf"])
    ax.bar(x + w / 2, lag1_mae_p, w, label="Naive Lag-1", color=COLORS["lag1"])
    ax.set_xlabel("Traffic Period")
    ax.set_ylabel("MAE (Probe Flow Units)")
    ax.set_title("Simulation MAE by Traffic Period",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels)
    ax.legend()

    # Add value labels
    for bar_group in [ax.containers[0], ax.containers[1]]:
        ax.bar_label(bar_group, fmt="%.1f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(viz_dir / "simulation_period_performance.png",
                dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # ---- 5. FRC performance ----
    print("  Generating simulation_frc_performance.png...")
    frc_data = report["by_frc"]
    frc_keys = sorted(frc_data.keys())
    frc_labels = [frc_data[k]["label"] for k in frc_keys]
    rf_mae_f = [frc_data[k]["random_forest"]["MAE"] for k in frc_keys]
    lag1_mae_f = [frc_data[k]["naive_lag1"]["MAE"] for k in frc_keys]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(frc_keys))
    w = 0.35
    ax.bar(x - w / 2, rf_mae_f, w, label="Random Forest", color=COLORS["rf"])
    ax.bar(x + w / 2, lag1_mae_f, w, label="Naive Lag-1", color=COLORS["lag1"])
    ax.set_xlabel("Functional Road Class")
    ax.set_ylabel("MAE (Probe Flow Units)")
    ax.set_title("Simulation MAE by Road Classification (FRC)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(frc_labels, rotation=20, ha="right")
    ax.legend()
    for bar_group in [ax.containers[0], ax.containers[1]]:
        ax.bar_label(bar_group, fmt="%.1f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(viz_dir / "simulation_frc_performance.png",
                dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # ---- 6. Error distribution ----
    print("  Generating simulation_error_distribution.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RF errors
    rf_errors = df["rf_error"].values
    axes[0].hist(rf_errors, bins=100, color=COLORS["rf"], alpha=0.7,
                 edgecolor="white", density=True)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].axvline(np.mean(rf_errors), color="black", linestyle="-",
                    linewidth=1.5, label=f"Mean Bias: {np.mean(rf_errors):.2f}")
    axes[0].set_xlabel("Forecast Error (Predicted - Actual)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Random Forest Error Distribution")
    axes[0].set_xlim(-150, 150)
    axes[0].legend()

    # Lag-1 errors
    lag1_errors = df["lag1_error"].values
    axes[1].hist(lag1_errors, bins=100, color=COLORS["lag1"], alpha=0.7,
                 edgecolor="white", density=True)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].axvline(np.mean(lag1_errors), color="black", linestyle="-",
                    linewidth=1.5, label=f"Mean Bias: {np.mean(lag1_errors):.2f}")
    axes[1].set_xlabel("Forecast Error (Predicted - Actual)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Naive Lag-1 Error Distribution")
    axes[1].set_xlim(-150, 150)
    axes[1].legend()

    fig.suptitle("Simulation Forecast Error Distributions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(viz_dir / "simulation_error_distribution.png",
                dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # ---- 7. Daily performance ----
    print("  Generating simulation_daily_performance.png...")
    daily_data = report["by_day"]
    days = sorted(daily_data.keys())
    day_labels = [str(d) for d in days]
    rf_mae_d = [daily_data[d]["random_forest"]["MAE"] for d in days]
    lag1_mae_d = [daily_data[d]["naive_lag1"]["MAE"] for d in days]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(days))
    w = 0.35
    ax.bar(x - w / 2, rf_mae_d, w, label="Random Forest", color=COLORS["rf"])
    ax.bar(x + w / 2, lag1_mae_d, w, label="Naive Lag-1", color=COLORS["lag1"])
    ax.set_xlabel("Test Date")
    ax.set_ylabel("MAE (Probe Flow Units)")
    ax.set_title("Simulation MAE by Test Date",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
    ax.legend()
    for bar_group in [ax.containers[0], ax.containers[1]]:
        ax.bar_label(bar_group, fmt="%.1f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(viz_dir / "simulation_daily_performance.png",
                dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # ---- 8. Relative Error (WAPE) Multi-Panel Analysis ----
    print("  Generating simulation_relative_error_analysis.png...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: WAPE by FRC
    frc_data = report["by_frc"]
    frc_keys = sorted(frc_data.keys())
    frc_short_labels = [f"FRC {k.split('_')[1]}" for k in frc_keys]
    rf_wape_f = [frc_data[k]["random_forest"]["WAPE_pct"] for k in frc_keys]
    lag1_wape_f = [frc_data[k]["naive_lag1"]["WAPE_pct"] for k in frc_keys]

    x = np.arange(len(frc_keys))
    w = 0.35
    axes[0, 0].bar(x - w / 2, rf_wape_f, w, label="Random Forest", color=COLORS["rf"])
    axes[0, 0].bar(x + w / 2, lag1_wape_f, w, label="Naive Lag-1", color=COLORS["lag1"])
    axes[0, 0].set_title("Relative Error (WAPE %) by Road Classification", fontweight="bold")
    axes[0, 0].set_xlabel("Functional Road Class")
    axes[0, 0].set_ylabel("WAPE (%) = MAE / Mean Flow")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(frc_short_labels)
    axes[0, 0].legend()
    for bar_group in [axes[0, 0].containers[0], axes[0, 0].containers[1]]:
        axes[0, 0].bar_label(bar_group, fmt="%.1f%%", padding=3, fontsize=8)

    # Panel 2: WAPE by Volume Quartile
    vol_data = report["by_volume_quartile"]
    vol_keys = list(vol_data.keys())
    vol_labels = [k.replace("_", " ") for k in vol_keys]
    rf_wape_v = [vol_data[k]["random_forest"]["WAPE_pct"] for k in vol_keys]
    lag1_wape_v = [vol_data[k]["naive_lag1"]["WAPE_pct"] for k in vol_keys]

    x_v = np.arange(len(vol_keys))
    axes[0, 1].bar(x_v - w / 2, rf_wape_v, w, label="Random Forest", color=COLORS["rf"])
    axes[0, 1].bar(x_v + w / 2, lag1_wape_v, w, label="Naive Lag-1", color=COLORS["lag1"])
    axes[0, 1].set_title("Relative Error (WAPE %) by Traffic Volume Quartile", fontweight="bold")
    axes[0, 1].set_xlabel("Volume Quartile")
    axes[0, 1].set_ylabel("WAPE (%) = MAE / Mean Flow")
    axes[0, 1].set_xticks(x_v)
    axes[0, 1].set_xticklabels(vol_labels, rotation=15)
    axes[0, 1].legend()
    for bar_group in [axes[0, 1].containers[0], axes[0, 1].containers[1]]:
        axes[0, 1].bar_label(bar_group, fmt="%.1f%%", padding=3, fontsize=8)

    # Panel 3: Absolute MAE vs Relative WAPE across Quartiles
    rf_mae_v = [vol_data[k]["random_forest"]["MAE"] for k in vol_keys]
    mean_flow_v = [vol_data[k]["random_forest"]["Mean_Actual"] for k in vol_keys]

    ax_left = axes[1, 0]
    ax_right = ax_left.twinx()
    l1 = ax_left.plot(x_v, rf_mae_v, color="#D32F2F", marker="o", linewidth=2.5, label="Absolute MAE (Left)")
    l2 = ax_left.plot(x_v, mean_flow_v, color="#388E3C", marker="s", linewidth=2.0, linestyle="--", label="Mean Flow (Left)")
    l3 = ax_right.plot(x_v, rf_wape_v, color="#1976D2", marker="^", linewidth=2.5, linestyle="-.", label="Relative WAPE % (Right)")
    ax_left.set_title("Scale Contrast: Absolute Error vs Relative Error across Quartiles", fontweight="bold")
    ax_left.set_xlabel("Volume Quartile")
    ax_left.set_ylabel("Flow Units (MAE & Mean Flow)")
    ax_right.set_ylabel("Relative Error (WAPE %)")
    ax_left.set_xticks(x_v)
    ax_left.set_xticklabels(vol_labels, rotation=15)
    # Combine legends
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc="upper left", fontsize=9)

    # Panel 4: WAPE by Hour of Day
    hourly_data = report["by_hour"]
    hours = sorted([int(h) for h in hourly_data.keys()])
    rf_wape_h = [hourly_data[str(h)]["random_forest"]["WAPE_pct"] for h in hours]
    lag1_wape_h = [hourly_data[str(h)]["naive_lag1"]["WAPE_pct"] for h in hours]

    x_h = np.arange(len(hours))
    axes[1, 1].plot(x_h, rf_wape_h, color=COLORS["rf"], marker="o", linewidth=2.0, label="Random Forest")
    axes[1, 1].plot(x_h, lag1_wape_h, color=COLORS["lag1"], marker="s", linewidth=1.8, linestyle="--", label="Naive Lag-1")
    axes[1, 1].set_title("Relative Error (WAPE %) by Hour of Day", fontweight="bold")
    axes[1, 1].set_xlabel("Hour of Day")
    axes[1, 1].set_ylabel("WAPE (%)")
    axes[1, 1].set_xticks(x_h)
    axes[1, 1].set_xticklabels([f"{h:02d}" for h in hours])
    axes[1, 1].legend()

    fig.suptitle("Scale-Normalized Relative Error Analysis (Zero-Flow Robust WAPE)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(viz_dir / "simulation_relative_error_analysis.png",
                dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------
def save_results(results_df: pd.DataFrame, report: Dict[str, Any]) -> None:
    """Persist simulation outputs."""

    # JSON report
    results_json_path = MODELS_DIR / "simulation_results.json"
    # Convert any numpy/pandas types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, type(pd.NaT)):
            return None
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            result = _convert(obj)
            if result is not obj:
                return result
            return super().default(obj)

    with open(results_json_path, "w") as f:
        json.dump(report, f, indent=2, cls=NpEncoder, default=str)
    print(f"\nSaved simulation report to {results_json_path}")

    # Detailed predictions as Parquet (gitignored)
    sim_dir = PROCESSED_DATA_DIR / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "segment_id", "street_name", "frc",
        "forecast_origin_datetime", "forecast_datetime",
        "hour", "actual_probe_count", "rf_predicted", "lag1_predicted",
        "rf_error", "rf_abs_error",
    ]
    results_df[output_cols].to_parquet(
        sim_dir / "detailed_predictions.parquet", index=False
    )
    print(f"Saved detailed predictions to {sim_dir / 'detailed_predictions.parquet'}")


# ---------------------------------------------------------------------------
# 6. Print summary
# ---------------------------------------------------------------------------
def print_summary(report: Dict[str, Any]) -> None:
    """Print a formatted summary to stdout."""

    print("\n" + "=" * 70)
    print(" HISTORICAL ROLLING FORECAST SIMULATION RESULTS")
    print(" One-Step-Ahead Rolling-Origin Backtest")
    print("=" * 70)

    ov = report["overall"]
    rf = ov["random_forest"]
    lag1 = ov["naive_lag1"]

    print(f"\n  Test Period:    {TEST_START_DATE} to {TEST_END_DATE}")
    print(f"  Total Forecasts: {ov['n_forecasts']:,}")
    print(f"  Mean Actual Flow: {rf['Mean_Actual']:.2f}")

    print(f"\n  {'Metric':<20} {'Random Forest':>15} {'Naive Lag-1':>15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'MAE':<20} {rf['MAE']:>15.4f} {lag1['MAE']:>15.4f}")
    print(f"  {'RMSE':<20} {rf['RMSE']:>15.4f} {lag1['RMSE']:>15.4f}")
    print(f"  {'R-squared':<20} {rf['R2']:>15.4f} {lag1['R2']:>15.4f}")
    print(f"  {'Mean Bias':<20} {rf['Mean_Bias']:>15.4f} {lag1['Mean_Bias']:>15.4f}")
    print(f"  {'WAPE (Relative MAE)':<20} {rf['WAPE_pct']:>14.2f}% {lag1['WAPE_pct']:>14.2f}%")
    print(f"\n  RF MAE improvement over Lag-1: {ov['mae_improvement_vs_lag1_pct']}%")

    # Original benchmark comparison
    print(f"\n  --- Benchmark Comparison ---")
    print(f"  Original RF benchmark MAE:  17.8112")
    print(f"  Simulation RF MAE:          {rf['MAE']}")
    diff = abs(rf["MAE"] - 17.8112)
    if diff < 0.01:
        print(f"  Status: Metrics reproduced (delta = {diff:.4f})")
    else:
        print(f"  Status: Metrics differ by {diff:.4f} -- investigation needed")

    # Volume Quartiles Breakdown (Relative Error Analysis)
    print(f"\n  --- Relative Error (WAPE) by Traffic Volume Quartile ---")
    print(f"  {'Quartile':<18} {'Flow Range':<16} {'Mean Flow':>10} {'RF MAE':>10} {'Lag1 MAE':>10} {'RF WAPE':>10} {'Lag1 WAPE':>11}")
    print(f"  {'-'*18} {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11}")
    for q_name, data in report["by_volume_quartile"].items():
        q_rf = data["random_forest"]
        q_lag = data["naive_lag1"]
        print(f"  {q_name:<18} {data['flow_range']:<16} {q_rf['Mean_Actual']:>10.1f} {q_rf['MAE']:>10.2f} {q_lag['MAE']:>10.2f} {q_rf['WAPE_pct']:>9.1f}% {q_lag['WAPE_pct']:>10.1f}%")

    # FRC
    print(f"\n  --- MAE & Relative WAPE by Functional Road Class ---")
    print(f"  {'Road Class':<30} {'Mean Flow':>10} {'RF MAE':>10} {'Lag1 MAE':>10} {'RF WAPE':>10} {'Lag1 WAPE':>11}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11}")
    for frc_key, data in report["by_frc"].items():
        rf_f = data["random_forest"]
        lag1_f = data["naive_lag1"]
        print(f"  {data['label']:<30} {rf_f['Mean_Actual']:>10.1f} {rf_f['MAE']:>10.2f} {lag1_f['MAE']:>10.2f} {rf_f['WAPE_pct']:>9.1f}% {lag1_f['WAPE_pct']:>10.1f}%")

    # Hourly
    print(f"\n  --- Hourly MAE & Relative WAPE ---")
    hourly = report["by_hour"]
    for h in sorted(hourly.keys(), key=int):
        rf_h = hourly[h]["random_forest"]
        lag1_h = hourly[h]["naive_lag1"]
        print(f"    {int(h):02d}:00  Mean Flow: {rf_h['Mean_Actual']:>6.1f}  RF MAE: {rf_h['MAE']:>6.2f} ({rf_h['WAPE_pct']:>5.1f}%)  Lag-1: {lag1_h['MAE']:>6.2f} ({lag1_h['WAPE_pct']:>5.1f}%)")

    # Period
    print(f"\n  --- MAE & Relative WAPE by Traffic Period ---")
    for period, data in report["by_period"].items():
        rf_p = data["random_forest"]
        lag1_p = data["naive_lag1"]
        label = period.replace("_", " ").title()
        print(f"    {label:<20} Mean Flow: {rf_p['Mean_Actual']:>6.1f}  RF MAE: {rf_p['MAE']:>6.2f} ({rf_p['WAPE_pct']:>5.1f}%)  Lag-1: {lag1_p['MAE']:>6.2f} ({lag1_p['WAPE_pct']:>5.1f}%)")

    # Daily
    print(f"\n  --- MAE by Test Date ---")
    for day, data in report["by_day"].items():
        rf_d = data["random_forest"]["MAE"]
        lag1_d = data["naive_lag1"]["MAE"]
        print(f"    {day}  RF: {rf_d:>8.2f}  Lag-1: {lag1_d:>8.2f}")

    # Worst segments
    print(f"\n  --- Top 5 Segments with Highest Forecasting Error ---")
    for seg in report["segment_analysis"]["worst_10_segments"][:5]:
        rel_err = (seg['mae'] / seg['mean_actual'] * 100) if seg['mean_actual'] > 0 else 0
        print(f"    Segment {seg['segment_id']}: "
              f"{seg['street_name']} (FRC {int(seg['frc'])}) "
              f"MAE={seg['mae']:.1f}, Mean Flow={seg['mean_actual']:.1f}, Relative Error={rel_err:.1f}%")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" HISTORICAL ROLLING FORECAST SIMULATION")
    print(" One-Step-Ahead Rolling-Origin Backtest")
    print("=" * 70)
    t_start = time.time()

    # 1. Prepare data
    test_df = prepare_test_data()

    # Validation assertions
    expected_raw = 24938 * 24 * 4  # 2,394,048
    print(f"\n  Raw test period rows expected:  {expected_raw:,}")
    print(f"  Clean test rows available:      {len(test_df):,}")
    if len(test_df) < expected_raw * 0.95:
        print("  WARNING: Significantly fewer rows than expected. "
              "Check lag NaN dropout.")

    # 2. Run simulation
    results_df = run_simulation(test_df)

    # 3. Compute breakdowns
    print("\nComputing performance breakdowns...")
    report = compute_breakdowns(results_df)

    # Add simulation metadata
    report["metadata"] = {
        "simulation_type": "one_step_ahead_rolling_origin_backtest",
        "model": "Random Forest Regressor (pre-trained, models/random_forest.joblib)",
        "test_period": f"{TEST_START_DATE} to {TEST_END_DATE}",
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "baseline": "Naive Lag-1 Persistence",
        "relative_metric_definition": (
            "WAPE (%) = (MAE / Mean_Actual) * 100 = sum(|y - y_hat|) / sum(y) * 100. "
            "Robust scale-normalized metric that handles zero-traffic observations safely."
        ),
        "note": (
            "This is a one-step-ahead rolling-origin backtest. At each "
            "forecast origin time t, only information available through t is "
            "used. Predictions are never fed back into future lag features. "
            "Every lag feature uses actual historical observations."
        ),
    }

    # 4. Visualizations
    sim_viz_dir = VISUALIZATIONS_DIR / "simulation"
    print(f"\nGenerating diagnostic visualizations in {sim_viz_dir}/...")
    generate_visualizations(results_df, report, sim_viz_dir)

    # 5. Save results
    save_results(results_df, report)

    # 6. Print summary
    print_summary(report)

    elapsed = time.time() - t_start
    print(f"\nTotal simulation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

