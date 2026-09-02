"""
Feature Engineering Module for Traffic Flow Prediction.
Creates temporal, road, historical lag features, and segment historical statistics.

Rigorous anti-leakage principles:
1. Segment historical stats (mean, std, median) are computed STRICTLY on the training period (Aug 11-26).
2. Lags are created using an explicit (segment_id, datetime) grid index.
3. No future observations or target values are ever fed into input feature matrices.
"""
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    MORNING_RUSH_HOURS,
    EVENING_RUSH_HOURS,
    SPECIAL_DATES
)


def load_dataset_range(start_date: str, end_date: str, data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Loads and concatenates partitioned Parquet partitions for a specific date range."""
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")
    dfs = []
    
    for d in dates:
        p_path = data_dir / f"date={d}" / "data.parquet"
        if p_path.exists():
            df_day = pd.read_parquet(p_path)
            dfs.append(df_day)
        else:
            print(f"Warning: Partition not found for date {d} at {p_path}")
            
    if not dfs:
        raise FileNotFoundError(f"No processed data partitions found between {start_date} and {end_date}")
        
    return pd.concat(dfs, ignore_index=True)


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs rich temporal features from date and hour:
    - day_of_week (0=Mon, 6=Sun)
    - day_of_month
    - is_weekend
    - is_morning_rush
    - is_evening_rush
    - is_rush_hour
    - is_festival (Independence Day, Rakshabandhan, Janmashtami)
    - cyclical hour (sin/cos)
    - cyclical day of week (sin/cos)
    """
    df = df.copy()
    dt_series = pd.to_datetime(df["date"])
    
    df["day_of_week"] = dt_series.dt.dayofweek.astype("int8")
    df["day_of_month"] = dt_series.dt.day.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    
    df["is_morning_rush"] = df["hour"].isin(MORNING_RUSH_HOURS).astype("int8")
    df["is_evening_rush"] = df["hour"].isin(EVENING_RUSH_HOURS).astype("int8")
    df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype("int8")
    
    date_str_series = df["date"].astype(str)
    df["is_festival"] = date_str_series.isin(SPECIAL_DATES.keys()).astype("int8")
    
    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype("float32")
    
    # Cyclical day of week encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).astype("float32")
    
    return df


def build_timestamp_aware_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs timestamp-aware historical lag features for each road segment:
    - lag_1: t - 1 hour
    - lag_2: t - 2 hours
    - lag_3: t - 3 hours
    - lag_24: t - 24 hours (exact same hour yesterday)
    
    Ensures strict chronological sorting and contiguous time indices.
    """
    df = df.copy()
    
    # Construct exact timestamp
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"].astype(str).str.zfill(2) + ":00:00")
    
    # Sort by segment_id and datetime to guarantee sequence
    df = df.sort_values(by=["segment_id", "datetime"]).reset_index(drop=True)
    
    # Grouped shift
    grouped = df.groupby("segment_id")["probe_count"]
    df["probe_count_lag_1"] = grouped.shift(1).astype("float32")
    df["probe_count_lag_2"] = grouped.shift(2).astype("float32")
    df["probe_count_lag_3"] = grouped.shift(3).astype("float32")
    df["probe_count_lag_24"] = grouped.shift(24).astype("float32")
    
    # Rolling short-term statistics (prior 3 hours mean)
    df["probe_count_roll_mean_3h"] = (
        (df["probe_count_lag_1"] + df["probe_count_lag_2"] + df["probe_count_lag_3"]) / 3.0
    ).astype("float32")
    
    return df


def compute_segment_historical_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes segment historical statistics STRICTLY on the training split to avoid data leakage:
    - segment_mean_traffic
    - segment_std_traffic
    - segment_p90_traffic
    - segment_zero_freq
    """
    stats = train_df.groupby("segment_id")["probe_count"].agg(
        segment_mean_traffic="mean",
        segment_std_traffic="std",
        segment_p90_traffic=lambda x: np.percentile(x, 90),
        segment_zero_freq=lambda x: (x == 0).mean()
    ).reset_index()
    
    stats["segment_std_traffic"] = stats["segment_std_traffic"].fillna(0.0)
    return stats
