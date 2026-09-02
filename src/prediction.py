"""
Inference & Prediction Pipeline.
Loads trained models and historical statistics to perform next-hour traffic flow inference
for specific road segments and contextual time inputs.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SPECIAL_DATES, MORNING_RUSH_HOURS, EVENING_RUSH_HOURS
from src.model_training import FEATURE_COLUMNS


class TrafficPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.load_models()
        self.segment_metadata = None
        self.load_metadata()
        
    def load_models(self):
        """Loads available pre-trained models from disk."""
        lr_path = self.models_dir / "linear_regression.joblib"
        rf_path = self.models_dir / "random_forest.joblib"
        xgb_path = self.models_dir / "xgboost.joblib"
        
        if lr_path.exists():
            self.models["linear_regression"] = joblib.load(lr_path)
        if rf_path.exists():
            self.models["random_forest"] = joblib.load(rf_path)
        if xgb_path.exists():
            self.models["xgboost"] = joblib.load(xgb_path)
            
    def load_metadata(self):
        """Loads segment historical statistics and road metadata."""
        stats_path = self.models_dir / "segment_stats.parquet"
        if stats_path.exists():
            self.segment_metadata = pd.read_parquet(stats_path)
            
    def predict_next_hour(self, segment_id: int, date_str: str, hour: int,
                          recent_lags: List[float], model_name: str = "xgboost") -> Dict[str, Any]:
        """
        Runs next-hour traffic flow prediction for a given road segment and time context.
        recent_lags: [lag_1, lag_2, lag_3, lag_24]
        """
        if model_name not in self.models:
            fallback = list(self.models.keys())[0] if self.models else None
            if not fallback:
                raise RuntimeError("No models loaded for prediction!")
            model_name = fallback
            
        model = self.models[model_name]
        
        # Build temporal features
        dt = pd.to_datetime(date_str)
        day_of_week = dt.dayofweek
        day_of_month = dt.day
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_morning_rush = 1 if hour in MORNING_RUSH_HOURS else 0
        is_evening_rush = 1 if hour in EVENING_RUSH_HOURS else 0
        is_rush_hour = 1 if (is_morning_rush or is_evening_rush) else 0
        is_festival = 1 if date_str in SPECIAL_DATES else 0
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        
        lag_1, lag_2, lag_3, lag_24 = recent_lags
        roll_mean_3h = (lag_1 + lag_2 + lag_3) / 3.0
        
        # Segment characteristics lookup
        seg_stats = {}
        if self.segment_metadata is not None:
            match = self.segment_metadata[self.segment_metadata["segment_id"] == segment_id]
            if not match.empty:
                row = match.iloc[0]
                speed_limit = float(row.get("speed_limit", 45))
                frc = int(row.get("frc", 4))
                distance = float(row.get("distance", 100))
                seg_mean = float(row.get("segment_mean_traffic", lag_1))
                seg_std = float(row.get("segment_std_traffic", 1.0))
                seg_p90 = float(row.get("segment_p90_traffic", lag_1 * 1.5))
                seg_zero = float(row.get("segment_zero_freq", 0.1))
            else:
                speed_limit, frc, distance, seg_mean, seg_std, seg_p90, seg_zero = 45.0, 4, 100.0, lag_1, 1.0, lag_1*1.5, 0.1
        else:
            speed_limit, frc, distance, seg_mean, seg_std, seg_p90, seg_zero = 45.0, 4, 100.0, lag_1, 1.0, lag_1*1.5, 0.1
            
        feature_dict = {
            "hour": hour,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "is_weekend": is_weekend,
            "is_morning_rush": is_morning_rush,
            "is_evening_rush": is_evening_rush,
            "is_rush_hour": is_rush_hour,
            "is_festival": is_festival,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "speed_limit": speed_limit,
            "frc": frc,
            "distance": distance,
            "probe_count_lag_1": lag_1,
            "probe_count_lag_2": lag_2,
            "probe_count_lag_3": lag_3,
            "probe_count_lag_24": lag_24,
            "probe_count_roll_mean_3h": roll_mean_3h,
            "segment_mean_traffic": seg_mean,
            "segment_std_traffic": seg_std,
            "segment_p90_traffic": seg_p90,
            "segment_zero_freq": seg_zero
        }
        
        feature_vector = np.array([[feature_dict[col] for col in FEATURE_COLUMNS]])
        raw_pred = float(model.predict(feature_vector)[0])
        pred_traffic = max(0.0, round(raw_pred, 1)) # Traffic counts are non-negative
        
        return {
            "segment_id": segment_id,
            "prediction_time": f"{date_str} {hour:02d}:00",
            "model_used": model_name,
            "predicted_next_hour_probe_flow": pred_traffic,
            "naive_lag1_baseline": lag_1,
            "naive_lag24_baseline": lag_24,
            "delta_vs_lag1": round(pred_traffic - lag_1, 1),
            "is_rush_hour": bool(is_rush_hour)
        }
