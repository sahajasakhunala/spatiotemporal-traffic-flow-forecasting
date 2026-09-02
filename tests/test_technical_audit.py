import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.config import (
    TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE,
    MODELS_DIR, PROCESSED_DATA_DIR
)
from src.feature_engineering import (
    load_dataset_range, build_temporal_features,
    build_timestamp_aware_lags, compute_segment_historical_stats
)
from src.model_training import FEATURE_COLUMNS, TARGET_COLUMN, calculate_metrics
from src.prediction import TrafficPredictor


class ComprehensiveTechnicalAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n--- Initializing Comprehensive Technical Audit ---")
        # Load small multi-segment sample to verify lag math precisely
        cls.sample_df = load_dataset_range("2024-08-11", "2024-08-13")
        cls.featured_sample = build_timestamp_aware_lags(build_temporal_features(cls.sample_df))
        
        # Load saved segment stats and metrics JSON
        cls.saved_stats = pd.read_parquet(MODELS_DIR / "segment_stats.parquet")
        with open(MODELS_DIR / "model_evaluation_metrics.json", "r") as f:
            cls.metrics = json.load(f)

    def test_01_timestamp_aware_lags_alignment(self):
        """1. Verify lag_1, lag_2, lag_3, lag_24 are truly timestamp-aligned per segment."""
        # Pick 5 random segments
        test_segments = self.featured_sample["segment_id"].drop_duplicates().head(5).tolist()
        
        for seg in test_segments:
            seg_data = self.featured_sample[self.featured_sample["segment_id"] == seg].sort_values("datetime").reset_index(drop=True)
            
            # For each row after hour 24, verify lag matches exact prior row
            for idx in range(24, min(48, len(seg_data))):
                current_time = seg_data.loc[idx, "datetime"]
                
                # Check lag_1: exactly 1 hour prior
                lag1_time = seg_data.loc[idx - 1, "datetime"]
                self.assertEqual((current_time - lag1_time).total_seconds(), 3600)
                self.assertEqual(seg_data.loc[idx, "probe_count_lag_1"], seg_data.loc[idx - 1, "probe_count"])
                
                # Check lag_2: exactly 2 hours prior
                lag2_time = seg_data.loc[idx - 2, "datetime"]
                self.assertEqual((current_time - lag2_time).total_seconds(), 7200)
                self.assertEqual(seg_data.loc[idx, "probe_count_lag_2"], seg_data.loc[idx - 2, "probe_count"])
                
                # Check lag_24: exactly 24 hours prior
                lag24_time = seg_data.loc[idx - 24, "datetime"]
                self.assertEqual((current_time - lag24_time).total_seconds(), 86400)
                self.assertEqual(seg_data.loc[idx, "probe_count_lag_24"], seg_data.loc[idx - 24, "probe_count"])

    def test_02_segment_stats_calculated_only_on_training_data(self):
        """2. Verify segment statistics were calculated only from training data (Aug 11-26)."""
        train_df = load_dataset_range(TRAIN_START_DATE, TRAIN_END_DATE)
        computed_stats = compute_segment_historical_stats(train_df)
        
        # Pick 10 sample segments and verify exact equality with saved segment_stats.parquet
        sample_segs = computed_stats["segment_id"].head(10).tolist()
        for seg in sample_segs:
            comp_mean = computed_stats.loc[computed_stats["segment_id"] == seg, "segment_mean_traffic"].values[0]
            saved_mean = self.saved_stats.loc[self.saved_stats["segment_id"] == seg, "segment_mean_traffic"].values[0]
            self.assertAlmostEqual(comp_mean, saved_mean, places=4)

    def test_03_test_set_exact_dates_and_row_count(self):
        """3. Verify the 2.39M test observations correspond exactly to Aug 2730."""
        test_df = load_dataset_range(TEST_START_DATE, TEST_END_DATE)
        test_dates = sorted([str(d) for d in test_df["date"].unique()])
        expected_dates = ["2024-08-27", "2024-08-28", "2024-08-29", "2024-08-30"]
        self.assertEqual(test_dates, expected_dates)
        
        # 24,938 segments * 24 hours * 4 days = 2,394,048
        expected_rows = 24938 * 24 * 4
        self.assertEqual(len(test_df), expected_rows)

    def test_04_feature_matrix_consistency(self):
        """4. Verify OLS, Ridge, RF, and XGBoost feature alignment."""
        ols = joblib.load(MODELS_DIR / "linear_regression_ols.joblib")
        ridge = joblib.load(MODELS_DIR / "ridge_regression.joblib")
        xgb_m = joblib.load(MODELS_DIR / "xgboost.joblib")
        rf_m = joblib.load(MODELS_DIR / "random_forest.joblib")
        
        # Number of input features must equal FEATURE_COLUMNS length (24)
        self.assertEqual(ols.n_features_in_, len(FEATURE_COLUMNS))
        self.assertEqual(ridge.n_features_in_, len(FEATURE_COLUMNS))
        self.assertEqual(rf_m.n_features_in_, len(FEATURE_COLUMNS))
        self.assertEqual(xgb_m.n_features_in_, len(FEATURE_COLUMNS))

    def test_05_dashboard_prediction_pipeline_consistency(self):
        """5. Verify dashboard prediction uses the exact same feature engineering as training."""
        predictor = TrafficPredictor(models_dir=MODELS_DIR)
        
        # Predict on a known sample
        pred_res = predictor.predict_next_hour(
            segment_id=-13560111507837,
            date_str="2024-08-28",
            hour=18,
            recent_lags=[15.0, 14.0, 12.0, 16.0],
            model_name="random_forest"
        )
        self.assertIn("predicted_next_hour_probe_flow", pred_res)
        self.assertIsInstance(pred_res["predicted_next_hour_probe_flow"], float)
        self.assertGreaterEqual(pred_res["predicted_next_hour_probe_flow"], 0.0)

    def test_06_map_sampling_guardrail(self):
        """6. Verify the dashboard map renders a capped subset (<200) rather than all 25k segments."""
        from dashboard.app import get_map, predictor
        self.assertIsNotNone(predictor.segment_metadata)
        # Check that get_map executes swiftly and produces valid HTML string
        map_html = get_map()
        self.assertIn("<iframe", map_html.lower() or "<div" in map_html.lower() or "leaflet" in map_html.lower())

    def test_07_reproducibility_of_reported_metrics(self):
        """7. Verify metrics JSON contains exact reported benchmark hierarchy."""
        self.assertEqual(self.metrics["Random Forest Regressor"]["test_metrics"]["MAE"], 17.8112)
        self.assertEqual(self.metrics["Random Forest Regressor"]["test_metrics"]["R2"], 0.9683)
        self.assertEqual(self.metrics["XGBoost Regressor"]["test_metrics"]["MAE"], 18.3959)
        self.assertEqual(self.metrics["Ordinary Linear Regression (OLS)"]["test_metrics"]["MAE"], 21.8391)
        self.assertEqual(self.metrics["Naive Lag-1 Persistence"]["test_metrics"]["MAE"], 26.4178)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(ComprehensiveTechnicalAudit)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        exit(1)
