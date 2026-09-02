"""
Comprehensive End-to-End Pipeline Execution Script.
Executes Step 1 through Step 14:
1. Ingests all 20 GeoJSON daily files into partitioned Parquet.
2. Performs data quality audits.
3. Computes comprehensive EDA and exports high-res charts.
4. Generates temporal, road, and timestamp-aware lag features.
5. Computes segment historical stats on training split (Aug 11-26).
6. Trains Naive Baselines, Linear Regression, Random Forest, and XGBoost.
7. Evaluates MAE, RMSE, R and saves comparison plots & metrics.
8. Runs urban congestion and speed benchmark analytics.
"""
import sys
import time
from pathlib import Path
import pandas as pd
import json

from src.config import (
    PROBE_COUNTS_GEOJSON_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    VISUALIZATIONS_DIR,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE
)
from src.data_ingestion import process_all_geojson_files, parse_single_geojson
from src.data_quality import validate_single_day, print_quality_report
from src.eda import run_comprehensive_eda
from src.feature_engineering import (
    load_dataset_range,
    build_temporal_features,
    build_timestamp_aware_lags,
    compute_segment_historical_stats
)
from src.model_training import train_and_evaluate_models
from src.congestion_analysis import run_congestion_analysis


def main():
    print("=" * 70)
    print(" INTELLIGENT URBAN TRAFFIC FLOW & CONGESTION PREDICTION SYSTEM ")
    print("=" * 70)
    start_total_time = time.time()
    
    # -------------------------------------------------------------
    # Step 1-4: Data Ingestion into Partitioned Parquet
    # -------------------------------------------------------------
    print("\n[STEP 1-4] Ingesting GeoJSON files to partitioned Parquet...")
    raw_files = sorted(list(PROBE_COUNTS_GEOJSON_DIR.glob("*.geojson")))
    if not raw_files:
        print("Error: Raw GeoJSON files not found!")
        sys.exit(1)
        
    # Check if already processed
    existing_partitions = list(PROCESSED_DATA_DIR.glob("date=*"))
    if len(existing_partitions) >= len(raw_files):
        print(f"-> Partitions already exist ({len(existing_partitions)} dates found). Skipping raw parsing.")
    else:
        process_all_geojson_files()
        
    # -------------------------------------------------------------
    # Step 5: Data Quality Validation
    # -------------------------------------------------------------
    print("\n[STEP 5] Running Data Quality Validation on representative partition...")
    sample_day_df = pd.read_parquet(PROCESSED_DATA_DIR / "date=2024-08-12" / "data.parquet")
    quality_report = validate_single_day(sample_day_df)
    print_quality_report(quality_report, title="Single Day Quality Audit (2024-08-12)")
    
    # Save quality audit summary to docs
    with open(MODELS_DIR.parent / "docs" / "data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)
        
    # -------------------------------------------------------------
    # Step 6: Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------
    print("\n[STEP 6] Executing Comprehensive EDA (10 Visual Analyses)...")
    # Load multi-day sample for rich EDA (Aug 11 to Aug 30)
    full_df = load_dataset_range(TRAIN_START_DATE, TEST_END_DATE)
    print(f"Loaded full dataset: {len(full_df):,} total observations across {full_df['segment_id'].nunique():,} road segments.")
    
    run_comprehensive_eda(full_df)
    
    # -------------------------------------------------------------
    # Step 7-8: Feature Engineering & Timestamp-Aware Lags
    # -------------------------------------------------------------
    print("\n[STEP 7-8] Feature Engineering: Building Temporal, Road, and Timestamp-Aware Lags...")
    df_temporal = build_temporal_features(full_df)
    df_featured = build_timestamp_aware_lags(df_temporal)
    
    # Chronological Split
    print(f"\n[STEP 9] Applying Chronological Split: Train ({TRAIN_START_DATE} to {TRAIN_END_DATE}) | Test ({TEST_START_DATE} to {TEST_END_DATE})...")
    train_mask = (df_featured["date"] >= pd.to_datetime(TRAIN_START_DATE).date()) & (df_featured["date"] <= pd.to_datetime(TRAIN_END_DATE).date())
    test_mask = (df_featured["date"] >= pd.to_datetime(TEST_START_DATE).date()) & (df_featured["date"] <= pd.to_datetime(TEST_END_DATE).date())
    
    train_df = df_featured[train_mask].copy()
    test_df = df_featured[test_mask].copy()
    
    # Anti-leakage: Compute segment stats STRICTLY on training split
    print("Computing road segment historical statistics strictly on training split...")
    seg_stats = compute_segment_historical_stats(train_df)
    
    # Save segment metadata for inference
    seg_meta = train_df[["segment_id", "street_name", "speed_limit", "frc", "distance", "longitude", "latitude"]].drop_duplicates(subset=["segment_id"])
    seg_combined = pd.merge(seg_meta, seg_stats, on="segment_id", how="left")
    seg_combined.to_parquet(MODELS_DIR / "segment_stats.parquet", index=False)
    print(f"Saved segment statistics & metadata ({len(seg_combined)} segments) to models/segment_stats.parquet")
    
    # Merge stats back into train and test
    train_df = pd.merge(train_df, seg_stats, on="segment_id", how="left")
    test_df = pd.merge(test_df, seg_stats, on="segment_id", how="left")
    
    # Fill any unseen test segment stats with global train defaults
    test_df["segment_mean_traffic"] = test_df["segment_mean_traffic"].fillna(train_df["probe_count"].mean())
    test_df["segment_std_traffic"] = test_df["segment_std_traffic"].fillna(train_df["probe_count"].std())
    test_df["segment_p90_traffic"] = test_df["segment_p90_traffic"].fillna(train_df["probe_count"].quantile(0.9))
    test_df["segment_zero_freq"] = test_df["segment_zero_freq"].fillna(0.1)
    
    # -------------------------------------------------------------
    # Step 10-13: Model Training & Rigorous Evaluation
    # -------------------------------------------------------------
    print("\n[STEP 10-13] Training Models (Linear Regression, Random Forest, XGBoost) & Benchmarking Naive Baselines...")
    results = train_and_evaluate_models(train_df, test_df)
    
    # Export clean metric results JSON
    metric_summary = {}
    for model_name, res in results.items():
        metric_summary[model_name] = {
            "test_metrics": res["test"],
            "training_time_sec": res.get("time_sec", 0.0)
        }
    with open(MODELS_DIR / "model_evaluation_metrics.json", "w") as f:
        json.dump(metric_summary, f, indent=2)
    print("Saved evaluation metrics to models/model_evaluation_metrics.json")
    
    # -------------------------------------------------------------
    # Step 14: Congestion & Macro Pattern Analytics
    # -------------------------------------------------------------
    print("\n[STEP 14] Executing Congestion & Macro Traffic Pattern Analytics...")
    congestion_report = run_congestion_analysis()
    
    elapsed = round(time.time() - start_total_time, 1)
    print("\n" + "=" * 70)
    print(f" PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed} SECONDS ")
    print("=" * 70)


if __name__ == "__main__":
    main()
