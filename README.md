# ?? Intelligent Urban Traffic Flow & Congestion Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ML Models](https://img.shields.io/badge/models-OLS%20%7C%20Ridge%20%7C%20RF%20%7C%20XGBoost-success)](https://github.com/)
[![Dataset](https://img.shields.io/badge/dataset-New%20Delhi%20Traffic%20(2024)-orange)](https://www.kaggle.com/datasets/ryanmadhuwala/new-delhi-traffic-probe-analytics-2024)

An end-to-end machine learning and spatial-temporal forecasting system that predicts the **next-hour traffic probe flow ($t+1$)** on **24,938 individual road segments** across the **Delhi NCR road network (15,000+ km)** and provides congestion-pattern analysis using aggregated city-level metrics.

---

## ?? Executive Summary & Problem Formulation

- **Target**: Next-hour traffic probe flow proxy ($\text{probe\_count}_{t+1}$).
- **Data Volume**: **11.97 million road-segment/hour records** representing approximately 24,938 segments across 20 days (August 1130, 2024).
- **Measurement Interpretation**: Vehicle probe detections per road segment per hour, functioning as an empirical proxy for traffic volume (from connected GPS mobility devices) rather than absolute physical vehicle census counts.
- **Scientific Integrity**:
  - No target leakage: strict timestamp-aware lag indexing ($t-1, t-2, t-3, t-24$) and out-of-sample segment priors.
  - Strict chronological evaluation: **Train (Aug 1126: 8.97M rows)** vs **Test (Aug 2730: 2.39M rows)**.
  - Contextual Congestion Analytics: Uses official aggregated statistics (43% morning vs 69% evening congestion; speeds dropping from 23.6 to 19.9 km/h) rather than inventing unverified road-level labels.

---

## ?? Empirical Model Benchmark Results

Evaluation conducted strictly on the **unseen 4-day chronological test split (2,394,048 samples)**:

| Model Architecture | Training Time | Test MAE (Probe Flow) | Test RMSE | Test $R^2$ Score | Error Reduction vs Naive Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Lag-1 Persistence** ($\hat{y}_t = y_{t-1}$) | 0.00s | 26.4178 | 51.9970 | 0.9349 | Baseline Reference |
| **Naive Lag-24 Seasonal** ($\hat{y}_t = y_{t-24}$) | 0.00s | 24.9289 | 54.9575 | 0.9272 | -5.6% MAE vs Lag-1 |
| **Ordinary Linear Regression (OLS)** | 3.15s | 21.8391 | 40.7394 | 0.9600 | **+17.3% MAE improvement** |
| **Ridge Regression (L2-Regularized)** | 3.31s | 21.8391 | 40.7393 | 0.9600 | **+17.3% MAE improvement** |
| **Random Forest Regressor (Best)** | 118.79s | **17.8112** | **36.2533** | **0.9683** | **+32.6% MAE improvement** |
| **XGBoost Regressor (Histogram)** | 111.55s | 18.3959 | 38.3657 | 0.9645 | **+30.4% MAE improvement** |

> **Key Finding**: **Random Forest** achieved the best predictive performance on the unseen future test set, reducing MAE by **32.6%** over the Naive Lag-1 baseline and outperforming XGBoost. The evaluation illustrates that incorporating multiple temporal, road-level, and lag features substantially enhances accuracy over simple auto-regressive persistence.

---

## ??? Urban Congestion & Rush-Hour Findings

Analysis of aggregated mobility benchmarks reveals key macro patterns:
- **Diurnal Congestion Surge**: Evening rush-hour conditions were substantially more congested than morning rush-hour conditions, with congestion reported at **69% versus 43%** and average city speed falling from **23.6 km/h to 19.9 km/h**.
- **Diurnal Progression**: City-wide traffic builds gradually from 10:00 AM (28% congestion) to peak at 6:00 PM (67% average weekday congestion) before tapering off late at night.

---

## ??? System Architecture

```
                 RAW DATA (20 GeoJSON Files)
                              
                              ?
                 Incremental Ingestion Pipeline
                              
                              ?
            Partitioned Parquet (date=YYYY-MM-DD)
                              
                              ?
                     Data Quality Checks
            (100% Valid Coords, 0 Duplicates, 8.06% Zeros)
                              
                              ?
                 Exploratory Data Analysis
            (10 High-Resolution Spatial-Temporal Plots)
                              
                              ?
                 Time-Aware Feature Engineering
               +-----------------------------+
                                            
        Temporal Features               Road Features
     (Cyclical, Rush Hours)         (FRC, Speed Limit)
                                            
               +-----------------------------+
                              
                              ?
                  Timestamp-Aware Lags (t-1, t-2, t-3, t-24)
                              
                              ?
                    Chronological Split
               +-----------------------------+
             TRAIN                         TEST
       (Aug 11-26, 2024)             (Aug 27-30, 2024)
         8,977,680 Rows                2,394,048 Rows
                                            
               ?                             
     Segment Historical Stats                
    (Target Encoding Priors)                 
                                            
               +-----------------------------+
                              
                              ?
                       Model Training
        (Naive Baselines, OLS, Ridge, Random Forest, XGBoost)
                              
                              ?
                Empirical Evaluation & Diagnostics
           (MAE, RMSE, R2, Feature Importances, Residuals)
                              
                              ?
             Congestion & Macro Traffic Analytics
               (global_metrics & weekday_stats)
                              
                              ?
             Interactive Flask Dashboard
               +-----------------------------+
               ?                             ?
        On-Demand Forecasts             Spatial Map
        (t+1 Rolling API)             (Folium Delhi NCR)
```

---

## ?? Project Structure

```
intelligent-traffic-prediction/

+-- data/
   +-- raw/                       # Original GeoJSON dataset
   +-- processed/                 # Partitioned Parquet datasets (date=YYYY-MM-DD/)

+-- src/
   +-- __init__.py
   +-- config.py                  # System paths, constants & special date mappings
   +-- data_ingestion.py          # Incremental GeoJSON -> Parquet ingestion
   +-- data_quality.py            # Quality audit, missing value & grid checks
   +-- eda.py                     # 10 comprehensive visualization generators
   +-- feature_engineering.py     # Temporal encodings, timestamp lags, segment priors
   +-- model_training.py          # Naive baselines, OLS, Ridge, Random Forest, XGBoost
   +-- congestion_analysis.py     # Global metrics & weekday speed/congestion heatmaps
   +-- prediction.py              # In-memory inference engine for the dashboard

+-- models/
   +-- linear_regression_ols.joblib # Ordinary Least Squares Linear Regression
   +-- ridge_regression.joblib      # L2-Regularized Ridge Regression
   +-- random_forest.joblib         # Random Forest ensemble (Best model)
   +-- xgboost.joblib               # XGBoost regressor
   +-- segment_stats.parquet        # Prior traffic statistics per road segment
   +-- model_evaluation_metrics.json

+-- dashboard/
   +-- app.py                     # Flask web server
   +-- templates/index.html       # Responsive dashboard UI
   +-- static/                    # Custom CSS styling & Chart.js client

+-- visualizations/                # 16 publication-grade diagnostic plots (.png)
+-- docs/
   +-- architecture.md            # Data pipeline Mermaid diagrams
   +-- methodology.md             # Scientific formulation & validation rationale
   +-- resume_description.md      # Resume bullet points & interview Q&A guide

+-- requirements.txt
+-- run_pipeline.py                # End-to-end pipeline execution runner
+-- README.md
```

---

## ?? Visual Insights Catalog (16 Plots)

All plots are located in [`visualizations/`](file:///c:/Users/LENOVO/Documents/antigravity/optimistic-einstein/visualizations):
- `01_traffic_flow_distribution.png`  Distribution and outlier spread
- `02_hourly_traffic_profile.png`  24-hour diurnal traffic volume curve
- `03_day_of_week_traffic.png`  Day-of-week traffic differences
- `04_weekday_vs_weekend.png`  Commuter weekday vs weekend dynamics
- `05_rush_vs_offpeak.png`  Morning/Evening peak vs off-peak intensity
- `06_frc_traffic_variation.png`  Functional Road Class (FRC 16) volume hierarchy
- `07_date_trend_and_festivals.png`  20-day timeline with festival annotations (Independence Day, Rakshabandhan, Janmashtami)
- `08_speed_limit_vs_traffic.png`  Flow variations across posted speed limits
- `09_spatial_density_map.png`  Delhi NCR spatial probe density map
- `10_festival_traffic_anomalies.png`  Festival vs regular diurnal curves
- `11_model_metrics_comparison.png`  Test MAE, RMSE, and $R^2$ benchmarks
- `12_actual_vs_predicted_and_residuals.png`  Scatter fit and error distribution
- `13_xgboost_feature_importance.png`  Top predictive features (lag-1, segment mean, lag-24)
- `14_weekday_congestion_heatmap.png`  Diurnal congestion percentages
- `15_weekday_speed_heatmap.png`  Hourly speed degradation patterns
- `16_monthly_congestion_comparison.png`  2024 vs 2023 YoY monthly benchmarks

---

## ?? Quickstart & Dashboard Launch

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Dashboard
```bash
python dashboard/app.py
```
Open **http://127.0.0.1:5000** in your browser to interact with the On-Demand Next-Hour Predictor and Folium Traffic Map.

---

## ?? Limitations & Future Scope

1. **Probe Sample Representation**: The dataset captures probe detections from connected GPS devices; future extensions could integrate loop sensor or toll booth counts for complete physical volume reconciliation.
2. **Real-Time Streaming**: The current system demonstrates on-demand forecasting on historical partitions. It can be extended with Apache Kafka/Flink to consume live IoT telemetry feeds.
3. **Graph Neural Networks (GNNs)**: Spatial relationships are currently modeled via coordinates and road characteristics; spatial graph convolutions (e.g., ST-GCN or DCRNN) represent an exciting avenue for modeling adjacent road segment spillovers.
