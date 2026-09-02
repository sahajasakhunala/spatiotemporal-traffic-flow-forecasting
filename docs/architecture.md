# Architecture & Data Flow Design

```mermaid
flowchart TD
    subgraph RawData [Raw Dataset Ingestion]
        A[20 Daily GeoJSON Files: Aug 11-30, 2024] --> B[Incremental Ingestion: parse_single_geojson]
        B --> C[(Partitioned Parquet: date=YYYY-MM-DD)]
    end

    subgraph QualityAudit [Quality & Validation]
        C --> D[Data Quality Audit: Grid Completeness, Outliers, Zeros]
        D --> E[Comprehensive EDA: 10 Diurnal, Spatial, & Festival Analyses]
    end

    subgraph FeaturePipeline [Feature Engineering & Anti-Leakage]
        E --> F[Temporal Features: Cyclical Hour, Day of Week, Rush Hour]
        F --> G[Timestamp-Aware Lags: lag_1, lag_2, lag_3, lag_24]
        G --> H[Chronological Split: Train Aug 11-26 | Test Aug 27-30]
        H --> I[Historical Segment Stats: Computed STRICTLY on Train Split]
    end

    subgraph ModelSuite [Empirical Forecasting Models]
        I --> J1[Naive Lag-1 Persistence]
        I --> J2[Naive Lag-24 Seasonal Persistence]
        I --> J3[Linear Regression Baseline]
        I --> J4[Random Forest Regressor]
        I --> J5[XGBoost Regressor]
    end

    subgraph Evaluation [Evaluation & Benchmarking]
        J1 & J2 & J3 & J4 & J5 --> K[Metric Evaluation: MAE, RMSE, R2, Residual Diagnostics]
    end

    subgraph Context & Serving [Serving & Dashboard]
        K --> L[Congestion Benchmarks: global_metrics & weekday_stats]
        L --> M[Flask Interactive Dashboard + Folium Delhi NCR Map]
    end
```
