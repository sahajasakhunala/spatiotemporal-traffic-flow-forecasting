# ?? Resume Description & Viva / Interview Guide

## ?? Bullet Points for Resume / Portfolio (Ready to copy-paste)

> **Intelligent Urban Traffic Flow & Congestion Prediction System**  
> *Python, Scikit-Learn, XGBoost, Pandas, PyArrow, Flask, Folium, Chart.js*
> - Engineered an end-to-end spatial-temporal machine learning pipeline processing **11.97 million road-segment/hour records** across **24,938 New Delhi road segments** using memory-efficient partitioned Parquet datasets.
> - Formulated strict time-aware feature engineering including timestamp-indexed historical lags ($t-1, t-2, t-3, t-24$), cyclical diurnal encodings, and out-of-sample segment priors to prevent data leakage.
> - Benchmarked persistence baselines, Ordinary Linear Regression (OLS), Ridge Regression (L2), Random Forest, and XGBoost on an unseen 4-day chronological test split (2.39M observations); **Random Forest achieved top performance with $R^2=0.9683$ and 17.81 MAE (a 32.6% error reduction over Naive Lag-1)**.
> - Built an interactive Flask and Folium spatial dashboard providing on-demand next-hour forecasting and urban rush-hour congestion analytics (identifying peak evening delays with 69% congestion vs 43% morning).

---

## ??? Viva & Technical Interview Talking Points

### Q1: What exactly does your model predict?
> *"The model predicts the **next-hour traffic probe flow ($\text{probe\_count}_{t+1}$)** on an individual road segment. We explicitly treat probeCount as an empirical proxy for traffic volume (originating from connected GPS mobility navigation devices) rather than claiming absolute physical vehicle census counts."*

### Q2: Why is a random `train_test_split` invalid for this task?
> *"Because traffic flow is a time series forecasting problem. A random split causes catastrophic temporal data leakage by interpolating a missing hour from the same day's past and future data. We implement a strict **chronological split**, training on August 11–26 (8.97M rows) and evaluating strictly on unseen future dates August 27–30 (2.39M rows)."*

### Q3: Why did you compute naive persistence baselines?
> *"Time series data often has high auto-correlation. Calculating Naive Lag-1 ($\hat{y}_t = y_{t-1}$) and Naive Seasonal Lag-24 ($\hat{y}_t = y_{t-24}$) establishes the true lower bound of utility. Our models prove their scientific value by demonstrating a **32.6% error reduction** over simple persistence."*

### Q4: Why did Random Forest outperform XGBoost?
> *"Model performance is empirical. On this spatial-temporal tabular feature space (combining tree-depth capacity with rich historical segment priors and multi-scale lag features), Random Forest achieved lower variance on out-of-sample future distributions ($17.81$ MAE vs $18.39$ for XGBoost). We honestly report and select the top empirical performer rather than arbitrarily forcing XGBoost."*

### Q5: Why is congestion analysis separated from road-level ML target prediction?
> *"The dataset provides granular probe counts per road segment, but ground-truth congestion percentages exist only as aggregated city/urban benchmarks. Rather than fabricating arbitrary road-level congestion labels, our model predicts road-level traffic flow and leverages the global metrics for contextual urban congestion and speed degradation analysis."*
