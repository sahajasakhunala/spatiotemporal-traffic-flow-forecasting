# ML Methodology & Scientific Rationale

## 1. Problem Formulation
- **Target**: Next-hour road-level traffic probe flow ($\text{probe\_count}_{t}$).
- **Measurement Interpretation**: Vehicle probe detections per road segment per hour, functioning as an empirical proxy for traffic volume (not absolute vehicle census counts).
- **Time Horizon**: One-step-ahead ($t+1$) rolling forecasting.

## 2. Leakage Prevention Principles
1. **Timestamp-Aware Historical Lags**: Lags ($t-1, t-2, t-3, t-24$) are formed via explicit temporal-grid indexing per `segment_id`. Under no circumstance is future or concurrent data accessible.
2. **Segment Statistics Isolation**: Target encoding and historical segment statistics ($\mu, \sigma, p_{90}, \text{zero\_freq}$) are estimated exclusively from the training date range (August 11–26, 2024).

## 3. Evaluation Protocol
- **Chronological Split**:
  - **Train**: August 11, 2024 to August 26, 2024 (16 days — encompasses Independence Day, Rakshabandhan, and Janmashtami).
  - **Test**: August 27, 2024 to August 30, 2024 (4 out-of-sample days — Tuesday through Friday).
- **Naive Baselines**: Models are strictly benchmarked against:
  - Naive Lag-1 Persistence ($\hat{y}_t = y_{t-1}$)
  - Naive Lag-24 Seasonal Persistence ($\hat{y}_t = y_{t-24}$)
