import json
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# ?? Intelligent Urban Traffic Flow & Congestion Prediction System
## End-to-End Machine Learning Pipeline Notebook

This notebook provides an interactive walkthrough of the machine learning pipeline:
- Ingestion of 11.97M spatial-temporal observations across 24,938 New Delhi road segments.
- Chronological evaluation on 2.39M test records (Aug 2730, 2024).
- Empirical benchmarking of Naive Persistence, OLS, Ridge, Random Forest, and XGBoost models.
"""),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import MODELS_DIR, VISUALIZATIONS_DIR, TRAIN_START_DATE, TEST_END_DATE
from src.prediction import TrafficPredictor

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Pipeline modules loaded successfully.")
"""),
    nbf.v4.new_markdown_cell("""### ?? 1. Official Model Evaluation Benchmarks (Chronological Test Split: Aug 2730)"""),
    nbf.v4.new_code_cell("""with open(MODELS_DIR / "model_evaluation_metrics.json", "r") as f:
    metrics = json.load(f)

benchmark_df = pd.DataFrame([
    {
        "Model Architecture": k,
        "Test MAE": v["test_metrics"]["MAE"],
        "Test RMSE": v["test_metrics"]["RMSE"],
        "Test R2": v["test_metrics"]["R2"],
        "Training Time (s)": v.get("training_time_sec", 0.0)
    }
    for k, v in metrics.items()
]).sort_values("Test MAE")

benchmark_df.reset_index(drop=True)
"""),
    nbf.v4.new_markdown_cell("""### ?? 2. Next-Hour Inference Pipeline Verification
Testing on-demand next-hour prediction for peak evening rush hour (6:00 PM):
"""),
    nbf.v4.new_code_cell("""predictor = TrafficPredictor()
test_pred = predictor.predict_next_hour(
    segment_id=-13560111507837,
    date_str="2024-08-28",
    hour=18,
    recent_lags=[15.0, 14.0, 12.0, 16.0],
    model_name="random_forest"
)
print("Inference Output:")
print(json.dumps(test_pred, indent=2))
""")
]

nb["cells"] = cells
with open("notebooks/traffic_flow_prediction_pipeline.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Jupyter Notebook created at notebooks/traffic_flow_prediction_pipeline.ipynb")
