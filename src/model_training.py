"""
Model Training and Evaluation Module.

Implements:
1. Empirical Naive Baselines:
   - Naive Lag-1 Persistence: y_pred = lag_1
   - Naive Lag-24 Seasonal Persistence: y_pred = lag_24
2. Linear Regression (Baseline Parametric Model)
3. Random Forest Regressor (Non-linear Ensemble)
4. XGBoost Regressor (Gradient Boosted Decision Trees)

Rigorous Evaluation:
- Metrics: MAE, RMSE, R
- Train & Test performance
- Actual vs Predicted visualizations
- Residual error distribution and residual vs predicted analysis
- Feature importance analysis
"""
from pathlib import Path
import time
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from src.config import MODELS_DIR, VISUALIZATIONS_DIR


# Define standardized feature set
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_rush_hour",
    "is_festival",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "speed_limit",
    "frc",
    "distance",
    "probe_count_lag_1",
    "probe_count_lag_2",
    "probe_count_lag_3",
    "probe_count_lag_24",
    "probe_count_roll_mean_3h",
    "segment_mean_traffic",
    "segment_std_traffic",
    "segment_p90_traffic",
    "segment_zero_freq"
]
TARGET_COLUMN = "probe_count"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }


def evaluate_naive_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evaluates empirical naive persistence baselines on the chronological test split."""
    results = {}
    
    # 1. Naive Lag-1 (Next hour will equal previous hour)
    valid_test_lag1 = test_df.dropna(subset=["probe_count", "probe_count_lag_1"])
    lag1_metrics = calculate_metrics(valid_test_lag1["probe_count"].values, valid_test_lag1["probe_count_lag_1"].values)
    results["Naive Lag-1 Persistence"] = lag1_metrics
    
    # 2. Naive Lag-24 (Next hour will equal same hour yesterday)
    valid_test_lag24 = test_df.dropna(subset=["probe_count", "probe_count_lag_24"])
    lag24_metrics = calculate_metrics(valid_test_lag24["probe_count"].values, valid_test_lag24["probe_count_lag_24"].values)
    results["Naive Lag-24 (Seasonal)"] = lag24_metrics
    
    return results


def train_and_evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              models_dir: Path = MODELS_DIR,
                              viz_dir: Path = VISUALIZATIONS_DIR) -> Dict[str, Any]:
    """Trains all models, evaluates performance, exports artifacts and visual diagnostics."""
    models_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean NaNs resulting from initial lag offsets
    clean_train = train_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    clean_test = test_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    
    print(f"Clean Training Set Rows: {len(clean_train):,}")
    print(f"Clean Test Set Rows: {len(clean_test):,}")
    
    X_train = clean_train[FEATURE_COLUMNS].values
    y_train = clean_train[TARGET_COLUMN].values
    X_test = clean_test[FEATURE_COLUMNS].values
    y_test = clean_test[TARGET_COLUMN].values
    
    results = {}
    
    # Evaluate naive baselines first
    print("\n--- Evaluating Naive Empirical Baselines ---")
    naive_results = evaluate_naive_baselines(clean_train, clean_test)
    for model_name, metrics in naive_results.items():
        print(f"[{model_name}] Test MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}, R: {metrics['R2']}")
        results[model_name] = {"train": None, "test": metrics, "time_sec": 0.0}
        
    # --- 1. Linear Regression Baseline ---
    print("\n--- Training Model 1: Linear Regression (Ridge L2 Regularized) ---")
    t0 = time.time()
    lr = Ridge(alpha=1.0)
    lr.fit(X_train, y_train)
    lr_time = time.time() - t0
    
    y_pred_train_lr = lr.predict(X_train)
    y_pred_test_lr = lr.predict(X_test)
    
    results["Linear Regression"] = {
        "model": lr,
        "train": calculate_metrics(y_train, y_pred_train_lr),
        "test": calculate_metrics(y_test, y_pred_test_lr),
        "time_sec": round(lr_time, 2),
        "y_pred_test": y_pred_test_lr
    }
    joblib.dump(lr, models_dir / "linear_regression.joblib")
    print(f"Linear Regression trained in {lr_time:.2f}s | Test MAE: {results['Linear Regression']['test']['MAE']}, RMSE: {results['Linear Regression']['test']['RMSE']}, R: {results['Linear Regression']['test']['R2']}")
    
    # --- 2. Random Forest Regressor ---
    print("\n--- Training Model 2: Random Forest Regressor ---")
    # Subsample if training set is extremely large for RF memory/time constraints
    if len(clean_train) > 500000:
        rf_sample = clean_train.sample(n=500000, random_state=42)
        X_tr_rf = rf_sample[FEATURE_COLUMNS].values
        y_tr_rf = rf_sample[TARGET_COLUMN].values
    else:
        X_tr_rf, y_tr_rf = X_train, y_train
        
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=10, n_jobs=-1, random_state=42)
    rf.fit(X_tr_rf, y_tr_rf)
    rf_time = time.time() - t0
    
    y_pred_train_rf = rf.predict(X_tr_rf)
    y_pred_test_rf = rf.predict(X_test)
    
    results["Random Forest"] = {
        "model": rf,
        "train": calculate_metrics(y_tr_rf, y_pred_train_rf),
        "test": calculate_metrics(y_test, y_pred_test_rf),
        "time_sec": round(rf_time, 2),
        "y_pred_test": y_pred_test_rf
    }
    joblib.dump(rf, models_dir / "random_forest.joblib")
    print(f"Random Forest trained in {rf_time:.2f}s | Test MAE: {results['Random Forest']['test']['MAE']}, RMSE: {results['Random Forest']['test']['RMSE']}, R: {results['Random Forest']['test']['R2']}")
    
    # --- 3. XGBoost Regressor ---
    print("\n--- Training Model 3: XGBoost Regressor ---")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - t0
    
    y_pred_train_xgb = xgb_model.predict(X_train)
    y_pred_test_xgb = xgb_model.predict(X_test)
    
    results["XGBoost"] = {
        "model": xgb_model,
        "train": calculate_metrics(y_train, y_pred_train_xgb),
        "test": calculate_metrics(y_test, y_pred_test_xgb),
        "time_sec": round(xgb_time, 2),
        "y_pred_test": y_pred_test_xgb
    }
    joblib.dump(xgb_model, models_dir / "xgboost.joblib")
    print(f"XGBoost trained in {xgb_time:.2f}s | Test MAE: {results['XGBoost']['test']['MAE']}, RMSE: {results['XGBoost']['test']['RMSE']}, R: {results['XGBoost']['test']['R2']}")
    
    # --- Visual Model Diagnostics & Comparisons ---
    generate_model_diagnostics(results, y_test, clean_test, viz_dir)
    
    return results


def generate_model_diagnostics(results: Dict[str, Any], y_test: np.ndarray, test_df: pd.DataFrame, viz_dir: Path):
    """Generates comparison bar charts, actual vs predicted scatter plots, residual plots, and feature importances."""
    print("\nGenerating model evaluation diagnostic plots...")
    
    # 1. Metric Comparison Bar Chart
    models = list(results.keys())
    mae_vals = [results[m]["test"]["MAE"] for m in models]
    rmse_vals = [results[m]["test"]["RMSE"] for m in models]
    r2_vals = [results[m]["test"]["R2"] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.barplot(x=models, y=mae_vals, palette="Blues_r", ax=axes[0])
    axes[0].set_title("Test MAE (Lower is Better)")
    axes[0].tick_params(axis='x', rotation=30)
    for i, v in enumerate(mae_vals): axes[0].text(i, v + 0.05, str(v), ha="center", fontweight="bold")
    
    sns.barplot(x=models, y=rmse_vals, palette="Reds_r", ax=axes[1])
    axes[1].set_title("Test RMSE (Lower is Better)")
    axes[1].tick_params(axis='x', rotation=30)
    for i, v in enumerate(rmse_vals): axes[1].text(i, v + 0.05, str(v), ha="center", fontweight="bold")
    
    sns.barplot(x=models, y=r2_vals, palette="Greens", ax=axes[2])
    axes[2].set_title("Test R Score (Higher is Better)")
    axes[2].tick_params(axis='x', rotation=30)
    for i, v in enumerate(r2_vals): axes[2].text(i, v + 0.01, str(v), ha="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(viz_dir / "11_model_metrics_comparison.png", dpi=300)
    plt.close()
    
    # 2. Actual vs Predicted (Best Model - XGBoost or Random Forest)
    best_model_name = "XGBoost" if "XGBoost" in results else "Random Forest"
    best_pred = results[best_model_name]["y_pred_test"]
    
    # Sample for scatter visualization
    sample_indices = np.random.choice(len(y_test), size=min(10000, len(y_test)), replace=False)
    y_sub = y_test[sample_indices]
    pred_sub = best_pred[sample_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_sub, pred_sub, alpha=0.3, color="#1f77b4", s=10)
    max_val = max(np.percentile(y_sub, 99.5), np.percentile(pred_sub, 99.5))
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label="Perfect 45 Fit")
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_title(f"Actual vs Predicted Traffic Flow ({best_model_name})")
    axes[0].set_xlabel("Actual Probe Count")
    axes[0].set_ylabel("Predicted Next-Hour Probe Count")
    axes[0].legend()
    
    # Residual error distribution
    residuals = y_sub - pred_sub
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color="#9467bd")
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_title(f"Residual Error Distribution (Actual - Pred)")
    axes[1].set_xlabel("Error (Probe Count)")
    plt.tight_layout()
    plt.savefig(viz_dir / "12_actual_vs_predicted_and_residuals.png", dpi=300)
    plt.close()
    
    # 3. Feature Importance (XGBoost)
    if "XGBoost" in results and hasattr(results["XGBoost"]["model"], "feature_importances_"):
        fi = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": results["XGBoost"]["model"].feature_importances_
        }).sort_values("importance", ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=fi.head(15), x="importance", y="feature", palette="viridis", ax=ax)
        ax.set_title("XGBoost Top 15 Feature Importances (Gini Gain)")
        ax.set_xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(viz_dir / "13_xgboost_feature_importance.png", dpi=300)
        plt.close()
