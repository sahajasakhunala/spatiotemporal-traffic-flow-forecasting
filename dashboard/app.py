"""
Flask Interactive Dashboard Application.
Serves interactive Next-Hour Traffic Flow Prediction, Historical Analytics,
Congestion/Speed Heatmap Visualizations, and an embedded Folium Delhi NCR Map.
"""
from pathlib import Path
import json
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import folium

from src.config import BASE_DIR, MODELS_DIR, VISUALIZATIONS_DIR, PROCESSED_DATA_DIR, GLOBAL_METRICS_DIR
from src.prediction import TrafficPredictor

app = Flask(__name__,
            template_folder=str(BASE_DIR / "dashboard" / "templates"),
            static_folder=str(BASE_DIR / "dashboard" / "static"))

# Initialize predictor
predictor = TrafficPredictor(models_dir=MODELS_DIR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segments")
def get_segments():
    """Returns top accessible road segments for dropdown selection."""
    if predictor.segment_metadata is not None:
        # Sort by mean traffic to showcase high-profile roads first
        top_segs = predictor.segment_metadata.sort_values("segment_mean_traffic", ascending=False).head(200)
        records = top_segs[["segment_id", "street_name", "speed_limit", "frc", "segment_mean_traffic", "latitude", "longitude"]].to_dict(orient="records")
        return jsonify({"status": "success", "segments": records})
    return jsonify({"status": "error", "message": "Segment metadata not loaded"}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """Performs next-hour traffic flow inference."""
    data = request.get_json() or {}
    try:
        segment_id = int(data.get("segment_id", 0))
        date_str = data.get("date", "2024-08-28")
        hour = int(data.get("hour", 18))
        model_name = data.get("model", "xgboost")
        
        # User supplied or simulated lags
        lag_1 = float(data.get("lag_1", 8.0))
        lag_2 = float(data.get("lag_2", 7.0))
        lag_3 = float(data.get("lag_3", 6.0))
        lag_24 = float(data.get("lag_24", 8.0))
        
        result = predictor.predict_next_hour(
            segment_id=segment_id,
            date_str=date_str,
            hour=hour,
            recent_lags=[lag_1, lag_2, lag_3, lag_24],
            model_name=model_name
        )
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/metrics")
def get_metrics():
    """Returns model benchmark metrics."""
    metrics_path = MODELS_DIR / "model_evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"status": "error", "message": "Metrics not generated yet"}), 404


@app.route("/api/congestion")
def get_congestion():
    """Returns urban congestion benchmarks and rush hour stats."""
    city_rush_path = GLOBAL_METRICS_DIR / "2024_city_rush_hour.json"
    city_traffic_path = GLOBAL_METRICS_DIR / "new_delhi_2024_city_traffic.json"
    
    res = {}
    if city_rush_path.exists():
        with open(city_rush_path, "r") as f:
            res["rush_hour"] = json.load(f)
    if city_traffic_path.exists():
        with open(city_traffic_path, "r") as f:
            res["city_traffic"] = json.load(f)
    return jsonify(res)


@app.route("/api/map")
def get_map():
    """Generates an embedded interactive Folium map of New Delhi road segments."""
    # Centered on New Delhi
    m = folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="CartoDB positron")
    
    if predictor.segment_metadata is not None:
        sample_map_segs = predictor.segment_metadata.dropna(subset=["latitude", "longitude"]).head(150)
        for _, row in sample_map_segs.iterrows():
            mean_vol = float(row.get("segment_mean_traffic", 0))
            color = "#d9534f" if mean_vol > 10 else "#f0ad4e" if mean_vol > 4 else "#5cb85c"
            
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4 + min(mean_vol, 12),
                popup=f"<b>{row['street_name']}</b><br>FRC: {row['frc']}<br>Speed Limit: {row['speed_limit']} km/h<br>Avg Probe Flow: {mean_vol:.1f}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)
            
    return m._repr_html_()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
