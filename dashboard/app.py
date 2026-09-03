"""
Flask Interactive Dashboard Application.
Serves interactive Next-Hour Traffic Flow Prediction, Historical Analytics,
Congestion/Speed Heatmap Visualizations, embedded Folium Delhi NCR Map,
and Phase 3 Interactive SUMO Vehicle-Level Microsimulation.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import gzip
from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import folium

from src.config import BASE_DIR, MODELS_DIR, VISUALIZATIONS_DIR, PROCESSED_DATA_DIR, GLOBAL_METRICS_DIR
from src.prediction import TrafficPredictor
from src.sumo_simulation import SUMO_PROCESSED_DIR

app = Flask(__name__,
            template_folder=str(BASE_DIR / "dashboard" / "templates"),
            static_folder=str(BASE_DIR / "dashboard" / "static"))

# Initialize predictor
predictor = TrafficPredictor(models_dir=MODELS_DIR)


# ---------------------------------------------------------------------------
# Core ML & Analytics Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segments")
def get_segments():
    """Returns top accessible road segments for dropdown selection."""
    if predictor.segment_metadata is not None:
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


# ---------------------------------------------------------------------------
# Phase 3: SUMO Microsimulation Interactive Web Routes
# ---------------------------------------------------------------------------
@app.route("/simulation")
def simulation():
    """Serves the interactive SUMO vehicle microsimulation dashboard page."""
    return render_template("simulation.html")


@app.route("/api/simulation/manifest")
def get_simulation_manifest():
    """Returns the manifest of available SUMO trajectory scenarios."""
    manifest_path = SUMO_PROCESSED_DIR / "simulation_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"status": "error", "message": "Simulation manifest not found"}), 404


@app.route("/api/simulation/network")
def get_simulation_network():
    """Returns the GeoJSON FeatureCollection of the 218 Barapullah road corridor segments."""
    network_path = SUMO_PROCESSED_DIR / "corridor_network.geojson"
    if network_path.exists():
        with open(network_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"status": "error", "message": "Corridor network GeoJSON not found"}), 404


@app.route("/api/simulation/trajectories")
def get_simulation_trajectories():
    """
    Returns time-indexed vehicle telemetry frames for the requested scenario and period.
    Supports gzip transport compression for fast web delivery.
    """
    scenario = request.args.get("scenario", "ml_forecast")
    period = request.args.get("period", "morning_rush")

    # Map aliases if needed
    if scenario == "rf_forecast":
        scenario = "ml_forecast"
    elif scenario == "naive_lag1":
        scenario = "naive_persistence"

    trajectories_dir = SUMO_PROCESSED_DIR / "trajectories"

    # Check for experiment query
    experiment = request.args.get("experiment")
    if experiment:
        gz_file = trajectories_dir / f"trajectories_experiment_{experiment}.json.gz"
        json_file = trajectories_dir / f"trajectories_experiment_{experiment}.json"
    else:
        gz_file = trajectories_dir / f"trajectories_{scenario}_{period}.json.gz"
        json_file = trajectories_dir / f"trajectories_{scenario}_{period}.json"

    # Efficient gzip delivery if supported by client
    accept_encoding = request.headers.get("Accept-Encoding", "")
    if "gzip" in accept_encoding and gz_file.exists():
        response = send_file(gz_file, mimetype="application/json")
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response
    elif json_file.exists():
        return send_file(json_file, mimetype="application/json")
    elif gz_file.exists():
        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            return jsonify(json.load(f))

    return jsonify({
        "status": "error",
        "message": f"Trajectory not found for scenario '{scenario}' and period '{period}' (experiment: {experiment})"
    }), 404


@app.route("/api/simulation/experiments")
def get_simulation_experiments():
    """
    Returns available disturbance experiment scenarios, target classifications,
    and pre-computed closed-loop evaluation metrics comparing RF baseline forecast vs actual SUMO state.
    """
    trajectories_dir = SUMO_PROCESSED_DIR / "trajectories"
    experiment_types = ["normal", "accident", "lane_closure", "heavy_vehicle", "signal_disruption"]
    experiments = []

    for exp in experiment_types:
        json_path = trajectories_dir / f"trajectories_experiment_{exp}.json"
        gz_path = trajectories_dir / f"trajectories_experiment_{exp}.json.gz"
        data = None
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                data = json.load(f)

        if data and "evaluation" in data:
            experiments.append({
                "id": exp,
                "name": data["evaluation"]["scenario_name"],
                "disturbance": data["evaluation"]["disturbance"],
                "target_classification": data["evaluation"]["target_classification"],
                "metrics": data["evaluation"]["metrics"],
                "total_unique_vehicles": data["metadata"]["total_unique_vehicles"],
                "duration_sec": data["metadata"]["duration_sec"]
            })

    return jsonify({
        "status": "success",
        "baseline_model": "Random Forest Regressor (Aug 11-26 Training)",
        "simulation_engine": "Eclipse SUMO 1.27.1 / TraCI",
        "experiments": experiments
    })


@app.route("/api/simulation/run_experiment", methods=["POST"])
def run_simulation_experiment():
    """
    Dynamically executes an actual SUMO microsimulation with injected disturbance via TraCI.
    Request JSON parameters:
      - disturbance_type: "none", "accident", "lane_closure", "heavy_vehicle", "signal_disruption"
      - start_time: int (seconds, default 240)
      - duration: int (seconds, default 360)
      - edge_id: str (default "edge_13560341041261")
      - lane_index: int (default 0)
      - max_duration_sec: int (default 1200)
    """
    from src.sumo_exporter import export_disturbance_experiment

    payload = request.get_json() or {}
    dist_type = payload.get("disturbance_type", "accident")
    start_time = int(payload.get("start_time", 240))
    duration = int(payload.get("duration", 360))
    edge_id = payload.get("edge_id", "edge_13560341041261")
    lane_index = int(payload.get("lane_index", 0))
    max_duration = int(payload.get("max_duration_sec", 1200))

    cfg_file = SUMO_PROCESSED_DIR / "delhi_ml_forecast_morning_rush.sumocfg"
    out_json = SUMO_PROCESSED_DIR / "trajectories" / f"trajectories_experiment_custom.json"

    try:
        res = export_disturbance_experiment(
            sumo_cfg_path=cfg_file,
            experiment_name="custom",
            output_json_path=out_json,
            disturbance_type=dist_type,
            title=f"Custom {dist_type.capitalize()} Experiment",
            description=f"User-configured disturbance: {dist_type} on {edge_id} for {duration}s",
            start_time=start_time,
            duration=duration,
            edge_id=edge_id,
            lane_index=lane_index,
            max_duration_sec=max_duration,
        )
        return jsonify({
            "status": "success",
            "experiment_id": "custom",
            "evaluation": res.get("evaluation", {}),
            "metadata": res.get("metadata", {})
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/simulation/buildings")
def get_simulation_buildings():
    """Returns real 3D building footprints along the Barapullah corridor."""
    bld_path = SUMO_PROCESSED_DIR / "barapullah_buildings.geojson"
    if bld_path.exists():
        with open(bld_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"type": "FeatureCollection", "features": []})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
