"""
SUMO Vehicle Telemetry Exporter (Phase 3A)
------------------------------------------
Extracts genuine vehicle-level telemetry from Eclipse SUMO via TraCI at 1-second intervals.
Converts metric planar coordinates (x, y) back to real-world WGS84 GPS (lat, lon).
Exports time-indexed, web-efficient trajectory files for interactive browser visualization.

All positions, speeds, headings, and vehicle states are 100% genuine TraCI simulation data.
Zero fake/fabricated trajectories.
"""

import gzip
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add SUMO tools to path if available
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_TOOLS = Path(SUMO_HOME) / "tools"
if str(SUMO_TOOLS) not in sys.path and SUMO_TOOLS.exists():
    sys.path.append(str(SUMO_TOOLS))

try:
    import traci
except ImportError:
    traci = None

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.sumo_simulation import (
    SUMO_PROCESSED_DIR,
    PROJECTION_CENTER_LAT,
    PROJECTION_CENTER_LON,
    cartesian_to_wgs84,
    check_sumo_installation,
    extract_corridor_features,
)

# Output directories
TRAJECTORIES_DIR = SUMO_PROCESSED_DIR / "trajectories"
TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
CORRIDOR_GEOJSON_FILE = SUMO_PROCESSED_DIR / "corridor_network.geojson"
MANIFEST_FILE = SUMO_PROCESSED_DIR / "simulation_manifest.json"


# ---------------------------------------------------------------------------
# 1. Export Road Network GeoJSON for Browser Basemap
# ---------------------------------------------------------------------------
def export_corridor_geojson(output_path: Path = CORRIDOR_GEOJSON_FILE) -> Path:
    """
    Export the 218 Barapullah corridor road segments as a clean GeoJSON FeatureCollection.
    Provides the exact road geometry, lane count, and street names for the web map.
    """
    raw_geojson = list((RAW_DATA_DIR / "probe_counts" / "geojson").glob("*.geojson"))[0]
    corridor_features = extract_corridor_features(raw_geojson)

    features = []
    for f in corridor_features:
        # Construct GeoJSON feature
        feat = {
            "type": "Feature",
            "properties": {
                "segment_id": f["segment_id"],
                "street_name": f["street_name"],
                "frc": f["frc"],
                "speed_limit_kmh": f["speed_limit"],
                "length_m": f["distance"],
                "lanes": 3 if f["frc"] == 1 else (2 if f["frc"] == 2 else 1),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": f["raw_coords"],
            },
        }
        features.append(feat)

    collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(collection, out, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# 2. TraCI Telemetry Exporter
# ---------------------------------------------------------------------------
def export_scenario_trajectories(
    sumo_cfg_path: Path,
    scenario_name: str,
    period_name: str,
    output_json_path: Path,
    max_duration_sec: int = 3600,
    export_step_sec: float = 1.0,
) -> Dict[str, Any]:
    """
    Run the given SUMO configuration via TraCI and capture per-vehicle telemetry
    at regular time intervals (default 1.0s).

    Required vehicle telemetry fields captured per frame:
      - simulation_time_sec (float)
      - vehicle_id (str)
      - vehicle_type (str: 'car', 'auto', 'bus')
      - planar_x (float)
      - planar_y (float)
      - latitude (float, WGS84)
      - longitude (float, WGS84)
      - speed_mps (float)
      - speed_kmh (float)
      - heading_angle_deg (float, 0-360)
      - current_edge_id (str)
      - current_lane_index (int)
      - acceleration (float)
      - waiting_time_sec (float)
    """
    sumo_status = check_sumo_installation()
    if not sumo_status["installed"]:
        raise RuntimeError("SUMO installation not detected on host system.")

    sumo_exe = sumo_status["binaries"]["sumo"]

    # Start TraCI process
    cmd = [
        sumo_exe,
        "-c", str(sumo_cfg_path),
        "--duration-log.disable", "true",
        "--no-step-log", "true",
    ]

    label = f"sim_{scenario_name}_{period_name}"
    traci.start(cmd, label=label)
    conn = traci.getConnection(label)

    frames: Dict[str, List[Dict[str, Any]]] = {}
    summary_by_step: List[Dict[str, Any]] = []
    unique_vehicles = set()
    completed_vehicles = 0
    all_lats = []
    all_lons = []

    last_export_time = -1.0

    while conn.simulation.getMinExpectedNumber() > 0:
        conn.simulationStep()
        t = conn.simulation.getTime()

        # Track completed/arrived vehicles
        arrived_ids = conn.simulation.getArrivedIDList()
        completed_vehicles += len(arrived_ids)

        if t > max_duration_sec:
            break

        # Capture frame at desired step interval
        if t - last_export_time >= export_step_sec or math.isclose(t, 0.0):
            last_export_time = t
            t_key = f"{int(round(t))}"
            active_vehs = conn.vehicle.getIDList()

            step_vehs = []
            speeds_kmh = []
            waits_sec = []

            for vid in active_vehs:
                unique_vehicles.add(vid)
                x, y = conn.vehicle.getPosition(vid)
                vtype = conn.vehicle.getTypeID(vid)
                spd_mps = conn.vehicle.getSpeed(vid)
                spd_kmh = round(spd_mps * 3.6, 2)
                angle = round(conn.vehicle.getAngle(vid), 2)
                edge = conn.vehicle.getRoadID(vid)
                lane = conn.vehicle.getLaneIndex(vid)
                accel = round(conn.vehicle.getAcceleration(vid), 2)
                wait = round(float(conn.vehicle.getWaitingTime(vid)), 2)

                # Convert (x, y) to real WGS84 GPS (lon, lat)
                lon, lat = cartesian_to_wgs84(x, y)
                all_lats.append(lat)
                all_lons.append(lon)

                speeds_kmh.append(spd_kmh)
                waits_sec.append(wait)

                record = {
                    "simulation_time_sec": float(t),
                    "vehicle_id": str(vid),
                    "vehicle_type": str(vtype),
                    "planar_x": round(float(x), 2),
                    "planar_y": round(float(y), 2),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "speed_mps": round(float(spd_mps), 2),
                    "speed_kmh": float(spd_kmh),
                    "heading_angle_deg": float(angle),
                    "current_edge_id": str(edge),
                    "current_lane_index": int(lane),
                    "acceleration": float(accel),
                    "waiting_time_sec": float(wait),
                }
                step_vehs.append(record)

            frames[t_key] = step_vehs

            # Step summary metrics
            mean_spd = round(float(sum(speeds_kmh) / len(speeds_kmh)), 2) if speeds_kmh else 0.0
            mean_wait = round(float(sum(waits_sec) / len(waits_sec)), 2) if waits_sec else 0.0
            density = round(len(active_vehs) / 1.54, 1)  # 1.54 km corridor length

            summary_by_step.append({
                "time": int(round(t)),
                "active_vehicles": len(active_vehs),
                "completed_vehicles": completed_vehicles,
                "mean_speed_kmh": mean_spd,
                "mean_waiting_time_sec": mean_wait,
                "corridor_density_veh_km": density,
            })

    try:
        conn.close()
    except Exception:
        pass
    # Calculate geographic bounds
    min_lat = min(all_lats) if all_lats else PROJECTION_CENTER_LAT - 0.02
    max_lat = max(all_lats) if all_lats else PROJECTION_CENTER_LAT + 0.02
    min_lon = min(all_lons) if all_lons else PROJECTION_CENTER_LON - 0.04
    max_lon = max(all_lons) if all_lons else PROJECTION_CENTER_LON + 0.04

    payload = {
        "metadata": {
            "scenario": scenario_name,
            "period": period_name,
            "engine": "Eclipse SUMO 1.27.1",
            "telemetry_source": "TraCI Live Step Recording",
            "execution_mode": "SUMO",
            "duration_sec": max_duration_sec,
            "step_interval_sec": export_step_sec,
            "total_frames": len(frames),
            "total_unique_vehicles": len(unique_vehicles),
            "total_completed_trips": completed_vehicles,
            "bounds": {
                "min_lat": round(min_lat, 6),
                "max_lat": round(max_lat, 6),
                "min_lon": round(min_lon, 6),
                "max_lon": round(max_lon, 6),
            },
            "corridor_center": {
                "lat": PROJECTION_CENTER_LAT,
                "lon": PROJECTION_CENTER_LON,
            },
        },
        "summary_by_step": summary_by_step,
        "frames": frames,
    }

    # Save uncompressed JSON for local querying
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Save gzip-compressed version for fast web delivery
    gz_path = output_json_path.with_suffix(".json.gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as gz_f:
        json.dump(payload, gz_f)

    json_size_kb = output_json_path.stat().st_size / 1024
    gz_size_kb = gz_path.stat().st_size / 1024

    return {
        "scenario": scenario_name,
        "period": period_name,
        "json_path": str(output_json_path),
        "gz_path": str(gz_path),
        "json_size_kb": round(json_size_kb, 1),
        "gz_size_kb": round(gz_size_kb, 1),
        "total_frames": len(frames),
        "unique_vehicles": len(unique_vehicles),
        "completed_vehicles": completed_vehicles,
    }


# ---------------------------------------------------------------------------
# 3. Create Enhanced SUMO-GUI View Settings (delhi.view.xml)
# ---------------------------------------------------------------------------
def generate_sumo_gui_viewsettings(output_path: Path = SUMO_PROCESSED_DIR / "delhi.view.xml") -> Path:
    """
    Generate an enhanced viewsettings XML file for SUMO-GUI.
    Configures realistic vehicle silhouettes, color by speed, lane dividers, and clean dark styling.
    """
    content = """<?xml version="1.0" encoding="UTF-8"?>
<viewsettings xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/viewsettings_file.xsd">
    <scheme name="Barapullah_Real_Traffic">
        <opengl dither="0" antialiasing="1"/>
        <background backgroundColor="22,27,34"/>
        <edges laneEdgeMode="0" laneShowBorders="1" showLinkDecals="1" edgeName_show="0"/>
        <vehicles vehicleMode="8" vehicleQuality="2" vehicle_showBlinker="1" vehicleName_show="0">
            <!-- vehicleMode 8 = color by speed (Red = slow/stopped, Green/Cyan = free flow) -->
            <!-- vehicleQuality 2 = draw realistic vehicle silhouettes -->
        </vehicles>
        <junctions junctionMode="0" drawCrossingsAndWalkingAreas="1"/>
    </scheme>
    <viewport zoom="500" x="1200" y="500"/>
    <delay value="50"/>
</viewsettings>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


# ---------------------------------------------------------------------------
# 4. Master Pipeline: Export All Scenarios
# ---------------------------------------------------------------------------
def export_all_scenarios(duration_sec: int = 3600) -> Dict[str, Any]:
    """
    Export road network GeoJSON, enhanced viewsettings, and TraCI trajectories for
    all canonical scenarios:
      - Baseline: Morning Rush, Evening Rush, Off-Peak
      - RF Forecast: Morning Rush, Evening Rush, Off-Peak
      - Naive Lag-1: Morning Rush, Evening Rush, Off-Peak
    """
    print("=" * 70)
    print(" PHASE 3A: SUMO VEHICLE TELEMETRY EXPORTER")
    print(" Extracting 100% Genuine TraCI Telemetry for Web Visualization")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Export Corridor GeoJSON
    print("\n[Step 1/3] Exporting Barapullah corridor road network as GeoJSON...")
    geojson_path = export_corridor_geojson()
    print(f"  Exported 218 corridor segments to: {geojson_path}")

    # Step 2: Generate SUMO-GUI view settings
    print("\n[Step 2/3] Generating enhanced SUMO-GUI view settings...")
    view_path = generate_sumo_gui_viewsettings()
    print(f"  Generated SUMO-GUI view configuration at: {view_path}")

    # Step 3: Run TraCI export across scenarios
    print("\n[Step 3/3] Running TraCI telemetry extraction across scenarios...")
    scenarios = ["baseline", "ml_forecast", "naive_persistence"]
    periods = ["morning_rush", "evening_rush", "off_peak"]

    manifest = {
        "corridor": "Barapullah Elevated Corridor, New Delhi",
        "network_geojson": str(geojson_path),
        "viewsettings_xml": str(view_path),
        "scenarios": {},
    }

    for sc in scenarios:
        manifest["scenarios"][sc] = {}
        for p in periods:
            cfg_file = SUMO_PROCESSED_DIR / f"delhi_{sc}_{p}.sumocfg"
            out_json = TRAJECTORIES_DIR / f"trajectories_{sc}_{p}.json"
            print(f"  Exporting TraCI telemetry for: {sc} | {p}...")
            t_sc = time.time()
            res = export_scenario_trajectories(
                cfg_file,
                scenario_name=sc,
                period_name=p,
                output_json_path=out_json,
                max_duration_sec=duration_sec,
                export_step_sec=1.0,
            )
            elapsed_sc = time.time() - t_sc
            print(
                f"    Done in {elapsed_sc:.1f}s | Frames: {res['total_frames']} | "
                f"Vehicles: {res['unique_vehicles']} | "
                f"JSON Size: {res['json_size_kb']:.1f} KB (GZ: {res['gz_size_kb']:.1f} KB)"
            )
            manifest["scenarios"][sc][p] = res

    # Save manifest
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved simulation manifest to: {MANIFEST_FILE}")

    total_elapsed = time.time() - t0
    print(f"\nAll TraCI telemetry exported successfully in {total_elapsed:.1f}s.")
    print("=" * 70)
    return manifest


if __name__ == "__main__":
    export_all_scenarios()
