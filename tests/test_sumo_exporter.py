"""
Unit and Integration Test Suite for SUMO TraCI Telemetry Exporter and Web Visualizer (Phase 3)
-----------------------------------------------------------------------------------------------
Verifies:
  1. Telemetry exporter execution & manifest generation
  2. Required vehicle telemetry fields completeness
  3. No fabricated coordinates (strict bounding box & projection validity)
  4. Valid scenario trajectory files across all 9 scenarios
  5. Chronological time-frame ordering (0 to 3600s)
  6. Vehicle type mapping ('car', 'auto', 'bus')
  7. Playback data integrity & frame consistency
  8. SUMO execution mode validation ('SUMO')
  9. Flask web simulation endpoints (/simulation, /api/simulation/*)
"""

import gzip
import json
import unittest
from pathlib import Path

from src.config import PROCESSED_DATA_DIR
from src.sumo_exporter import (
    CORRIDOR_GEOJSON_FILE,
    MANIFEST_FILE,
    TRAJECTORIES_DIR,
)
from src.sumo_simulation import (
    PROJECTION_CENTER_LAT,
    PROJECTION_CENTER_LON,
    SUMO_PROCESSED_DIR,
    cartesian_to_wgs84,
    project_wgs84_to_cartesian,
)
from dashboard.app import app


class TestSumoExporterAndVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectories_dir = TRAJECTORIES_DIR
        cls.manifest_file = MANIFEST_FILE
        cls.geojson_file = CORRIDOR_GEOJSON_FILE
        cls.viewsettings_file = SUMO_PROCESSED_DIR / "delhi.view.xml"
        cls.client = app.test_client()

        # Load RF Morning Rush trajectory as canonical test sample
        cls.sample_json_file = cls.trajectories_dir / "trajectories_ml_forecast_morning_rush.json"
        if cls.sample_json_file.exists():
            with open(cls.sample_json_file, "r", encoding="utf-8") as f:
                cls.sample_data = json.load(f)
        else:
            cls.sample_data = None

    def test_01_telemetry_exporter_artifacts_exist(self):
        """1. Verify exporter produced manifest, network GeoJSON, viewsettings, and trajectory files."""
        self.assertTrue(self.geojson_file.exists(), "Corridor network GeoJSON missing")
        self.assertTrue(self.manifest_file.exists(), "Simulation manifest missing")
        self.assertTrue(self.viewsettings_file.exists(), "SUMO-GUI viewsettings missing")
        self.assertTrue(self.sample_json_file.exists(), "Sample trajectory JSON missing")

        # Verify GeoJSON has 218 features
        with open(self.geojson_file, "r", encoding="utf-8") as f:
            geo_data = json.load(f)
        self.assertEqual(geo_data["type"], "FeatureCollection")
        self.assertEqual(len(geo_data["features"]), 218)

    def test_02_required_vehicle_telemetry_fields(self):
        """2. Verify that every captured vehicle record contains all required telemetry fields."""
        self.assertIsNotNone(self.sample_data, "Sample trajectory data not loaded")
        frames = self.sample_data.get("frames", {})
        self.assertGreater(len(frames), 0, "Frames dictionary must not be empty")

        # Find first non-empty frame
        sample_vehicle = None
        for step_key, veh_list in frames.items():
            if veh_list:
                sample_vehicle = veh_list[0]
                break

        self.assertIsNotNone(sample_vehicle, "At least one active vehicle record must exist")

        required_fields = [
            "simulation_time_sec",
            "vehicle_id",
            "vehicle_type",
            "planar_x",
            "planar_y",
            "latitude",
            "longitude",
            "speed_mps",
            "speed_kmh",
            "heading_angle_deg",
            "current_edge_id",
            "current_lane_index",
            "acceleration",
            "waiting_time_sec",
        ]

        for field in required_fields:
            self.assertIn(field, sample_vehicle, f"Field '{field}' missing from vehicle telemetry")

    def test_03_no_fabricated_coordinates(self):
        """3. Verify all coordinates fall strictly within real Barapullah geographic bounds with zero fabrication."""
        self.assertIsNotNone(self.sample_data)
        frames = self.sample_data.get("frames", {})

        # Barapullah Corridor bounding box: Lat [28.55, 28.62], Lon [77.18, 77.29]
        min_valid_lat, max_valid_lat = 28.55, 28.62
        min_valid_lon, max_valid_lon = 77.18, 77.29

        checked_records = 0
        for step_key in list(frames.keys())[::100]:  # sample across time
            for v in frames[step_key]:
                lat = v["latitude"]
                lon = v["longitude"]
                self.assertTrue(
                    min_valid_lat <= lat <= max_valid_lat,
                    f"Vehicle {v['vehicle_id']} latitude {lat} outside valid corridor bounds"
                )
                self.assertTrue(
                    min_valid_lon <= lon <= max_valid_lon,
                    f"Vehicle {v['vehicle_id']} longitude {lon} outside valid corridor bounds"
                )

                # Verify planar coordinates match GPS via inverse projection
                inv_lon, inv_lat = cartesian_to_wgs84(v["planar_x"], v["planar_y"])
                self.assertAlmostEqual(lat, inv_lat, places=4)
                self.assertAlmostEqual(lon, inv_lon, places=4)
                checked_records += 1

        self.assertGreater(checked_records, 0, "No records checked")

    def test_04_valid_scenario_trajectory_files(self):
        """4. Verify all 9 scenarios exist and have valid uncompressed and gzip files."""
        scenarios = ["baseline", "ml_forecast", "naive_persistence"]
        periods = ["morning_rush", "evening_rush", "off_peak"]

        for sc in scenarios:
            for p in periods:
                json_path = self.trajectories_dir / f"trajectories_{sc}_{p}.json"
                gz_path = self.trajectories_dir / f"trajectories_{sc}_{p}.json.gz"
                self.assertTrue(json_path.exists(), f"Missing trajectory JSON: {json_path}")
                self.assertTrue(gz_path.exists(), f"Missing trajectory GZ: {gz_path}")

                # Ensure non-trivial file size (>100KB)
                self.assertGreater(json_path.stat().st_size, 100_000)
                self.assertGreater(gz_path.stat().st_size, 50_000)

    def test_05_time_frame_ordering(self):
        """5. Verify time frames in trajectory files are strictly chronological."""
        self.assertIsNotNone(self.sample_data)
        summary = self.sample_data.get("summary_by_step", [])
        self.assertGreaterEqual(len(summary), 3500)

        # Ensure monotonically increasing timestamps
        times = [row["time"] for row in summary]
        self.assertEqual(times, sorted(times), "Simulation timestamps must be strictly chronological")
        self.assertIn(times[0], [0, 1])
        self.assertGreaterEqual(times[-1], 3590)

    def test_06_vehicle_type_mapping(self):
        """6. Verify all vehicle types belong to genuine categories: car, auto, bus."""
        self.assertIsNotNone(self.sample_data)
        frames = self.sample_data.get("frames", {})
        found_types = set()

        for step_key, veh_list in frames.items():
            for v in veh_list:
                found_types.add(v["vehicle_type"])

        valid_types = {"car", "auto", "bus"}
        self.assertTrue(found_types.issubset(valid_types), f"Invalid vehicle types found: {found_types}")
        # Must contain all 3 calibrated types
        self.assertEqual(found_types, valid_types, "All 3 vehicle types (car, auto, bus) must be simulated")

    def test_07_playback_data_integrity(self):
        """7. Verify consistency between frames, active vehicle counts, and step summaries."""
        self.assertIsNotNone(self.sample_data)
        frames = self.sample_data.get("frames", {})
        summary = self.sample_data.get("summary_by_step", [])

        # Step-by-step consistency check
        for row in summary[::250]:
            t_str = str(row["time"])
            if t_str in frames:
                self.assertEqual(row["active_vehicles"], len(frames[t_str]))

    def test_08_sumo_execution_mode_verification(self):
        """8. Verify metadata records execution_mode = 'SUMO' and genuine TraCI telemetry."""
        self.assertIsNotNone(self.sample_data)
        meta = self.sample_data.get("metadata", {})
        self.assertEqual(meta.get("execution_mode"), "SUMO")
        self.assertIn("SUMO", meta.get("engine", ""))
        self.assertIn("TraCI", meta.get("telemetry_source", ""))

    def test_09_flask_simulation_endpoints(self):
        """9. Verify all Flask simulation routes return HTTP 200 with valid schema."""
        # A. HTML page
        res_page = self.client.get("/simulation")
        self.assertEqual(res_page.status_code, 200)
        self.assertIn(b"Barapullah Elevated Corridor Microsimulation", res_page.data)

        # B. Manifest
        res_manifest = self.client.get("/api/simulation/manifest")
        self.assertEqual(res_manifest.status_code, 200)
        manifest_json = json.loads(res_manifest.data)
        self.assertIn("scenarios", manifest_json)

        # C. Network GeoJSON
        res_net = self.client.get("/api/simulation/network")
        self.assertEqual(res_net.status_code, 200)
        net_json = json.loads(res_net.data)
        self.assertEqual(len(net_json["features"]), 218)

        # D. Trajectory query
        res_traj = self.client.get("/api/simulation/trajectories?scenario=ml_forecast&period=morning_rush")
        self.assertEqual(res_traj.status_code, 200)
        traj_json = json.loads(res_traj.data)
        self.assertIn("frames", traj_json)
        self.assertIn("metadata", traj_json)


if __name__ == "__main__":
    unittest.main()
