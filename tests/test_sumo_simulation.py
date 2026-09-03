"""
Test Suite for Phase 2: SUMO Vehicle-Level Microsimulation.

Verifies:
  1. SUMO environment detection & Windows setup instructions.
  2. Geographic coordinate projection (WGS84 -> Planar Cartesian).
  3. Real corridor geometry extraction from GeoJSON.
  4. SUMO road network XML generation (.net.xml, .nod.xml, .edg.xml).
  5. Segment-to-edge mapping persistence & coverage.
  6. Demand calibration layer (probe flow -> vehicles/hour).
  7. Scenario route files & simulation configuration generation.
  8. Graceful execution & analytical dynamics fallback when SUMO is absent.
"""

import math
import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, BASE_DIR
from src.sumo_simulation import (
    SEGMENT_EDGE_MAPPING_FILE,
    SUMO_PROCESSED_DIR,
    SUMO_RESULTS_FILE,
    build_sumo_network,
    calibrate_demand,
    check_sumo_installation,
    create_segment_edge_mapping,
    extract_corridor_features,
    generate_scenario_routes,
    generate_sumo_config,
    project_wgs84_to_cartesian,
    run_sumo_simulation,
)


class TestSumoSimulation(unittest.TestCase):
    """Unit and integration tests for SUMO simulation pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and cached network data."""
        raw_geojson_files = list((RAW_DATA_DIR / "probe_counts" / "geojson").glob("*.geojson"))
        cls.geojson_file = raw_geojson_files[0] if raw_geojson_files else None
        cls.predictions_file = PROCESSED_DATA_DIR / "simulation" / "detailed_predictions.parquet"

    def test_01_sumo_installation_detection(self):
        """1. Verify SUMO detection returns structured status and setup guide."""
        status = check_sumo_installation()
        self.assertIsInstance(status, dict)
        self.assertIn("installed", status)
        self.assertIn("binaries", status)
        self.assertIn("install_guide", status)
        self.assertIn("winget install Eclipse.SUMO", status["install_guide"])
        self.assertIn("pip install eclipse-sumo", status["install_guide"])

    def test_02_coordinate_projection(self):
        """2. Verify WGS84 GPS to Cartesian projection preserves origin and metric scale."""
        origin_lon = 77.24072
        origin_lat = 28.58215

        # Origin should project to (0.0, 0.0)
        x0, y0 = project_wgs84_to_cartesian(origin_lon, origin_lat, origin_lon, origin_lat)
        self.assertAlmostEqual(x0, 0.0, places=1)
        self.assertAlmostEqual(y0, 0.0, places=1)

        # 0.01 deg east (~975m at 28.58N)
        xe, ye = project_wgs84_to_cartesian(origin_lon + 0.01, origin_lat, origin_lon, origin_lat)
        self.assertGreater(xe, 800.0)
        self.assertLess(xe, 1100.0)
        self.assertAlmostEqual(ye, 0.0, places=1)

        # 0.01 deg north (~1112m)
        xn, yn = project_wgs84_to_cartesian(origin_lon, origin_lat + 0.01, origin_lon, origin_lat)
        self.assertAlmostEqual(xn, 0.0, places=1)
        self.assertGreater(yn, 1000.0)
        self.assertLess(yn, 1250.0)

    def test_03_corridor_feature_extraction(self):
        """3. Verify real Barapullah corridor features extracted from GeoJSON."""
        self.assertIsNotNone(self.geojson_file, "Raw GeoJSON file not found")
        features = extract_corridor_features(self.geojson_file)
        self.assertEqual(len(features), 218, "Expected exactly 218 Barapullah segments")

        for f in features:
            self.assertIn("segment_id", f)
            self.assertIn("raw_coords", f)
            self.assertGreater(len(f["raw_coords"]), 1, "LineString must have at least 2 points")
            self.assertIn(f["frc"], [1, 2, 4], "Barapullah FRCs should be 1, 2, or 4")
            self.assertGreater(f["speed_limit"], 0)
            self.assertGreater(f["distance"], 0)

    def test_04_network_generation_and_xml_validity(self):
        """4. Verify SUMO network generation produces valid XML nodes, edges, and .net.xml."""
        features = extract_corridor_features(self.geojson_file)
        net_info = build_sumo_network(features, SUMO_PROCESSED_DIR)

        self.assertEqual(net_info["n_edges"], 218)
        self.assertGreater(net_info["n_junctions"], 100)

        # Validate node XML parsing
        tree_nod = ET.parse(net_info["nod_file"])
        root_nod = tree_nod.getroot()
        self.assertEqual(root_nod.tag, "nodes")
        self.assertGreater(len(root_nod.findall("node")), 100)

        # Validate edge XML parsing
        tree_edg = ET.parse(net_info["edg_file"])
        root_edg = tree_edg.getroot()
        self.assertEqual(root_edg.tag, "edges")
        self.assertEqual(len(root_edg.findall("edge")), 218)

        # Validate compiled net XML parsing
        tree_net = ET.parse(net_info["net_file"])
        root_net = tree_net.getroot()
        self.assertEqual(root_net.tag, "net")
        self.assertIsNotNone(root_net.find("location"))
        self.assertEqual(len(root_net.findall("edge")), 218)

    def test_05_segment_edge_mapping_layer(self):
        """5. Verify segment-to-edge mapping Parquet table coverage and schema."""
        self.assertTrue(SEGMENT_EDGE_MAPPING_FILE.exists(), "Mapping file missing")
        df_map = pd.read_parquet(SEGMENT_EDGE_MAPPING_FILE)

        self.assertEqual(len(df_map), 218, "All 218 segments must be mapped")
        expected_cols = [
            "segment_id", "street_name", "frc", "speed_limit_kmh",
            "speed_limit_ms", "sumo_edge_id", "from_junction",
            "to_junction", "length_m", "lanes", "corridor_direction",
            "match_confidence",
        ]
        for col in expected_cols:
            self.assertIn(col, df_map.columns)

        # Match confidence should be 1.0
        self.assertTrue((df_map["match_confidence"] == 1.0).all())
        # All segment IDs must be unique
        self.assertEqual(df_map["segment_id"].nunique(), 218)

    def test_06_demand_calibration(self):
        """6. Verify demand calibration scaling and non-negativity."""
        # Non-negative output
        self.assertEqual(calibrate_demand(0.0), 1)
        self.assertEqual(calibrate_demand(-10.0), 1)

        # Scaling logic
        self.assertEqual(calibrate_demand(100.0, scale_factor=0.5), 50)
        self.assertEqual(calibrate_demand(400.0, scale_factor=0.5), 200)
        self.assertEqual(calibrate_demand(500.0, scale_factor=1.0), 500)

        # Monotonicity
        self.assertLess(calibrate_demand(100.0), calibrate_demand(300.0))

    def test_07_scenario_route_and_sumocfg_generation(self):
        """7. Verify route XML and .sumocfg generation for simulation scenarios."""
        self.assertTrue(self.predictions_file.exists(), "Detailed predictions missing")
        preds_df = pd.read_parquet(self.predictions_file)

        features = extract_corridor_features(self.geojson_file)
        net_info = build_sumo_network(features, SUMO_PROCESSED_DIR)

        test_rou = SUMO_PROCESSED_DIR / "test_sample_route.rou.xml"
        sc_info = generate_scenario_routes(
            net_info["edges"],
            preds_df,
            scenario_name="ml_forecast",
            period_name="morning_rush",
            output_file=test_rou,
        )

        self.assertGreater(sc_info["calibrated_vehicles_per_hour"], 0)
        self.assertGreater(sc_info["total_vehicles_generated"], 0)

        # Validate .rou.xml XML structure
        tree_rou = ET.parse(test_rou)
        root_rou = tree_rou.getroot()
        self.assertEqual(root_rou.tag, "routes")
        self.assertGreaterEqual(len(root_rou.findall("vType")), 3)
        self.assertGreaterEqual(len(root_rou.findall("route")), 2)
        self.assertGreaterEqual(len(root_rou.findall("vehicle")), 1)

        # Validate .sumocfg XML structure
        test_cfg = SUMO_PROCESSED_DIR / "test_sample.sumocfg"
        generate_sumo_config(net_info["net_file"], test_rou, test_cfg)
        tree_cfg = ET.parse(test_cfg)
        root_cfg = tree_cfg.getroot()
        self.assertEqual(root_cfg.tag, "configuration")
        self.assertIsNotNone(root_cfg.find("input/net-file"))
        self.assertIsNotNone(root_cfg.find("input/route-files"))

    def test_08_graceful_simulation_runner(self):
        """8. Verify simulation runner handles missing binary gracefully with explicit execution mode."""
        features = extract_corridor_features(self.geojson_file)
        net_info = build_sumo_network(features, SUMO_PROCESSED_DIR)
        sumo_status = check_sumo_installation()

        sc_info = {
            "scenario": "test_scenario",
            "period": "morning_rush",
            "raw_mean_probe_flow": 400.0,
            "calibrated_vehicles_per_hour": 200,
        }
        cfg_file = SUMO_PROCESSED_DIR / "delhi_baseline_morning_rush.sumocfg"

        result = run_sumo_simulation(sumo_status, cfg_file, sc_info, net_info)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertIn("execution_mode", result)
        self.assertIn(result["execution_mode"], ["SUMO", "ANALYTICAL_FALLBACK"])
        self.assertIn("results_source", result)
        if not sumo_status["installed"]:
            self.assertEqual(result["execution_mode"], "ANALYTICAL_FALLBACK")
            self.assertIn("Analytical Fallback", result["results_source"])
        self.assertIn("mean_simulated_speed_kmh", result)
        self.assertIn("mean_density_veh_per_km", result)
        self.assertIn("mean_travel_time_sec", result)
        self.assertGreater(result["mean_simulated_speed_kmh"], 0.0)


if __name__ == "__main__":
    unittest.main()
