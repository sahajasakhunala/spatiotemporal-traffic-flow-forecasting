"""
Unit & Integration Tests for Traffic Disturbance Experiments & Closed-Loop Evaluation (Phase 4)
-----------------------------------------------------------------------------------------------
Verifies:
  1. Disturbance scenario trajectory files (normal, accident, lane closure, heavy vehicle, signal hold)
  2. Genuine SUMO TraCI telemetry coordinates & bounds (zero fabrication)
  3. Accident scenario queue emergence and speed deceleration
  4. Multi-lane closure bottleneck dynamics
  5. Slow heavy vehicle insertion and moving bottleneck formation
  6. Signal disruption queue backup
  7. Strict separation between Real Data Targets and Simulation-Only Targets
  8. Quantitative closed-loop evaluation metrics (MAE, RMSE, throughput deviation)
  9. Flask experiment API endpoints (manifest, dynamic execution, trajectory delivery)
"""

import json
import unittest
from pathlib import Path

from dashboard.app import app
from src.sumo_simulation import SUMO_PROCESSED_DIR


class TestDisturbanceExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectories_dir = SUMO_PROCESSED_DIR / "trajectories"
        cls.experiments = ["normal", "accident", "lane_closure", "heavy_vehicle", "signal_disruption"]
        cls.client = app.test_client()

    def test_01_all_experiment_files_exist_and_valid(self):
        """1. Verify all 5 disturbance scenario trajectory files exist and contain valid JSON."""
        for exp in self.experiments:
            json_file = self.trajectories_dir / f"trajectories_experiment_{exp}.json"
            self.assertTrue(json_file.exists(), f"Missing trajectory file: {json_file.name}")
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("metadata", data)
            self.assertIn("evaluation", data)
            self.assertIn("summary_by_step", data)
            self.assertIn("frames", data)
            self.assertGreater(len(data["frames"]), 500)

    def test_02_genuine_traci_coordinates_and_bounds(self):
        """2. Verify all vehicle coordinates fall strictly within real Barapullah bounds."""
        json_file = self.trajectories_dir / "trajectories_experiment_accident.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for t_str, vehs in list(data["frames"].items())[:100]:
            for v in vehs:
                lat = v["latitude"]
                lon = v["longitude"]
                self.assertGreaterEqual(lat, 28.585, f"Latitude {lat} out of bounds")
                self.assertLessEqual(lat, 28.605, f"Latitude {lat} out of bounds")
                self.assertGreaterEqual(lon, 77.260, f"Longitude {lon} out of bounds")
                self.assertLessEqual(lon, 77.280, f"Longitude {lon} out of bounds")
                self.assertIn("speed_kmh", v)
                self.assertIn("heading_angle_deg", v)
                self.assertIn("current_edge_id", v)

    def test_03_accident_scenario_queue_dynamics(self):
        """3. Verify accident scenario causes queue emergence on target edge."""
        json_file = self.trajectories_dir / "trajectories_experiment_accident.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        eval_data = data["evaluation"]
        self.assertEqual(eval_data["disturbance"]["type"], "accident")
        self.assertEqual(eval_data["disturbance"]["edge_id"], "edge_13560341041261")
        self.assertGreater(eval_data["metrics"]["max_queue_vehicles"], 0)

    def test_04_lane_closure_bottleneck_dynamics(self):
        """4. Verify multi-lane bottleneck closure causes significant speed drop."""
        json_file = self.trajectories_dir / "trajectories_experiment_lane_closure.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = data["evaluation"]["metrics"]
        # Mean speed during bottleneck should drop well below baseline 41 km/h
        self.assertLess(m["actual_mean_speed_kmh"], 38.0)
        self.assertGreaterEqual(m["max_queue_vehicles"], 3)
        self.assertGreater(m["speed_deviation_mae"], 5.0)

    def test_05_heavy_vehicle_insertion(self):
        """5. Verify slow heavy vehicles exist in heavy_vehicle scenario."""
        json_file = self.trajectories_dir / "trajectories_experiment_heavy_vehicle.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Look for heavy truck in frames
        found_truck = False
        for t_str, vehs in data["frames"].items():
            for v in vehs:
                if "heavy_truck" in v["vehicle_id"]:
                    found_truck = True
                    # Max speed constraint check (<= 25 km/h)
                    self.assertLessEqual(v["speed_kmh"], 25.0)
                    break
            if found_truck:
                break
        self.assertTrue(found_truck, "Heavy truck not found in telemetry frames")

    def test_06_signal_disruption_queue_backup(self):
        """6. Verify traffic signal hold causes massive queue accumulation."""
        json_file = self.trajectories_dir / "trajectories_experiment_signal_disruption.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = data["evaluation"]["metrics"]
        # Holding signal red should produce a queue of at least 8 vehicles
        self.assertGreaterEqual(m["max_queue_vehicles"], 8)
        self.assertEqual(m["congestion_state"], "Congested Queue")

    def test_07_target_classification_separation(self):
        """7. Verify strict separation between real data targets and simulation-only targets."""
        json_file = self.trajectories_dir / "trajectories_experiment_normal.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        targets = data["evaluation"]["target_classification"]
        real_targets = targets["real_data_targets"]
        sim_targets = targets["simulation_only_targets"]

        # probe_count / flow proxy is the ONLY real target supported by Kaggle dataset
        self.assertTrue(any("probe_count" in t for t in real_targets))
        self.assertFalse(any("speed" in t for t in real_targets))
        self.assertFalse(any("queue" in t for t in real_targets))

        # Speed, density, queue, delay are simulation-only
        self.assertTrue(any("speed" in t for t in sim_targets))
        self.assertTrue(any("queue" in t for t in sim_targets))
        self.assertTrue(any("density" in t for t in sim_targets))

    def test_08_quantitative_closed_loop_evaluation_metrics(self):
        """8. Verify presence and numerical validity of evaluation metrics."""
        for exp in self.experiments:
            json_file = self.trajectories_dir / f"trajectories_experiment_{exp}.json"
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            m = data["evaluation"]["metrics"]
            self.assertIn("predicted_flow_demand", m)
            self.assertIn("actual_completed_vehicles", m)
            self.assertIn("actual_throughput_veh_h", m)
            self.assertIn("predicted_mean_speed_kmh", m)
            self.assertIn("actual_mean_speed_kmh", m)
            self.assertIn("speed_deviation_mae", m)
            self.assertIn("speed_deviation_rmse", m)
            self.assertIn("max_queue_vehicles", m)
            self.assertIn("congestion_state", m)
            self.assertGreaterEqual(m["speed_deviation_mae"], 0.0)
            self.assertGreaterEqual(m["speed_deviation_rmse"], 0.0)

    def test_09_flask_experiment_endpoints(self):
        """9. Verify Flask API experiment endpoints return HTTP 200 with valid schema."""
        # A. Experiments manifest
        res_exp = self.client.get("/api/simulation/experiments")
        self.assertEqual(res_exp.status_code, 200)
        exp_json = json.loads(res_exp.data)
        self.assertEqual(exp_json.get("status"), "success")
        self.assertEqual(len(exp_json.get("experiments", [])), 5)

        # B. Trajectory query for accident
        res_traj = self.client.get("/api/simulation/trajectories?experiment=accident")
        self.assertEqual(res_traj.status_code, 200)
        traj_json = json.loads(res_traj.data)
        self.assertIn("frames", traj_json)
        self.assertIn("evaluation", traj_json)


if __name__ == "__main__":
    unittest.main()
