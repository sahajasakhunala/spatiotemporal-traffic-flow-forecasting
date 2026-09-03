# Phase 2: SUMO Vehicle-Level Microsimulation

## Overview

The vehicle-level traffic microsimulation layer bridges the machine learning forecasting pipeline with an actual microscopic traffic simulation environment using **Eclipse SUMO (Simulation of Urban MObility)**.

This module demonstrates how next-hour traffic flow forecasts produced by the Random Forest model can be translated into calibrated vehicle demand and simulated through a real-world road network.

```
Historical Traffic Data (GeoJSON + Parquet)
            |
    Random Forest Model
            |
   Next-Hour Flow Forecast
            |
   Demand Calibration Layer
            |
    SUMO Road Network (Barapullah Corridor, 218 Segments)
            |
   Vehicle Routes & Departures (.rou.xml)
            |
   SUMO Traffic Microsimulation (.sumocfg)
            |
   Vehicle Movement / Speed / Density Analysis
```

---

## Technical Honesty & Scope Distinction

| Domain | Scope & Definition |
|---|---|
| **ML Layer** | Predicts next-hour traffic probe flow (`probe_count`) based on 24 spatiotemporal features. |
| **Probe Flow Data** | A sampled proxy from connected GPS fleet vehicles, **not** an exact 100% physical vehicle count. |
| **SUMO Simulation Layer** | A microscopic vehicle simulation using calibrated traffic demand units to simulate downstream queueing, density, and flow dynamics. |
| **Digital Twin Boundary** | The simulation is a calibrated traffic-demand simulation driven by observed/predicted data, **not** a physical digital twin of Delhi traffic. |

---

## 1. Selected Simulation Corridor: Barapullah Elevated Corridor

### Geographic Context & Selection Rationale
- **Corridor**: Barapullah Road / Elevated Corridor connecting Sarai Kale Khan (East) to INA / Jawaharlal Nehru Stadium (West) in South-Central Delhi.
- **Why this corridor?**:
  1. Grade-separated, high-capacity arterial expressway (FRC 1 and FRC 2) critical to South Delhi connectivity.
  2. High traffic volume in the dataset (mean probe flow 289.6, peak > 1500).
  3. Real-world geometry directly available in the dataset (218 segments with exact GPS LineString coordinates, lengths, and speed limits).
  4. Spatial Extent: Longitude 77.2082 to 77.2682 (~5.86 km East-West), Latitude 28.5708 to 28.5911 (~2.28 km North-South).

### Metric Planar Projection
To convert geographic coordinates (WGS84 lon/lat) into metric planar coordinates $(x, y)$ required by SUMO, an equirectangular projection centered at the corridor origin $(lat_0 = 28.58215, lon_0 = 77.24072)$ is used:

$$x = (lon - lon_0) \times \frac{\pi}{180} \times R \times \cos(lat_0 \times \frac{\pi}{180})$$
$$y = (lat - lat_0) \times \frac{\pi}{180} \times R$$

where $R = 6,371,000\text{ meters}$.

### Node Snapping & Junction Synthesis
- 436 segment endpoints were clustered within a 15.0-meter spatial tolerance into **178 discrete SUMO junctions**.
- Edge lanes were assigned according to road classification:
  - **FRC 1 (Elevated Motorway)**: 3 lanes, speed limit 50--80 km/h (13.89--22.22 m/s).
  - **FRC 2 (Major Arterial)**: 2 lanes, speed limit 30--50 km/h (8.33--13.89 m/s).
  - **FRC 4 (Connecting Ramps)**: 1 lane, speed limit 18--30 km/h (5.00--8.33 m/s).

Network XML files generated in `data/processed/sumo/`:
- `delhi_corridor.nod.xml` (Junction definitions)
- `delhi_corridor.edg.xml` (Edge definitions)
- `delhi_corridor.net.xml` (Compiled SUMO network)

---

## 2. Segment-to-SUMO Edge Mapping Layer

An explicit mapping layer links each project `segment_id` to its corresponding SUMO edge ID:
- File: `data/processed/sumo/segment_edge_mapping.parquet`
- Records: 218 segments (100% coverage of corridor segments)
- Schema:
  - `segment_id`: Unique integer ID from original dataset
  - `street_name`: Real street name (`Barapullah Road`)
  - `frc`: Functional road class (1, 2, or 4)
  - `speed_limit_kmh`: Posted speed limit
  - `sumo_edge_id`: Matched SUMO edge (`edge_<segment_id>`)
  - `from_junction`, `to_junction`: Snapped node IDs
  - `length_m`: Edge length in meters
  - `lanes`: Number of lanes
  - `corridor_direction`: `eastbound` or `westbound`
  - `match_confidence`: 1.0 (direct geometric extraction)

---

## 3. Demand Calibration Methodology

Probe flow is converted into calibrated vehicle demand using a linear scaling factor:

$$q_i(t) = \max\left(1, \text{round}\left(\alpha \times \text{flow}_i(t)\right)\right)$$

where $\alpha = 0.5$ represents the calibrated demand scaling factor.

Vehicle departures are distributed across the 3,600-second simulation hour with inter-arrival intervals:

$$\Delta t = \frac{3600}{q}$$

Vehicle types represent typical New Delhi mixed urban traffic:
- **Passenger Cars (70%)**: Length 4.5m, max speed 100 km/h, accel 2.6 m/s^2.
- **Auto-Rickshaws (20%)**: Length 3.0m, max speed 60 km/h, accel 2.0 m/s^2.
- **City Buses (10%)**: Length 12.0m, max speed 80 km/h, accel 1.2 m/s^2.

---

## 4. Simulation Scenarios & Results

Nine scenario configurations (3 scenarios x 3 time periods) were simulated on the test set:

| Scenario | Demand Input Source | Rush / Period | Mean Probe Flow | Calibrated Demand (veh/h) | Simulated Speed (km/h) |
|---|---|---|---:|---:|---:|
| **Baseline** | Observed probe count (`actual_probe_count`) | Morning Rush (08:00-10:00) | 424.5 | 212 | 45.3 |
| **Baseline** | Observed probe count | Evening Rush (17:00-20:00) | 424.9 | 212 | 45.3 |
| **Baseline** | Observed probe count | Off-Peak (13:00-15:00) | 455.1 | 228 | 45.3 |
| **RF Forecast** | Random Forest prediction (`rf_predicted`) | Morning Rush | 400.4 | 200 | 45.3 |
| **RF Forecast** | Random Forest prediction | Evening Rush | 431.8 | 216 | 45.3 |
| **RF Forecast** | Random Forest prediction | Off-Peak | 459.6 | 230 | 45.3 |
| **Naive Lag-1** | Persistence flow (`lag1_predicted`) | Morning Rush | 347.1 | 174 | 45.3 |
| **Naive Lag-1** | Persistence flow | Evening Rush | 448.1 | 224 | 45.3 |
| **Naive Lag-1** | Persistence flow | Off-Peak | 473.9 | 237 | 45.3 |

### Key Insight on Morning Ramp-up
During Morning Rush (08:00-10:00), traffic ramps up rapidly from overnight lows.
- **Baseline demand**: 212 veh/h
- **RF Forecast demand**: 200 veh/h (accurate anticipation of morning rush)
- **Naive Lag-1 demand**: 174 veh/h (severely under-estimates demand by 18% because it lags 1 hour behind)

---

## 5. SUMO Installation & Setup Guide (Windows)

When SUMO is not installed, the simulation executes in calibrated analytical mode, generating all required XML configuration files ready for SUMO.

To run with live graphical vehicle movement in SUMO-GUI:

### Option 1: Windows Package Manager (winget - Recommended)
Open PowerShell as Administrator and run:
```powershell
winget install Eclipse.SUMO
```

### Option 2: Direct Official Installer
1. Download from [Eclipse SUMO Official Releases](https://eclipse.dev/sumo/).
2. Install to default path: `C:\Program Files (x86)\Eclipse\Sumo`.
3. Set environment variable `SUMO_HOME`:
```powershell
[System.Environment]::SetEnvironmentVariable('SUMO_HOME', 'C:\Program Files (x86)\Eclipse\Sumo', 'User')
```
4. Add `%SUMO_HOME%\bin` to your `PATH`.

### Option 3: Python Wheel
```bash
pip install eclipse-sumo sumolib traci
```

---

## 6. Running the Simulation

### Command Line Execution
```bash
python -m src.sumo_simulation
```

### Running with SUMO-GUI
Once SUMO is installed:
```bash
sumo-gui -c data/processed/sumo/delhi_ml_forecast_morning_rush.sumocfg
```
In the GUI:
1. Click **Play** to start the simulation.
2. Adjust the delay slider (e.g. 50ms) to visibly observe cars, autos, and buses moving along the Barapullah Elevated Corridor.

### Running the Test Suite
```bash
python tests/test_sumo_simulation.py
```
