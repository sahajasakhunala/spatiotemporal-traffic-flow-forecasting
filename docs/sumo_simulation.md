# Phase 2: SUMO Vehicle-Level Microsimulation

> [!IMPORTANT]
> **Execution Status**: Genuine SUMO Microsimulation Verified (Eclipse SUMO 1.27.1)  
> **Host Environment**: SUMO is **installed and verified** at `C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe`.  
> **Active Execution Mode**: `SUMO` (with automatic fallback to `ANALYTICAL_FALLBACK` preserved for environments lacking SUMO).  
> **Output Attribution**: All simulation speed, density, and travel time metrics reported in this section are generated directly by **Eclipse SUMO vehicle microsimulation** and parsed from `tripinfo.xml`.

---

## System Architecture

The vehicle-level traffic microsimulation layer bridges the machine learning forecasting pipeline with an actual microscopic traffic simulation environment using **Eclipse SUMO (Simulation of Urban MObility)**.

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
   SUMO Configuration (.sumocfg)
            |
   Execution Mode Switch
   ├── [SUMO Installed]        --> Live Vehicle Microsimulation (tripinfo.xml) [ACTIVE]
   └── [SUMO Not Installed]    --> Calibrated Analytical Fallback (BPR Delay Curve) [PRESERVED]
```

---

## Technical Scope & Layer Distinction

| Component | Nature of Output | Source of Truth |
|---|---|---|
| **ML Forecasting Layer** | Next-hour traffic probe flow (`probe_count`) | Validated Random Forest model on 2.39M test rows ($R^2 = 0.9683$) |
| **Probe Flow Data** | Proxy measurement from connected GPS fleets | Observed telematics data (not physical vehicle census) |
| **SUMO Network Infrastructure** | Real-world road geometry (`.net.xml`) | 218 Barapullah segments compiled via `netconvert` |
| **Segment Mapping** | Direct ID linking (`segment_edge_mapping.parquet`) | 218/218 segments with 1.0 match confidence |
| **Vehicle Demand** | Calibrated departure schedule (`.rou.xml`) | Scaled from ML forecasts / observed flow ($q = 0.5 \times \text{flow}$) |
| **Execution Mode** | Vehicle-level trajectories, speeds, arrivals | **SUMO Microsimulation (`execution_mode: "SUMO"`)** |
| **Safety Net** | Macroscopic BPR delay curve | **Analytical Fallback (retained for portability)** |

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

## 4. Simulation Scenarios & Genuine SUMO 1.27.1 Results

Nine scenario configurations (3 demand scenarios $\times$ 3 canonical time periods) were simulated in **Eclipse SUMO 1.27.1** over the corridor network:

| Scenario | Demand Input Source | Rush / Period | Mean Probe Flow | Calibrated Demand (veh/h) | Vehicles Simulated | SUMO Mean Speed (km/h) | Corridor Density (veh/km) | Mean Traverse Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Baseline** | Observed probe count (`actual_probe_count`) | Morning Rush (08:00-10:00) | 424.5 | **212** | 205 | 40.6 | 5.2 | 137.1 |
| **Baseline** | Observed probe count | Evening Rush (17:00-20:00) | 424.9 | **212** | 205 | 40.6 | 5.2 | 137.1 |
| **Baseline** | Observed probe count | Off-Peak (13:00-15:00) | 455.1 | **228** | 219 | 40.6 | 5.6 | 137.0 |
| **RF Forecast** | Random Forest prediction (`rf_predicted`) | Morning Rush | 400.4 | **200** | 193 | 40.7 | 4.9 | 136.7 |
| **RF Forecast** | Random Forest prediction | Evening Rush | 431.8 | **216** | 208 | 40.6 | 5.3 | 137.0 |
| **RF Forecast** | Random Forest prediction | Off-Peak | 459.6 | **230** | 222 | 40.6 | 5.7 | 136.9 |
| **Naive Lag-1** | Persistence flow (`lag1_predicted`) | Morning Rush | 347.1 | **174** | 168 | 40.8 | 4.3 | 136.4 |
| **Naive Lag-1** | Persistence flow | Evening Rush | 448.1 | **224** | 215 | 40.7 | 5.5 | 136.9 |
| **Naive Lag-1** | Persistence flow | Off-Peak | 473.9 | **237** | 229 | 40.7 | 5.8 | 136.9 |

---

### Comparative Analysis: Genuine SUMO vs Analytical Fallback

| Dimension | Analytical Fallback Model | Genuine SUMO 1.27.1 Microsimulation | Technical Distinction |
|---|---|---|---|
| **Underlying Dynamics** | Macroscopic BPR speed-flow curve | Microscopic Krauss car-following & lane-changing model | SUMO models vehicle-to-vehicle headways, acceleration, and turns |
| **Execution Engine** | Python numerical function | `sumo.exe` binary (C++ DLR core) | Direct process execution via `subprocess` |
| **Data Output Source** | Calculated scalar formulas | `tripinfo.xml` and `summary.xml` | Derived from individual vehicle telemetry entries |
| **Morning Rush Speed** | 45.3 km/h | 40.6 -- 40.8 km/h | SUMO captures turn friction, intersection slowdowns, and acceleration delays |
| **Corridor Traverse Time** | 15.9 s (isolated 200m edge) | 136.4 -- 137.1 s (full corridor route) | SUMO computes continuous end-to-end trip duration across 15 connected edges |
| **Simulated Density** | 3.8 -- 5.2 veh/km | 4.3 -- 5.8 veh/km | SUMO accounts for actual spatial queues formed at junctions |

### Demand Comparison in Morning Rush
During Morning Rush (08:00-10:00), traffic ramps up rapidly from overnight lows:
- **Baseline (Observed)**: 212 veh/h demand -> 205 vehicles completed in SUMO
- **Random Forest Forecast**: 200 veh/h demand -> 193 vehicles completed in SUMO (closer demand estimate to observed baseline with a 5.7% delta)
- **Naive Lag-1 Persistence**: 174 veh/h demand -> 168 vehicles completed in SUMO (underestimates demand by 17.9% because it lags one hour behind the morning ramp-up)

*Note: This comparison reflects the relative demand calibration derived from the ML forecasting models; it does not constitute independent proof of forecasting accuracy from the simulation itself.*

---

## 5. Automatic Execution Mode Switching

`src/sumo_simulation.py` incorporates automatic execution mode detection:
1. When `sumo` is available in PATH or `%SUMO_HOME%\bin`:
   - Executes: `sumo -c <config> --tripinfo-output <xml> --summary-output <xml>`
   - Parses: genuine trip durations, speeds, and vehicle completions from `tripinfo.xml`
   - Sets: `execution_mode = "SUMO"`, `results_source = "Genuine SUMO Microsimulation Output (tripinfo.xml)"`
2. When `sumo` is absent:
   - Executes: analytical BPR macroscopic delay curve
   - Sets: `execution_mode = "ANALYTICAL_FALLBACK"`, `results_source = "Analytical Fallback (BPR Delay Curve - SUMO Binary Not Installed)"`
   - Generates: all production-grade XML files ready for execution once SUMO is installed

---

## 6. Windows Installation Instructions for SUMO

To execute genuine microsimulation and observe visual vehicle movement in SUMO-GUI:

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

## 7. Running the Simulation

### Command Line Execution
```bash
python -m src.sumo_simulation
```

### Running with SUMO-GUI (Technical Reference View)
With SUMO installed on Windows:
```bash
"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" -c data/processed/sumo/delhi_ml_forecast_morning_rush.sumocfg -g data/processed/sumo/delhi.view.xml
```
In the GUI:
1. Click **Play** (or press Space) to start the simulation.
2. The pre-configured `delhi.view.xml` automatically activates realistic vehicle silhouettes, color-by-speed (green for high speed, red for queues), and 50ms smooth stepping.

### Running the Full Test Suite
```powershell
$env:PYTHONPATH = "C:\Users\LENOVO\Documents\antigravity\optimistic-einstein"; python -m unittest discover tests -v
```
*(All 24/24 tests pass: 9/9 TraCI Exporter & Web Visualizer, 8/8 SUMO Simulation, 7/7 Technical Audit).*

---

## 8. Phase 3: Interactive Web Microsimulation Visualizer

Phase 3 provides a modern, interactive web-based traffic visualizer built on top of genuine Eclipse SUMO 1.27.1 TraCI simulation data.

```
Random Forest Flow Forecast (Aug 27-30)
                 ↓
Calibrated Vehicle Demand (q = 0.5 * flow)
                 ↓
Eclipse SUMO 1.27.1 Microsimulation Engine
                 ↓
TraCI Telemetry Exporter (`src/sumo_exporter.py`)
(Captures 14 fields per vehicle step @ 1-second resolution)
                 ↓
Compressed Trajectory Stores (`data/processed/sumo/trajectories/`)
(3,600 frames per scenario, GZ: ~950 KB, ~27,000 vehicle states)
                 ↓
Flask Web Dashboard (`dashboard/app.py` -> `/simulation`)
                 ↓
Interactive Leaflet 2.5D Real-Map Visualization
- Real Barapullah Road Geometry (218 Segments)
- Vehicle Glyphs (Sedan Cars, Auto-Rickshaws, Transit Buses)
- Color-by-Speed (Green >35 km/h, Amber 15-35 km/h, Red <15 km/h)
- 60 FPS Smooth Playback (Play/Pause, Scrub, 1x/2x/5x/10x Multipliers)
- Clickable Vehicle Telemetry Inspector HUD
- Real-time Corridor KPIs (Speed, Density, Active & Completed Counts)
```

### Telemetry Exporter Schema (`src/sumo_exporter.py`)
Each frame captures 14 genuine telemetry parameters from TraCI without fabrication:
1. `simulation_time_sec`: Timestamp in the 3,600-second hour
2. `vehicle_id`: Unique vehicle identifier
3. `vehicle_type`: Calibrated vehicle classification (`car`, `auto`, `bus`)
4. `planar_x`, `planar_y`: Metric Cartesian coordinates on corridor
5. `latitude`, `longitude`: Exact WGS84 GPS coordinates via inverse projection
6. `speed_mps`: Instantaneous speed in meters per second
7. `speed_kmh`: Speed converted to km/h
8. `heading_angle_deg`: Vehicle orientation angle ($0^\circ$ North, $90^\circ$ East)
9. `current_edge_id`: Traversed SUMO edge
10. `current_lane_index`: Active lane index ($0$: curb, $1$: center, $2$: median)
11. `acceleration`: Vehicle acceleration/deceleration in $\text{m/s}^2$
12. `waiting_time_sec`: Queue waiting delay at bottlenecks

### How to Launch the Web Microsimulation
1. **Start the Flask Dashboard**:
```powershell
python dashboard/app.py
```
2. **Open the Simulation URL in your browser**:
```
http://127.0.0.1:5000/simulation
```
3. **Interact with the Simulation**:
- Click **Play** to observe genuine simulated cars, auto-rickshaws, and buses traversing the Barapullah Elevated Corridor.
- Drag the **Time Scrubber** to jump to any point in the 1-hour simulation.
- Click any moving vehicle to inspect its live speed, lane, acceleration, and GPS coordinates in the **Vehicle Inspector HUD**.
- Switch scenarios using the top dropdowns (e.g. compare **Random Forest Forecast** vs **Baseline Observed** vs **Naive Lag-1**) to visually observe differences in corridor vehicle density and bottleneck buildup.
