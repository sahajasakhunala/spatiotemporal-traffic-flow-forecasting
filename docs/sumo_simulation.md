# Phase 2: SUMO Vehicle-Level Microsimulation

> [!IMPORTANT]
> **Execution Status**: SUMO-Ready Pipeline with Analytical Fallback  
> **Host Environment**: SUMO is currently **not installed** on the host machine.  
> The road network (`.net.xml`), route definitions (`.rou.xml`), configuration files (`.sumocfg`), and segment-to-edge mapping are fully generated, validated, and ready for execution.  
> **Simulation Metrics Note**: The numerical speed (45.3 km/h), density (3.8--5.2 veh/km), and travel time (15.9 s) values in the tables and figures below are produced by the **calibrated analytical fallback model (BPR delay curve)**, **not** by genuine SUMO microsimulation. When SUMO is installed, the pipeline automatically executes `sumo` and replaces these values with genuine vehicle trajectory outputs from `tripinfo.xml`.

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
   ├── [SUMO Installed]        --> Live Vehicle Microsimulation (tripinfo.xml)
   └── [SUMO Not Installed]    --> Calibrated Analytical Fallback (BPR Delay Curve)
```

---

## Technical Scope & Layer Distinction

| Component | Nature of Output | Source of Truth |
|---|---|---|
| **ML Forecasting Layer** | Next-hour traffic probe flow (`probe_count`) | Validated Random Forest model on 2.39M test rows |
| **Probe Flow Data** | Proxy measurement from connected GPS fleets | Observed telematics data (not physical vehicle census) |
| **SUMO Network Infrastructure** | Real-world road geometry (.net.xml) | 218 Barapullah segments extracted from GeoJSON |
| **Segment Mapping** | Direct ID linking (`segment_edge_mapping.parquet`) | Spatial snapping and coordinate projection |
| **Vehicle Demand** | Calibrated departure schedule (.rou.xml) | Scaled from ML forecasts / observed flow ($q = 0.5 \times \text{flow}$) |
| **Current Metrics Mode** | Density, speed, traverse time | **Analytical Fallback (BPR delay curve)** |
| **Target Verified Mode** | Vehicle-level trajectories, collision, lane changes | **SUMO Microsimulation (requires local SUMO installation)** |

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

Nine scenario configurations (3 scenarios $\times$ 3 time periods) were evaluated on the test set.

> [!NOTE]
> Values for simulated speed, travel time, and density in this table are generated by the **analytical fallback model (BPR delay curve)** because SUMO is not currently installed on the host.

| Scenario | Demand Input Source | Rush / Period | Mean Probe Flow | Calibrated Demand (veh/h) | Fallback Speed (km/h) | Fallback Density (veh/km) |
|---|---|---|---:|---:|---:|---:|
| **Baseline** | Observed probe count (`actual_probe_count`) | Morning Rush (08:00-10:00) | 424.5 | **212** | 45.3 | 4.7 |
| **Baseline** | Observed probe count | Evening Rush (17:00-20:00) | 424.9 | **212** | 45.3 | 4.7 |
| **Baseline** | Observed probe count | Off-Peak (13:00-15:00) | 455.1 | **228** | 45.3 | 5.0 |
| **RF Forecast** | Random Forest prediction (`rf_predicted`) | Morning Rush | 400.4 | **200** | 45.3 | 4.4 |
| **RF Forecast** | Random Forest prediction | Evening Rush | 431.8 | **216** | 45.3 | 4.8 |
| **RF Forecast** | Random Forest prediction | Off-Peak | 459.6 | **230** | 45.3 | 5.1 |
| **Naive Lag-1** | Persistence flow (`lag1_predicted`) | Morning Rush | 347.1 | **174** | 45.3 | 3.8 |
| **Naive Lag-1** | Persistence flow | Evening Rush | 448.1 | **224** | 45.3 | 4.9 |
| **Naive Lag-1** | Persistence flow | Off-Peak | 473.9 | **237** | 45.3 | 5.2 |

### Demand Comparison in Morning Rush
During Morning Rush (08:00-10:00), traffic ramps up rapidly from overnight lows:
- **Baseline (Observed)**: 212 veh/h
- **Random Forest Forecast**: 200 veh/h (produces a closer demand estimate to observed baseline with a 5.7% delta)
- **Naive Lag-1 Persistence**: 174 veh/h (severely underestimates demand by 17.9% because it lags one hour behind the morning ramp-up)

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
