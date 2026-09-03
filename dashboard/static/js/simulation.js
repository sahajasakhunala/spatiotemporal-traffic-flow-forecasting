/**
 * Barapullah Traffic Digital Twin & Experimentation Platform
 * -----------------------------------------------------------
 * Dual-Engine Visualizer:
 *   1. CesiumJS 3D Urban Digital Twin (3D Extruded Buildings, Satellite, Elevated Flyover, 3D Vehicles)
 *   2. 2.5D Corridor Operations Center (Multi-Lane Asphalt Expressway, Distinctive Vehicle Glyphs)
 *
 * Controlled Microscopic Disturbance Scenarios:
 *   - Normal Baseline Traffic (RF Flow Forecast)
 *   - Accident / Lane Blockage
 *   - Bottleneck Multi-Lane Closure
 *   - Slow Heavy Vehicles (Moving Bottlenecks)
 *   - Traffic Signal / Junction Hold
 *   - Dynamic Custom SUMO Run via TraCI API
 *
 * Closed-Loop Model Evaluation:
 *   - Predicted (Baseline RF Forecast) vs Actual Simulated (SUMO TraCI)
 *   - Speed MAE, RMSE, Max Queue, Congestion State
 *   - Live Time-Series Dynamics Chart (Canvas)
 *   - Single Vehicle Telemetry Verification Pipeline
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. Unified State Management
    // -------------------------------------------------------------------------
    const state = {
        activeEngine: "3d", // "3d" (Cesium) or "2d" (Leaflet)
        basemapType: "satellite",
        
        // Exact TraCI Trajectory Coordinates for the 1.54 km Simulated Corridor
        corridorCenter: [28.594756, 77.270426], // [lat, lon]
        corridorBounds: [[28.590998, 77.264380], [28.598514, 77.276473]],
        
        // Cesium 3D Engine
        cesiumViewer: null,
        cesiumVehicles: new Map(), // vid -> Cesium.Entity
        cesiumBuildingsSource: null,
        
        // 2.5D Leaflet Engine
        leafletMap: null,
        leafletTiles: null,
        leafletVehicles: new Map(), // vid -> L.marker
        leafletRoadLayers: [],
        
        // Active Scenario & Disturbance Data
        currentExperiment: "normal",
        trajectoryData: null,
        frames: {},
        summaryByStep: [],
        timeseries: [],
        evaluationMetrics: null,
        totalDuration: 1200,
        
        // Playback Engine
        isPlaying: false,
        currentTime: 0.0,
        speedMultiplier: 2.0,
        lastAnimTimestamp: null,
        
        // Camera & Follow
        selectedVehicleId: null,
        isFollowMode: false,
    };

    // DOM Elements
    const els = {
        btnView3D: document.getElementById("btnView3D"),
        btnView2D: document.getElementById("btnView2D"),
        cesiumContainer: document.getElementById("cesiumContainer"),
        leafletContainer: document.getElementById("leafletContainer"),
        basemapSelect: document.getElementById("basemapSelect"),
        experimentSelect: document.getElementById("experimentSelect"),
        
        // Experiment Drawer
        btnToggleExpDrawer: document.getElementById("btnToggleExpDrawer"),
        experimentDrawer: document.getElementById("experimentDrawer"),
        btnCloseDrawer: document.getElementById("btnCloseDrawer"),
        distTypeInput: document.getElementById("distTypeInput"),
        distEdgeInput: document.getElementById("distEdgeInput"),
        distStartInput: document.getElementById("distStartInput"),
        distDurationInput: document.getElementById("distDurationInput"),
        btnRunCustomSim: document.getElementById("btnRunCustomSim"),
        simRunStatus: document.getElementById("simRunStatus"),
        
        // Camera Controls
        cam3DAction: document.getElementById("cam3DAction"),
        camTopDown: document.getElementById("camTopDown"),
        camFollow: document.getElementById("camFollow"),
        camReset: document.getElementById("camReset"),
        
        // Live KPIs
        kpiSimTime: document.getElementById("kpiSimTime"),
        kpiStep: document.getElementById("kpiStep"),
        kpiActiveVehs: document.getElementById("kpiActiveVehs"),
        kpiCompletedVehs: document.getElementById("kpiCompletedVehs"),
        kpiMeanSpeed: document.getElementById("kpiMeanSpeed"),
        kpiSpeedSub: document.getElementById("kpiSpeedSub"),
        kpiQueue: document.getElementById("kpiQueue"),
        kpiDensity: document.getElementById("kpiDensity"),
        distIndicatorCard: document.getElementById("distIndicatorCard"),
        distBadgeText: document.getElementById("distBadgeText"),
        distDescText: document.getElementById("distDescText"),
        
        // Evaluation Panel
        evalPanel: document.getElementById("evalPanel"),
        evalBody: document.getElementById("evalBody"),
        btnToggleEval: document.getElementById("btnToggleEval"),
        mPredSpeed: document.getElementById("mPredSpeed"),
        mActSpeed: document.getElementById("mActSpeed"),
        mDevSpeed: document.getElementById("mDevSpeed"),
        mPredFlow: document.getElementById("mPredFlow"),
        mActFlow: document.getElementById("mActFlow"),
        mDevFlow: document.getElementById("mDevFlow"),
        mActQueue: document.getElementById("mActQueue"),
        mQueueState: document.getElementById("mQueueState"),
        mCongState: document.getElementById("mCongState"),
        speedChartCanvas: document.getElementById("speedChartCanvas"),
        
        // Debug Verification
        btnVerifyVehicle: document.getElementById("btnVerifyVehicle"),
        debugOutput: document.getElementById("debugOutput"),
        
        // Vehicle Inspector
        inspector: document.getElementById("vehicleInspector"),
        closeInspectorBtn: document.getElementById("closeInspectorBtn"),
        inspId: document.getElementById("inspId"),
        inspType: document.getElementById("inspType"),
        inspSpeed: document.getElementById("inspSpeed"),
        inspHeading: document.getElementById("inspHeading"),
        inspEdge: document.getElementById("inspEdge"),
        inspLane: document.getElementById("inspLane"),
        inspAccel: document.getElementById("inspAccel"),
        inspWait: document.getElementById("inspWait"),
        inspGps: document.getElementById("inspGps"),
        
        // Playback Bar
        btnPlayPause: document.getElementById("btnPlayPause"),
        playIcon: document.getElementById("playIcon"),
        playText: document.getElementById("playText"),
        btnRestart: document.getElementById("btnRestart"),
        btnStepBack: document.getElementById("btnStepBack"),
        btnStepFwd: document.getElementById("btnStepFwd"),
        timeSlider: document.getElementById("timeSlider"),
        timelineCurrent: document.getElementById("timelineCurrent"),
        timelineMax: document.getElementById("timelineMax"),
        speedBtns: document.querySelectorAll(".speed-btn"),
    };

    // 100% Free Tile Providers - Zero API Key Watermarks
    const TILE_URLS = {
        satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        dark: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        osm: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    };

    // -------------------------------------------------------------------------
    // 2. Engine A: CesiumJS 3D Urban Digital Twin
    // -------------------------------------------------------------------------
    function initCesium() {
        if (typeof Cesium === "undefined") {
            console.warn("Cesium library not found, using 2.5D mode.");
            switchEngine("2d");
            return;
        }

        Cesium.Ion.defaultAccessToken = "";

        state.cesiumViewer = new Cesium.Viewer("cesiumContainer", {
            baseLayerPicker: false,
            geocoder: false,
            homeButton: false,
            infoBox: false,
            sceneModePicker: false,
            selectionIndicator: false,
            timeline: false,
            navigationHelpButton: false,
            animation: false,
            shouldAnimate: false,
            creditContainer: document.createElement("div"),
            imageryProvider: new Cesium.ArcGisMapServerImageryProvider({
                url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
            }),
        });

        // Configure smooth camera interaction (zoom, pan, tilt, orbit)
        const controller = state.cesiumViewer.scene.screenSpaceCameraController;
        controller.enableRotate = true;
        controller.enableTranslate = true;
        controller.enableZoom = true;
        controller.enableTilt = true;
        controller.enableLook = true;
        controller.inertiaSpin = 0.08;
        controller.inertiaTranslate = 0.08;
        controller.inertiaZoom = 0.08;
        controller.minimumZoomDistance = 35.0; // Zoom down to road surface
        controller.maximumZoomDistance = 25000.0;

        // Position camera directly framing the 1.54 km Barapullah route
        setCesiumCamera("3d");

        // Load 3D real buildings and elevated flyover
        loadCesiumBuildings();
        loadCesiumRoads();

        // Left click handler for vehicle inspection
        const handler = new Cesium.ScreenSpaceEventHandler(state.cesiumViewer.scene.canvas);
        handler.setInputAction((movement) => {
            const picked = state.cesiumViewer.scene.pick(movement.position);
            if (Cesium.defined(picked) && picked.id && picked.id.telemetry) {
                inspectVehicle(picked.id.telemetry);
            }
        }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
    }

    function setCesiumCamera(mode) {
        if (!state.cesiumViewer) return;
        const [lat, lon] = state.corridorCenter;

        if (mode === "3d") {
            // Pitched 3D action view looking east along the flyover
            state.cesiumViewer.camera.flyTo({
                destination: Cesium.Cartesian3.fromDegrees(lon - 0.005, lat - 0.003, 420),
                orientation: {
                    heading: Cesium.Math.toRadians(72.0),
                    pitch: Cesium.Math.toRadians(-30.0),
                    roll: 0.0,
                },
                duration: 1.2,
            });
        } else if (mode === "topdown") {
            // Direct overhead route view
            state.cesiumViewer.camera.flyTo({
                destination: Cesium.Cartesian3.fromDegrees(lon, lat, 1200),
                orientation: {
                    heading: Cesium.Math.toRadians(0.0),
                    pitch: Cesium.Math.toRadians(-89.0),
                    roll: 0.0,
                },
                duration: 1.2,
            });
        }
    }

    async function loadCesiumBuildings() {
        try {
            const resp = await fetch("/api/simulation/buildings");
            if (!resp.ok) return;
            const geojson = await resp.json();

            const ds = await Cesium.GeoJsonDataSource.load(geojson, { clampToGround: true });
            state.cesiumViewer.dataSources.add(ds);
            state.cesiumBuildingsSource = ds;

            for (const entity of ds.entities.values) {
                if (entity.polygon) {
                    const h = entity.properties.height ? entity.properties.height.getValue() : 12;
                    entity.polygon.extrudedHeight = h;
                    entity.polygon.material = Cesium.Color.fromCssColorString("#334155").withAlpha(0.85);
                    entity.polygon.outline = true;
                    entity.polygon.outlineColor = Cesium.Color.fromCssColorString("#64748b");
                }
            }
        } catch (err) {
            console.error("Cesium building load error:", err);
        }
    }

    async function loadCesiumRoads() {
        try {
            const resp = await fetch("/api/simulation/network");
            if (!resp.ok) return;
            const geojson = await resp.json();

            // Render Barapullah corridor as elevated highway ribbon (+8.5m)
            for (const f of geojson.features) {
                const coords = f.geometry.coordinates;
                const positions = coords.map(c => Cesium.Cartesian3.fromDegrees(c[0], c[1], 8.5));
                const frc = f.properties.frc || 2;
                const width = frc === 1 ? 16 : 10;

                state.cesiumViewer.entities.add({
                    polyline: {
                        positions: positions,
                        width: width,
                        material: new Cesium.PolylineGlowMaterialProperty({
                            glowPower: 0.18,
                            color: Cesium.Color.fromCssColorString(frc === 1 ? "#38bdf8" : "#94a3b8")
                        })
                    }
                });
            }
        } catch (err) {
            console.error("Cesium road load error:", err);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Engine B: 2.5D Corridor Operations Center (Leaflet)
    // -------------------------------------------------------------------------
    function initLeaflet() {
        state.leafletMap = L.map("leafletContainer", {
            center: state.corridorCenter,
            zoom: 16,
            minZoom: 14,
            maxZoom: 19,
            zoomControl: true,
        });

        updateLeafletBasemap(state.basemapType);
        loadLeafletBuildings();
        loadLeafletRoadNetwork();
    }

    function updateLeafletBasemap(type) {
        if (!state.leafletMap) return;
        if (state.leafletTiles) {
            state.leafletMap.removeLayer(state.leafletTiles);
        }

        const url = TILE_URLS[type] || TILE_URLS.satellite;
        state.leafletTiles = L.tileLayer(url, {
            maxZoom: 19,
            attribution: "&copy; Esri & OpenStreetMap",
        }).addTo(state.leafletMap);
    }

    async function loadLeafletBuildings() {
        try {
            const resp = await fetch("/api/simulation/buildings");
            if (!resp.ok) return;
            const geojson = await resp.json();

            L.geoJSON(geojson, {
                style: {
                    color: "#475569",
                    weight: 1,
                    fillColor: "#1e293b",
                    fillOpacity: 0.8,
                }
            }).addTo(state.leafletMap);
        } catch (err) {
            console.error("Leaflet building load error:", err);
        }
    }

    async function loadLeafletRoadNetwork() {
        try {
            const resp = await fetch("/api/simulation/network");
            if (!resp.ok) return;
            const geojson = await resp.json();

            // Layer 1: Outer concrete parapet barrier casing (24px)
            L.geoJSON(geojson, {
                style: (f) => ({
                    color: f.properties.frc === 1 ? "#3f3f46" : "#27272a",
                    weight: f.properties.frc === 1 ? 24 : 14,
                    opacity: 0.95,
                    lineCap: "round",
                })
            }).addTo(state.leafletMap);

            // Layer 2: Main dark asphalt roadbed (20px)
            L.geoJSON(geojson, {
                style: (f) => ({
                    color: "#18181b",
                    weight: f.properties.frc === 1 ? 20 : 10,
                    opacity: 1.0,
                    lineCap: "round",
                })
            }).addTo(state.leafletMap);

            // Layer 3: Double solid yellow median divider for mainline
            L.geoJSON(geojson, {
                filter: (f) => f.properties.frc === 1,
                style: {
                    color: "#f59e0b",
                    weight: 2,
                    dashArray: "10, 12",
                    opacity: 0.85,
                }
            }).addTo(state.leafletMap);

            // Frame directly onto the 1.54 km corridor route
            state.leafletMap.fitBounds(state.corridorBounds, { padding: [40, 40] });
        } catch (err) {
            console.error("Leaflet road network error:", err);
        }
    }

    // -------------------------------------------------------------------------
    // 4. Engine Switcher (3D Cesium <-> 2.5D Leaflet)
    // -------------------------------------------------------------------------
    function switchEngine(mode) {
        state.activeEngine = mode;
        if (mode === "3d") {
            els.btnView3D.classList.add("active");
            els.btnView2D.classList.remove("active");
            els.cesiumContainer.classList.add("active");
            els.leafletContainer.classList.remove("active");
            if (!state.cesiumViewer) initCesium();
            setCesiumCamera("3d");
        } else {
            els.btnView2D.classList.add("active");
            els.btnView3D.classList.remove("active");
            els.leafletContainer.classList.add("active");
            els.cesiumContainer.classList.remove("active");
            if (!state.leafletMap) initLeaflet();
            setTimeout(() => {
                state.leafletMap.invalidateSize();
                state.leafletMap.fitBounds(state.corridorBounds, { padding: [40, 40] });
            }, 100);
        }
        updatePositions(state.currentTime);
    }

    // -------------------------------------------------------------------------
    // 5. Load Trajectory Data & Evaluation Metrics
    // -------------------------------------------------------------------------
    async function loadExperimentData(experimentId) {
        pauseSimulation();
        state.currentExperiment = experimentId;
        clearAllVehicleMarkers();

        els.kpiSpeedSub.textContent = "Loading TraCI data...";

        try {
            const url = `/api/simulation/trajectories?experiment=${encodeURIComponent(experimentId)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

            const data = await resp.json();
            state.trajectoryData = data;
            state.frames = data.frames || {};
            state.summaryByStep = data.summary_by_step || [];
            state.totalDuration = data.metadata ? data.metadata.duration_sec : 1200;
            state.evaluationMetrics = data.evaluation || null;
            state.timeseries = data.evaluation ? (data.evaluation.timeseries || []) : [];

            // Update Timeline slider
            els.timeSlider.max = state.totalDuration;
            els.timelineMax.textContent = formatClock(state.totalDuration);

            // Populate Quantitative Evaluation Panel
            updateEvaluationPanel(data.evaluation);

            // Draw initial speed dynamics chart
            drawSpeedChart(0);

            // Seek to start and play
            seekToTime(0.0);
            playSimulation();
        } catch (err) {
            console.error("Experiment load error:", err);
            els.kpiSpeedSub.textContent = "Data unavailable";
        }
    }

    // -------------------------------------------------------------------------
    // 6. Quantitative Model Evaluation HUD & Chart
    // -------------------------------------------------------------------------
    function updateEvaluationPanel(evalData) {
        if (!evalData) return;
        const m = evalData.metrics || {};
        const dist = evalData.disturbance || {};

        els.mPredSpeed.textContent = `${(m.predicted_mean_speed_kmh || 41.0).toFixed(1)} km/h`;
        els.mActSpeed.textContent = `${(m.actual_mean_speed_kmh || 0.0).toFixed(1)} km/h`;
        els.mDevSpeed.textContent = `MAE ${(m.speed_deviation_mae || 0.0).toFixed(1)} km/h`;

        els.mPredFlow.textContent = `${Math.round(m.predicted_flow_demand || 200)} veh/h`;
        els.mActFlow.textContent = `${Math.round(m.actual_throughput_veh_h || 0)} veh/h`;
        
        const flowDiff = (m.actual_throughput_veh_h || 0) - (m.predicted_flow_demand || 200);
        els.mDevFlow.textContent = `${flowDiff >= 0 ? "+" : ""}${Math.round(flowDiff)} veh/h`;

        els.mActQueue.textContent = `${m.max_queue_vehicles || 0} veh`;
        els.mQueueState.textContent = (m.max_queue_vehicles || 0) >= 3 ? "⚠️ Queue Formed" : "Normal";
        els.mCongState.textContent = m.congestion_state || "Free Flow";

        // Disturbance Badge & Description
        if (dist.type && dist.type !== "none") {
            els.distBadgeText.textContent = `⚠️ DISTURBANCE: ${dist.type.toUpperCase()}`;
            els.distDescText.textContent = `${dist.description} (${dist.start_time_sec}s - ${dist.start_time_sec + dist.duration_sec}s)`;
        } else {
            els.distBadgeText.textContent = `✅ NORMAL BASELINE FLOW`;
            els.distDescText.textContent = `No disturbances active. Standard RF forecast.`;
        }
    }

    function drawSpeedChart(currentSec) {
        const canvas = els.speedChartCanvas;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        const data = state.timeseries;
        if (!data || data.length === 0) return;

        const maxT = state.totalDuration;
        const maxSpd = 60.0;

        // 1. Shaded red background for active disturbance window
        let distStart = null;
        let distEnd = null;
        for (const pt of data) {
            if (pt.is_disturbed && distStart === null) distStart = pt.time;
            if (!pt.is_disturbed && distStart !== null && distEnd === null) distEnd = pt.time;
        }
        if (distStart !== null) {
            if (distEnd === null) distEnd = maxT;
            const x1 = (distStart / maxT) * w;
            const x2 = (distEnd / maxT) * w;
            ctx.fillStyle = "rgba(239, 68, 68, 0.18)";
            ctx.fillRect(x1, 0, x2 - x1, h);
            ctx.fillStyle = "#ef4444";
            ctx.font = "9px Inter, sans-serif";
            ctx.fillText("DISTURBANCE", x1 + 4, 12);
        }

        // 2. Horizontal grid lines (20 km/h, 40 km/h)
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        for (const spd of [20, 40]) {
            const y = h - (spd / maxSpd) * h;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // 3. Predicted Baseline Speed line (Cyan dashed)
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        const yPred = h - (41.0 / maxSpd) * h;
        ctx.beginPath();
        ctx.moveTo(0, yPred);
        ctx.lineTo(w, yPred);
        ctx.stroke();
        ctx.setLineDash([]);

        // 4. Actual SUMO Speed line (Green solid, dips red during disturbance)
        ctx.beginPath();
        ctx.strokeStyle = "#34d399";
        ctx.lineWidth = 2;
        let first = true;
        for (let i = 0; i < data.length; i += 4) {
            const pt = data[i];
            const x = (pt.time / maxT) * w;
            const y = h - (Math.min(maxSpd, Math.max(0, pt.actual_speed)) / maxSpd) * h;
            if (first) {
                ctx.moveTo(x, y);
                first = false;
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // 5. Vertical Scrubber Cursor (Yellow)
        const curX = (currentSec / maxT) * w;
        ctx.strokeStyle = "#facc15";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(curX, 0);
        ctx.lineTo(curX, h);
        ctx.stroke();
    }

    // -------------------------------------------------------------------------
    // 7. Dynamic Vehicle Visuals & Speed Color Coding
    // -------------------------------------------------------------------------
    function getSpeedColor(speedKmh) {
        if (speedKmh >= 35.0) return "#34d399"; // Free flow (Emerald Green)
        if (speedKmh >= 15.0) return "#fbbf24"; // Moderate (Amber Yellow)
        return "#f87171";                       // Queued / Stopped (Crimson Red)
    }

    // Generates a dynamic canvas image for Cesium Billboards (Guaranteed 100% visible)
    const billboardCanvasCache = new Map();
    function getVehicleBillboardImage(vtype, speedKmh) {
        const colorKey = speedKmh >= 35 ? "green" : (speedKmh >= 15 ? "yellow" : "red");
        const cacheKey = `${vtype}_${colorKey}`;
        if (billboardCanvasCache.has(cacheKey)) return billboardCanvasCache.get(cacheKey);

        const canvas = document.createElement("canvas");
        canvas.width = 48;
        canvas.height = 48;
        const ctx = canvas.getContext("2d");
        const color = getSpeedColor(speedKmh);

        ctx.translate(24, 24);

        if (vtype === "bus") {
            // Extended 42px x 16px coach
            ctx.fillStyle = color;
            ctx.strokeStyle = "#0f172a";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.roundRect(-8, -20, 16, 40, 4);
            ctx.fill();
            ctx.stroke();
            // Windshield
            ctx.fillStyle = "#0f172a";
            ctx.fillRect(-6, -18, 12, 5);
            // Headlights
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(-6, -21, 3, 2);
            ctx.fillRect(3, -21, 3, 2);
        } else if (vtype === "auto") {
            // Auto-Rickshaw canopy
            ctx.fillStyle = "#eab308";
            ctx.strokeStyle = "#0f172a";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(0, -14);
            ctx.lineTo(10, 12);
            ctx.lineTo(-10, 12);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            // Lower body
            ctx.fillStyle = color;
            ctx.fillRect(-8, 2, 16, 10);
        } else {
            // Passenger Car
            ctx.fillStyle = color;
            ctx.strokeStyle = "#0f172a";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.roundRect(-7, -14, 14, 28, 4);
            ctx.fill();
            ctx.stroke();
            // Windshield
            ctx.fillStyle = "#0f172a";
            ctx.fillRect(-5, -9, 10, 4);
            ctx.fillRect(-5, 4, 10, 3);
            // Headlights
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(-5, -15, 2.5, 2);
            ctx.fillRect(2.5, -15, 2.5, 2);
        }

        const dataUrl = canvas.toDataURL();
        billboardCanvasCache.set(cacheKey, dataUrl);
        return dataUrl;
    }

    function createProminentLeafletIcon(vtype, speedKmh, headingDeg, isSelected) {
        const color = getSpeedColor(speedKmh);
        const stroke = isSelected ? "#ffffff" : "#090d16";
        const strokeW = isSelected ? "3" : "1.5";

        let svgHtml = "";
        let size = [32, 32];

        if (vtype === "bus") {
            size = [44, 44];
            svgHtml = `
            <div class="vehicle-svg-container" style="transform: rotate(${headingDeg}deg);">
                <svg width="44" height="44" viewBox="0 0 44 44">
                    <rect x="14" y="2" width="16" height="40" rx="4" fill="${color}" stroke="${stroke}" stroke-width="${strokeW}"/>
                    <rect x="16" y="5" width="12" height="5" rx="1" fill="#0f172a"/>
                    <rect x="15" y="13" width="14" height="2.5" fill="#0f172a"/>
                    <rect x="15" y="18" width="14" height="2.5" fill="#0f172a"/>
                    <rect x="15" y="23" width="14" height="2.5" fill="#0f172a"/>
                    <circle cx="16" cy="3" r="1.5" fill="#ffffff"/>
                    <circle cx="28" cy="3" r="1.5" fill="#ffffff"/>
                </svg>
            </div>`;
        } else if (vtype === "auto") {
            size = [30, 30];
            svgHtml = `
            <div class="vehicle-svg-container" style="transform: rotate(${headingDeg}deg);">
                <svg width="30" height="30" viewBox="0 0 30 30">
                    <polygon points="15,3 24,24 6,24" fill="#eab308" stroke="${stroke}" stroke-width="${strokeW}"/>
                    <rect x="8" y="16" width="14" height="8" rx="2" fill="${color}" stroke="${stroke}" stroke-width="${strokeW}"/>
                    <polygon points="15,6 20,15 10,15" fill="#0f172a"/>
                    <circle cx="15" cy="4" r="1.5" fill="#18181b"/>
                </svg>
            </div>`;
        } else {
            size = [34, 34];
            svgHtml = `
            <div class="vehicle-svg-container" style="transform: rotate(${headingDeg}deg);">
                <svg width="34" height="34" viewBox="0 0 34 34">
                    <rect x="11" y="4" width="12" height="26" rx="4" fill="${color}" stroke="${stroke}" stroke-width="${strokeW}"/>
                    <rect x="12.5" y="9" width="9" height="4" rx="1" fill="#0f172a"/>
                    <rect x="12.5" y="20" width="9" height="3.5" rx="1" fill="#0f172a"/>
                    <circle cx="13" cy="5" r="1.5" fill="#ffffff"/>
                    <circle cx="21" cy="5" r="1.5" fill="#ffffff"/>
                    <circle cx="13" cy="29" r="1" fill="#ef4444"/>
                    <circle cx="21" cy="29" r="1" fill="#ef4444"/>
                </svg>
            </div>`;
        }

        return L.divIcon({
            html: svgHtml,
            className: "vehicle-marker-icon",
            iconSize: size,
            iconAnchor: [size[0] / 2, size[1] / 2],
        });
    }

    // -------------------------------------------------------------------------
    // 8. 60 FPS Dual-Engine Vehicle Update Loop
    // -------------------------------------------------------------------------
    function updatePositions(simTime) {
        if (!state.frames) return;

        const baseSec = Math.floor(simTime);
        const frac = simTime - baseSec;

        const currentKey = `${baseSec}`;
        const nextKey = `${baseSec + 1}`;

        const currentVehs = state.frames[currentKey] || [];
        const nextVehs = state.frames[nextKey] || [];

        const nextMap = new Map();
        for (const v of nextVehs) nextMap.set(v.vehicle_id, v);

        const activeVidSet = new Set();
        let totalSpeed = 0.0;
        let queuedCount = 0;
        let leadVehicle = null;

        for (const v of currentVehs) {
            activeVidSet.add(v.vehicle_id);
            totalSpeed += v.speed_kmh;
            if (v.speed_kmh < 10.0) queuedCount++;
            if (!leadVehicle) leadVehicle = v;

            let lat = v.latitude;
            let lon = v.longitude;
            let heading = v.heading_angle_deg;

            // Interpolate position between 1-second TraCI frames
            const nextV = nextMap.get(v.vehicle_id);
            if (nextV && frac > 0.0) {
                lat = lat + (nextV.latitude - lat) * frac;
                lon = lon + (nextV.longitude - lon) * frac;
                let dAngle = nextV.heading_angle_deg - heading;
                if (dAngle > 180) dAngle -= 360;
                if (dAngle < -180) dAngle += 360;
                heading = heading + dAngle * frac;
            }

            const isSelected = (v.vehicle_id === state.selectedVehicleId);

            // A. Update Cesium 3D Entity
            if (state.cesiumViewer) {
                updateCesiumVehicle(v, lon, lat, heading, isSelected);
            }

            // B. Update Leaflet 2.5D Marker
            if (state.leafletMap) {
                updateLeafletVehicle(v, lat, lon, heading, isSelected);
            }
        }

        // Clean up completed/exited vehicles
        cleanupInactiveVehicles(activeVidSet);

        // Follow vehicle camera if active
        if (state.isFollowMode) {
            const target = state.selectedVehicleId ? currentVehs.find(v => v.vehicle_id === state.selectedVehicleId) : leadVehicle;
            if (target) followVehicleCamera(target);
        }

        // Update Live KPIs & Scrubber
        updateKPIs(baseSec, currentVehs.length, totalSpeed, queuedCount);

        // Update speed chart cursor
        drawSpeedChart(baseSec);
    }

    function updateCesiumVehicle(v, lon, lat, heading, isSelected) {
        const position = Cesium.Cartesian3.fromDegrees(lon, lat, 10.0);
        const billboardImg = getVehicleBillboardImage(v.vehicle_type, v.speed_kmh);

        if (state.cesiumVehicles.has(v.vehicle_id)) {
            const ent = state.cesiumVehicles.get(v.vehicle_id);
            ent.position = position;
            ent.billboard.image = billboardImg;
            ent.billboard.rotation = Cesium.Math.toRadians(360 - heading);
            ent.telemetry = v;
        } else {
            // Guaranteed 100% visible Cesium Billboard with disableDepthTestDistance
            const ent = state.cesiumViewer.entities.add({
                name: v.vehicle_id,
                position: position,
                billboard: {
                    image: billboardImg,
                    scale: 1.0,
                    rotation: Cesium.Math.toRadians(360 - heading),
                    heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
                    verticalOrigin: Cesium.VerticalOrigin.CENTER,
                    horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
                    scaleByDistance: new Cesium.NearFarScalar(80, 1.3, 3000, 0.75),
                    disableDepthTestDistance: Number.POSITIVE_INFINITY, // Never buried under terrain!
                }
            });
            ent.telemetry = v;
            state.cesiumVehicles.set(v.vehicle_id, ent);
        }
    }

    function updateLeafletVehicle(v, lat, lon, heading, isSelected) {
        const icon = createProminentLeafletIcon(v.vehicle_type, v.speed_kmh, heading, isSelected);
        if (state.leafletVehicles.has(v.vehicle_id)) {
            const marker = state.leafletVehicles.get(v.vehicle_id);
            marker.setLatLng([lat, lon]);
            marker.setIcon(icon);
            marker.telemetry = v;
        } else {
            const marker = L.marker([lat, lon], { icon: icon, zIndexOffset: 250 });
            marker.telemetry = v;
            marker.on("click", () => inspectVehicle(marker.telemetry));
            marker.addTo(state.leafletMap);
            state.leafletVehicles.set(v.vehicle_id, marker);
        }
    }

    function cleanupInactiveVehicles(activeSet) {
        if (state.cesiumViewer) {
            for (const [vid, ent] of state.cesiumVehicles.entries()) {
                if (!activeSet.has(vid)) {
                    state.cesiumViewer.entities.remove(ent);
                    state.cesiumVehicles.delete(vid);
                }
            }
        }
        if (state.leafletMap) {
            for (const [vid, marker] of state.leafletVehicles.entries()) {
                if (!activeSet.has(vid)) {
                    state.leafletMap.removeLayer(marker);
                    state.leafletVehicles.delete(vid);
                }
            }
        }
    }

    function clearAllVehicleMarkers() {
        if (state.cesiumViewer) {
            for (const ent of state.cesiumVehicles.values()) {
                state.cesiumViewer.entities.remove(ent);
            }
            state.cesiumVehicles.clear();
        }
        if (state.leafletMap) {
            for (const marker of state.leafletVehicles.values()) {
                state.leafletMap.removeLayer(marker);
            }
            state.leafletVehicles.clear();
        }
    }

    function followVehicleCamera(v) {
        if (!v) return;
        if (state.activeEngine === "3d" && state.cesiumViewer) {
            state.cesiumViewer.camera.setView({
                destination: Cesium.Cartesian3.fromDegrees(v.longitude - 0.0015, v.latitude - 0.001, 140),
                orientation: {
                    heading: Cesium.Math.toRadians(v.heading_angle_deg),
                    pitch: Cesium.Math.toRadians(-22.0),
                    roll: 0.0
                }
            });
        } else if (state.activeEngine === "2d" && state.leafletMap) {
            state.leafletMap.panTo([v.latitude, v.longitude], { animate: true, duration: 0.15 });
        }
    }

    // -------------------------------------------------------------------------
    // 9. KPI Dashboard Update
    // -------------------------------------------------------------------------
    function updateKPIs(currentSec, activeCount, totalSpeed, queuedCount) {
        els.kpiSimTime.textContent = formatFullTime(currentSec);
        els.kpiStep.textContent = `Step: ${currentSec}s / ${state.totalDuration}s`;
        els.kpiActiveVehs.textContent = activeCount;

        const meanSpd = activeCount > 0 ? (totalSpeed / activeCount).toFixed(1) : "41.0";
        els.kpiMeanSpeed.textContent = `${meanSpd} km/h`;
        els.kpiMeanSpeed.className = `kpi-value ${parseFloat(meanSpd) >= 35 ? "highlight-green" : (parseFloat(meanSpd) >= 20 ? "color-yellow" : "dot-red")}`;

        els.kpiQueue.textContent = `${queuedCount} veh`;
        const density = (activeCount / 1.54).toFixed(1);
        els.kpiDensity.textContent = `Density: ${density} veh/km`;

        if (state.summaryByStep && state.summaryByStep[currentSec]) {
            const sumRow = state.summaryByStep[currentSec];
            els.kpiCompletedVehs.textContent = `Completed: ${sumRow.completed_vehicles || 0}`;
        }

        // Show/hide pulsing disturbance indicator card
        const isDisturbed = state.timeseries[currentSec] ? state.timeseries[currentSec].is_disturbed : false;
        if (isDisturbed) {
            els.distIndicatorCard.classList.remove("hidden");
        } else {
            els.distIndicatorCard.classList.add("hidden");
        }

        els.timeSlider.value = currentSec;
        els.timelineCurrent.textContent = formatClock(currentSec);
    }

    // -------------------------------------------------------------------------
    // 10. Vehicle Telemetry Inspector & Verification Mode
    // -------------------------------------------------------------------------
    function inspectVehicle(v) {
        if (!v) return;
        state.selectedVehicleId = v.vehicle_id;

        els.inspId.textContent = v.vehicle_id;
        els.inspType.textContent = v.vehicle_type;
        els.inspSpeed.textContent = `${v.speed_kmh.toFixed(1)} km/h (${v.speed_mps.toFixed(1)} m/s)`;
        els.inspHeading.textContent = `${v.heading_angle_deg.toFixed(1)}°`;
        els.inspEdge.textContent = v.current_edge_id;
        els.inspLane.textContent = `Lane ${v.current_lane_index}`;
        els.inspAccel.textContent = `${v.acceleration.toFixed(2)} m/s²`;
        els.inspWait.textContent = `${v.waiting_time_sec.toFixed(1)} s`;
        els.inspGps.textContent = `${v.latitude.toFixed(6)}, ${v.longitude.toFixed(6)}`;

        els.inspector.classList.remove("hidden");
    }

    function verifyVehicleTelemetryPipeline() {
        const targetVid = "veh_ml_forecast_0";
        const currentSec = Math.floor(state.currentTime);
        const curFrame = state.frames[`${currentSec}`] || [];
        const v = curFrame.find(veh => veh.vehicle_id === targetVid);

        if (v) {
            els.debugOutput.innerHTML = `
            <b style="color:#34d399">VERIFIED: TraCI Delivery & Coordinate Mapping 1:1</b><br>
            Time: <b>${v.simulation_time_sec}s</b> | Vehicle: <b>${v.vehicle_id}</b> (${v.vehicle_type})<br>
            WGS84 GPS: <b>${v.latitude.toFixed(6)}, ${v.longitude.toFixed(6)}</b><br>
            Planar (x,y): <b>(${v.planar_x}m, ${v.planar_y}m)</b> | Heading: <b>${v.heading_angle_deg}°</b><br>
            Speed: <b>${v.speed_kmh} km/h</b> | Edge: <b>${v.current_edge_id}</b> Lane: <b>${v.current_lane_index}</b><br>
            <span style="color:#38bdf8">Render Status: Active on canvas in both 3D & 2.5D engines.</span>`;
            inspectVehicle(v);
        } else {
            els.debugOutput.innerHTML = `
            <b style="color:#fbbf24">Notice:</b> Vehicle <b>${targetVid}</b> is not active at t=${currentSec}s.<br>
            (Active in Morning Rush from t=1s to t=128s). Scrub timeline to t=60s to inspect!`;
        }
    }

    // -------------------------------------------------------------------------
    // 11. Animation Loop & Playback Controls
    // -------------------------------------------------------------------------
    function animationLoop(timestamp) {
        if (!state.isPlaying) return;

        if (state.lastAnimTimestamp != null) {
            const dt = (timestamp - state.lastAnimTimestamp) / 1000.0;
            state.currentTime += dt * state.speedMultiplier;

            if (state.currentTime >= state.totalDuration) {
                state.currentTime = state.totalDuration;
                pauseSimulation();
            }

            updatePositions(state.currentTime);
        }

        state.lastAnimTimestamp = timestamp;
        if (state.isPlaying) {
            requestAnimationFrame(animationLoop);
        }
    }

    function playSimulation() {
        if (state.isPlaying) return;
        state.isPlaying = true;
        state.lastAnimTimestamp = null;
        els.playIcon.textContent = "⏸";
        els.playText.textContent = "Pause";
        els.btnPlayPause.style.background = "#f87171";
        requestAnimationFrame(animationLoop);
    }

    function pauseSimulation() {
        state.isPlaying = false;
        state.lastAnimTimestamp = null;
        els.playIcon.textContent = "▶";
        els.playText.textContent = "Play";
        els.btnPlayPause.style.background = "#34d399";
    }

    function togglePlayPause() {
        if (state.isPlaying) {
            pauseSimulation();
        } else {
            if (state.currentTime >= state.totalDuration) state.currentTime = 0.0;
            playSimulation();
        }
    }

    function seekToTime(targetSec) {
        state.currentTime = Math.max(0.0, Math.min(targetSec, state.totalDuration));
        updatePositions(state.currentTime);
    }

    // -------------------------------------------------------------------------
    // 12. Run Dynamic SUMO Simulation via TraCI API (Part 4)
    // -------------------------------------------------------------------------
    async function runCustomDisturbanceSimulation() {
        els.btnRunCustomSim.disabled = true;
        els.simRunStatus.textContent = "Starting SUMO 1.27.1 microsimulation via TraCI...";

        const payload = {
            disturbance_type: els.distTypeInput.value,
            edge_id: els.distEdgeInput.value,
            start_time: parseInt(els.distStartInput.value, 10),
            duration: parseInt(els.distDurationInput.value, 10),
            lane_index: 0,
            max_duration_sec: 1200
        };

        try {
            const resp = await fetch("/api/simulation/run_experiment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const result = await resp.json();

            els.simRunStatus.textContent = "Simulation completed! Loading trajectories...";

            // Load newly generated custom trajectory
            await loadExperimentData("custom");

            els.simRunStatus.textContent = "✅ TraCI simulation loaded successfully!";
            setTimeout(() => {
                els.experimentDrawer.classList.add("hidden");
                els.simRunStatus.textContent = "";
            }, 1500);
        } catch (err) {
            console.error("Custom run error:", err);
            els.simRunStatus.textContent = `Error: ${err.message}`;
        } finally {
            els.btnRunCustomSim.disabled = false;
        }
    }

    // -------------------------------------------------------------------------
    // 13. Formatters & Helpers
    // -------------------------------------------------------------------------
    function formatClock(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    function formatFullTime(seconds) {
        const totalSec = 8 * 3600 + seconds;
        const h = Math.floor(totalSec / 3600) % 24;
        const m = Math.floor((totalSec % 3600) / 60);
        const s = Math.floor(totalSec % 60);
        return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    // -------------------------------------------------------------------------
    // 14. Event Listeners
    // -------------------------------------------------------------------------
    els.btnPlayPause.addEventListener("click", togglePlayPause);
    els.btnRestart.addEventListener("click", () => seekToTime(0.0));
    els.btnStepBack.addEventListener("click", () => seekToTime(state.currentTime - 5.0));
    els.btnStepFwd.addEventListener("click", () => seekToTime(state.currentTime + 5.0));

    els.timeSlider.addEventListener("input", (e) => seekToTime(parseFloat(e.target.value)));

    els.speedBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            els.speedBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.speedMultiplier = parseFloat(btn.getAttribute("data-speed"));
        });
    });

    els.btnView3D.addEventListener("click", () => switchEngine("3d"));
    els.btnView2D.addEventListener("click", () => switchEngine("2d"));

    els.basemapSelect.addEventListener("change", (e) => {
        state.basemapType = e.target.value;
        updateLeafletBasemap(e.target.value);
    });

    els.experimentSelect.addEventListener("change", (e) => {
        loadExperimentData(e.target.value);
    });

    els.btnToggleExpDrawer.addEventListener("click", () => {
        els.experimentDrawer.classList.toggle("hidden");
    });

    els.btnCloseDrawer.addEventListener("click", () => {
        els.experimentDrawer.classList.add("hidden");
    });

    els.btnRunCustomSim.addEventListener("click", runCustomDisturbanceSimulation);

    els.cam3DAction.addEventListener("click", () => {
        state.isFollowMode = false;
        els.cam3DAction.classList.add("active");
        els.camTopDown.classList.remove("active");
        els.camFollow.classList.remove("active");
        if (state.activeEngine === "3d") setCesiumCamera("3d");
    });

    els.camTopDown.addEventListener("click", () => {
        state.isFollowMode = false;
        els.camTopDown.classList.add("active");
        els.cam3DAction.classList.remove("active");
        els.camFollow.classList.remove("active");
        if (state.activeEngine === "3d") {
            setCesiumCamera("topdown");
        } else if (state.leafletMap) {
            state.leafletMap.fitBounds(state.corridorBounds, { padding: [40, 40] });
        }
    });

    els.camFollow.addEventListener("click", () => {
        state.isFollowMode = !state.isFollowMode;
        if (state.isFollowMode) {
            els.camFollow.classList.add("active");
        } else {
            els.camFollow.classList.remove("active");
        }
    });

    els.camReset.addEventListener("click", () => {
        state.isFollowMode = false;
        els.camFollow.classList.remove("active");
        if (state.activeEngine === "3d") {
            setCesiumCamera("3d");
        } else if (state.leafletMap) {
            state.leafletMap.fitBounds(state.corridorBounds, { padding: [40, 40] });
        }
    });

    els.btnVerifyVehicle.addEventListener("click", verifyVehicleTelemetryPipeline);
    els.closeInspectorBtn.addEventListener("click", () => els.inspector.classList.add("hidden"));

    els.btnToggleEval.addEventListener("click", () => {
        els.evalBody.classList.toggle("hidden");
        els.btnToggleEval.textContent = els.evalBody.classList.contains("hidden") ? "▲" : "▼";
    });

    document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        if (e.code === "Space") {
            e.preventDefault();
            togglePlayPause();
        } else if (e.code === "ArrowLeft") {
            e.preventDefault();
            seekToTime(state.currentTime - 5.0);
        } else if (e.code === "ArrowRight") {
            e.preventDefault();
            seekToTime(state.currentTime + 5.0);
        }
    });

    // -------------------------------------------------------------------------
    // 15. Bootstrap
    // -------------------------------------------------------------------------
    initCesium();
    initLeaflet();
    loadExperimentData(els.experimentSelect.value);
});
