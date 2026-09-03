/**
 * Barapullah Corridor 3D & 2.5D Traffic Microsimulation Engine
 * -----------------------------------------------------------
 * Dual-Engine Visualizer:
 *   1. CesiumJS 3D Urban Digital Twin (3D Extruded Buildings, Esri Satellite, Elevated Flyover, 3D Vehicles)
 *   2. 2.5D Corridor Operations Center (High-Contrast Multi-Lane Asphalt, Large Vehicle Glyphs)
 *
 * 100% Genuine SUMO TraCI Telemetry. Zero Fabricated Animations. Zero API Key Watermarks.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. Unified State Management
    // -------------------------------------------------------------------------
    const state = {
        activeEngine: "3d", // "3d" (Cesium) or "2d" (Leaflet)
        basemapType: "satellite", // "satellite", "dark", "osm"
        
        // Corridor Centers & Bounds (1.54 km simulated route)
        corridorCenter: [28.58342, 77.23791], // [lat, lon]
        corridorBounds: [[28.57967, 77.23187], [28.58716, 77.24395]],
        
        // Cesium 3D Engine State
        cesiumViewer: null,
        cesiumVehicles: new Map(), // vid -> Cesium.Entity
        cesiumBuildingsSource: null,
        cesiumRoadsSource: null,
        
        // 2.5D Leaflet Engine State
        leafletMap: null,
        leafletTiles: null,
        leafletVehicles: new Map(), // vid -> L.marker
        leafletBuildingsLayer: null,
        leafletRoadLayer: null,
        
        // Simulation Dataset
        scenario: "ml_forecast",
        period: "morning_rush",
        trajectoryData: null,
        frames: {},
        summaryByStep: [],
        totalDuration: 3600,
        
        // Playback State
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
        scenarioSelect: document.getElementById("scenarioSelect"),
        periodSelect: document.getElementById("periodSelect"),
        
        // Camera Buttons
        cam3DAction: document.getElementById("cam3DAction"),
        camTopDown: document.getElementById("camTopDown"),
        camFollow: document.getElementById("camFollow"),
        camReset: document.getElementById("camReset"),
        
        // Playback Controls
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
        
        // KPI Badges
        kpiSimTime: document.getElementById("kpiSimTime"),
        kpiStep: document.getElementById("kpiStep"),
        kpiActiveVehs: document.getElementById("kpiActiveVehs"),
        kpiCompletedVehs: document.getElementById("kpiCompletedVehs"),
        kpiMeanSpeed: document.getElementById("kpiMeanSpeed"),
        kpiDensity: document.getElementById("kpiDensity"),
        kpiDemandRate: document.getElementById("kpiDemandRate"),
        
        // Inspector
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
    };

    // -------------------------------------------------------------------------
    // 2. Tile Provider URLs (100% Free - ZERO API Key Required)
    // -------------------------------------------------------------------------
    const TILE_URLS = {
        satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        dark: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        osm: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    };

    // -------------------------------------------------------------------------
    // 3. Engine A: Initialize CesiumJS 3D Urban Digital Twin
    // -------------------------------------------------------------------------
    function initCesium() {
        if (typeof Cesium === "undefined") {
            console.warn("CesiumJS library not loaded, falling back to 2.5D mode.");
            switchEngine("2d");
            return;
        }

        // Disable Cesium Ion token prompt
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
            creditContainer: document.createElement("div"), // Hide default credit watermark
            imageryProvider: new Cesium.ArcGisMapServerImageryProvider({
                url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
            }),
        });

        // Set initial corridor-focused 3D flyover camera
        setCesiumCamera("3d");

        // Load 2,748 Real Buildings and Road Geometry in 3D
        loadCesiumBuildings();
        loadCesiumRoads();

        // Entity click handler for inspector
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
            // Pitched dramatic 3D flyover angle looking east along Barapullah flyover
            state.cesiumViewer.camera.flyTo({
                destination: Cesium.Cartesian3.fromDegrees(lon - 0.006, lat - 0.003, 520),
                orientation: {
                    heading: Cesium.Math.toRadians(75.0),
                    pitch: Cesium.Math.toRadians(-32.0),
                    roll: 0.0,
                },
                duration: 1.5,
            });
        } else if (mode === "topdown") {
            // Direct overhead corridor route framing
            state.cesiumViewer.camera.flyTo({
                destination: Cesium.Cartesian3.fromDegrees(lon, lat, 1400),
                orientation: {
                    heading: Cesium.Math.toRadians(0.0),
                    pitch: Cesium.Math.toRadians(-90.0),
                    roll: 0.0,
                },
                duration: 1.5,
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

            // Extrude buildings to real heights with architectural styling
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

            // Render Barapullah corridor roads as 3D elevated highway ribbon (+8.5m)
            for (const f of geojson.features) {
                const coords = f.geometry.coordinates;
                const positions = coords.map(c => Cesium.Cartesian3.fromDegrees(c[0], c[1], 8.5));
                const frc = f.properties.frc || 2;
                const width = frc === 1 ? 16 : (frc === 2 ? 12 : 8);

                state.cesiumViewer.entities.add({
                    polyline: {
                        positions: positions,
                        width: width,
                        material: new Cesium.PolylineGlowMaterialProperty({
                            glowPower: 0.15,
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
    // 4. Engine B: Initialize 2.5D Corridor Operations Center (Leaflet)
    // -------------------------------------------------------------------------
    function initLeaflet() {
        // Center immediately on the 1.54 km Barapullah route
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
            attribution: type === "satellite" ? "&copy; Esri & OpenStreetMap" : "&copy; OpenStreetMap",
        }).addTo(state.leafletMap);
    }

    async function loadLeafletBuildings() {
        try {
            const resp = await fetch("/api/simulation/buildings");
            if (!resp.ok) return;
            const geojson = await resp.json();

            state.leafletBuildingsLayer = L.geoJSON(geojson, {
                style: {
                    color: "#64748b",
                    weight: 1,
                    fillColor: "#1e293b",
                    fillOpacity: 0.75,
                },
                onEachFeature: (feature, layer) => {
                    const props = feature.properties;
                    layer.bindTooltip(`<b>${props.name}</b><br>Height: ${props.height.toFixed(1)}m (${props.levels} floors)`, { sticky: true });
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

            // Render layered road: Concrete barrier casing + Dark asphalt deck + Dashed lane line
            // Layer 1: Concrete barrier casing (wide)
            L.geoJSON(geojson, {
                style: (f) => ({
                    color: f.properties.frc === 1 ? "#52525b" : "#3f3f46",
                    weight: f.properties.frc === 1 ? 22 : 14,
                    opacity: 0.95,
                    lineCap: "round",
                })
            }).addTo(state.leafletMap);

            // Layer 2: Main dark asphalt roadbed
            L.geoJSON(geojson, {
                style: (f) => ({
                    color: "#18181b",
                    weight: f.properties.frc === 1 ? 18 : 10,
                    opacity: 1.0,
                    lineCap: "round",
                })
            }).addTo(state.leafletMap);

            // Layer 3: Yellow center median line for mainline
            state.leafletRoadLayer = L.geoJSON(geojson, {
                filter: (f) => f.properties.frc === 1,
                style: {
                    color: "#f59e0b",
                    weight: 2,
                    dashArray: "8, 12",
                    opacity: 0.85,
                },
                onEachFeature: (feature, layer) => {
                    layer.bindTooltip(`<b>${feature.properties.street_name}</b><br>FRC ${feature.properties.frc} · ${feature.properties.speed_limit_kmh} km/h`, { sticky: true });
                }
            }).addTo(state.leafletMap);

            // Frame directly to the 1.54 km corridor route
            state.leafletMap.fitBounds(state.corridorBounds, { padding: [40, 40] });
        } catch (err) {
            console.error("Leaflet road network error:", err);
        }
    }

    // -------------------------------------------------------------------------
    // 5. Engine Switching (3D Cesium <-> 2.5D Leaflet)
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
    // 6. Load Genuine SUMO Trajectory Data
    // -------------------------------------------------------------------------
    async function loadTrajectories(scenario, period) {
        pauseSimulation();
        state.scenario = scenario;
        state.period = period;
        clearAllVehicleMarkers();

        els.kpiDemandRate.textContent = "Loading TraCI data...";

        try {
            const url = `/api/simulation/trajectories?scenario=${encodeURIComponent(scenario)}&period=${encodeURIComponent(period)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}: Trajectory file not found`);

            const data = await resp.json();
            state.trajectoryData = data;
            state.frames = data.frames || {};
            state.summaryByStep = data.summary_by_step || [];
            state.totalDuration = data.metadata ? data.metadata.duration_sec : 3600;

            els.timeSlider.max = state.totalDuration;
            els.timelineMax.textContent = formatClock(state.totalDuration);

            const count = data.metadata ? data.metadata.total_unique_vehicles : 200;
            els.kpiDemandRate.textContent = `Demand: ~${count} veh/h`;

            seekToTime(state.currentTime || 0.0);
            playSimulation();
        } catch (err) {
            console.error("Trajectory load error:", err);
            els.kpiDemandRate.textContent = "Data unavailable";
        }
    }

    // -------------------------------------------------------------------------
    // 7. Vehicle SVG Glyphs (Prominent, Oriented, Speed-Colored)
    // -------------------------------------------------------------------------
    function getSpeedColor(speedKmh) {
        if (speedKmh >= 35.0) return "#34d399"; // Free flow (Emerald)
        if (speedKmh >= 15.0) return "#fbbf24"; // Moderate (Amber)
        return "#f87171";                       // Slow/Queue (Crimson)
    }

    function createProminentVehicleIcon(vtype, speedKmh, headingDeg, isSelected) {
        const color = getSpeedColor(speedKmh);
        const stroke = isSelected ? "#ffffff" : "#090d16";
        const strokeW = isSelected ? "3" : "1.5";

        let svgHtml = "";
        let size = [26, 26];

        if (vtype === "bus") {
            // Delhi DTC Transit Bus (Extended 42px x 16px coach)
            size = [36, 36];
            svgHtml = `
            <div class="vehicle-svg-container" style="transform: rotate(${headingDeg}deg);">
                <svg width="36" height="36" viewBox="0 0 36 36">
                    <!-- Bus Chassis -->
                    <rect x="11" y="2" width="14" height="32" rx="3.5" fill="${color}" stroke="${stroke}" stroke-width="${strokeW}"/>
                    <!-- Front Windshield -->
                    <rect x="12.5" y="4" width="11" height="4" rx="1" fill="#0f172a"/>
                    <!-- Windows -->
                    <rect x="12" y="10" width="12" height="2" fill="#0f172a"/>
                    <rect x="12" y="14" width="12" height="2" fill="#0f172a"/>
                    <rect x="12" y="18" width="12" height="2" fill="#0f172a"/>
                    <rect x="12" y="22" width="12" height="2" fill="#0f172a"/>
                    <!-- Headlights -->
                    <circle cx="13" cy="3" r="1.2" fill="#ffffff"/>
                    <circle cx="23" cy="3" r="1.2" fill="#ffffff"/>
                    <!-- Rear lights -->
                    <circle cx="13" cy="33" r="1" fill="#ef4444"/>
                    <circle cx="23" cy="33" r="1" fill="#ef4444"/>
                </svg>
            </div>`;
        } else if (vtype === "auto") {
            // Auto-Rickshaw (Indian three-wheeler: vibrant green base, bright yellow canopy)
            size = [28, 28];
            svgHtml = `
            <div class="vehicle-svg-container" style="transform: rotate(${headingDeg}deg);">
                <svg width="28" height="28" viewBox="0 0 28 28">
                    <!-- Auto Triangular Canopy -->
                    <polygon points="14,3 22,22 6,22" fill="#eab308" stroke="${stroke}" stroke-width="${strokeW}"/>
                    <!-- Green Lower Body -->
                    <rect x="8" y="16" width="12" height="7" rx="2" fill="${color}" stroke="${stroke}" stroke-width="${strokeW}"/>
                    <!-- Windshield -->
                    <polygon points="14,6 19,15 9,15" fill="#0f172a"/>
                    <!-- Front Single Wheel -->
                    <circle cx="14" cy="4" r="1.5" fill="#18181b"/>
                    <!-- Rear Wheels -->
                    <circle cx="7" cy="20" r="1.5" fill="#18181b"/>
                    <circle cx="21" cy="20" r="1.5" fill="#18181b"/>
                </svg>
            </div>`;
        } else {
            // Passenger Car (Prominent 28px x 14px Sedan)
            size = [30, 30];
            svgHtml = `
            <div class="vehicle-svg-container" style="transform: rotate(${headingDeg}deg);">
                <svg width="30" height="30" viewBox="0 0 30 30">
                    <!-- Car Body -->
                    <rect x="9.5" y="4" width="11" height="22" rx="3.5" fill="${color}" stroke="${stroke}" stroke-width="${strokeW}"/>
                    <!-- Front Windshield -->
                    <rect x="11" y="8" width="8" height="3.5" rx="1" fill="#0f172a"/>
                    <!-- Rear Windshield -->
                    <rect x="11" y="18" width="8" height="3" rx="1" fill="#0f172a"/>
                    <!-- Headlights -->
                    <circle cx="11.5" cy="4.5" r="1.2" fill="#ffffff"/>
                    <circle cx="18.5" cy="4.5" r="1.2" fill="#ffffff"/>
                    <!-- Taillights -->
                    <circle cx="11.5" cy="25.5" r="1" fill="#ef4444"/>
                    <circle cx="18.5" cy="25.5" r="1" fill="#ef4444"/>
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
    // 8. Dual-Engine Vehicle Update Loop
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
        let leadVehicle = null;

        for (const v of currentVehs) {
            activeVidSet.add(v.vehicle_id);
            totalSpeed += v.speed_kmh;
            if (!leadVehicle) leadVehicle = v;

            let lat = v.latitude;
            let lon = v.longitude;
            let heading = v.heading_angle_deg;

            // Interpolate coordinates if next frame exists
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

        // Clean up inactive markers
        cleanupInactiveVehicles(activeVidSet);

        // Follow vehicle camera if active
        if (state.isFollowMode) {
            const target = state.selectedVehicleId ? currentVehs.find(v => v.vehicle_id === state.selectedVehicleId) : leadVehicle;
            if (target) followVehicleCamera(target);
        }

        // Update KPIs
        updateKPIs(baseSec, currentVehs.length, totalSpeed);
    }

    function updateCesiumVehicle(v, lon, lat, heading, isSelected) {
        const speedColor = getSpeedColor(v.speed_kmh);
        const position = Cesium.Cartesian3.fromDegrees(lon, lat, 9.5); // Elevated flyover height
        const headingRad = Cesium.Math.toRadians(heading - 90);
        const hpr = new Cesium.HeadingPitchRoll(headingRad, 0, 0);
        const orientation = Cesium.Transforms.headingPitchRollQuaternion(position, hpr);

        let dims = new Cesium.Cartesian3(5.0, 2.2, 1.6); // Car
        if (v.vehicle_type === "bus") dims = new Cesium.Cartesian3(12.0, 2.8, 3.4);
        if (v.vehicle_type === "auto") dims = new Cesium.Cartesian3(3.0, 1.6, 2.0);

        if (state.cesiumVehicles.has(v.vehicle_id)) {
            const ent = state.cesiumVehicles.get(v.vehicle_id);
            ent.position = position;
            ent.orientation = orientation;
            ent.box.material = Cesium.Color.fromCssColorString(speedColor);
            ent.telemetry = v;
        } else {
            const ent = state.cesiumViewer.entities.add({
                name: v.vehicle_id,
                position: position,
                orientation: orientation,
                box: {
                    dimensions: dims,
                    material: Cesium.Color.fromCssColorString(speedColor),
                    outline: true,
                    outlineColor: Cesium.Color.fromCssColorString(isSelected ? "#ffffff" : "#000000"),
                }
            });
            ent.telemetry = v;
            state.cesiumVehicles.set(v.vehicle_id, ent);
        }
    }

    function updateLeafletVehicle(v, lat, lon, heading, isSelected) {
        const icon = createProminentVehicleIcon(v.vehicle_type, v.speed_kmh, heading, isSelected);
        if (state.leafletVehicles.has(v.vehicle_id)) {
            const marker = state.leafletVehicles.get(v.vehicle_id);
            marker.setLatLng([lat, lon]);
            marker.setIcon(icon);
            marker.telemetry = v;
        } else {
            const marker = L.marker([lat, lon], { icon: icon, zIndexOffset: 200 });
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
                destination: Cesium.Cartesian3.fromDegrees(v.longitude - 0.002, v.latitude - 0.001, 150),
                orientation: {
                    heading: Cesium.Math.toRadians(v.heading_angle_deg),
                    pitch: Cesium.Math.toRadians(-25.0),
                    roll: 0.0
                }
            });
        } else if (state.activeEngine === "2d" && state.leafletMap) {
            state.leafletMap.panTo([v.latitude, v.longitude], { animate: true, duration: 0.2 });
        }
    }

    // -------------------------------------------------------------------------
    // 9. KPI Metrics Update
    // -------------------------------------------------------------------------
    function updateKPIs(currentSec, activeCount, totalSpeed) {
        els.kpiSimTime.textContent = formatFullTime(currentSec);
        els.kpiStep.textContent = `Step: ${currentSec}s / ${state.totalDuration}s`;
        els.kpiActiveVehs.textContent = activeCount;

        const meanSpd = activeCount > 0 ? (totalSpeed / activeCount).toFixed(1) : "0.0";
        els.kpiMeanSpeed.textContent = `${meanSpd} km/h`;

        const density = (activeCount / 1.54).toFixed(1);
        els.kpiDensity.textContent = `${density} veh/km`;

        if (state.summaryByStep && state.summaryByStep[currentSec]) {
            const sumRow = state.summaryByStep[currentSec];
            els.kpiCompletedVehs.textContent = `Completed: ${sumRow.completed_vehicles || 0}`;
        }

        els.timeSlider.value = currentSec;
        els.timelineCurrent.textContent = formatClock(currentSec);
    }

    // -------------------------------------------------------------------------
    // 10. Vehicle Telemetry Inspector
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

    function closeInspector() {
        state.selectedVehicleId = null;
        els.inspector.classList.add("hidden");
    }

    // -------------------------------------------------------------------------
    // 11. 60 FPS Animation Loop
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
            if (state.currentTime >= state.totalDuration) {
                state.currentTime = 0.0;
            }
            playSimulation();
        }
    }

    function seekToTime(targetSec) {
        state.currentTime = Math.max(0.0, Math.min(targetSec, state.totalDuration));
        updatePositions(state.currentTime);
    }

    // -------------------------------------------------------------------------
    // 12. Helpers & Formatters
    // -------------------------------------------------------------------------
    function formatClock(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    function formatFullTime(seconds) {
        let baseH = 8;
        if (state.period === "evening_rush") baseH = 17;
        if (state.period === "off_peak") baseH = 13;

        const totalSec = baseH * 3600 + seconds;
        const h = Math.floor(totalSec / 3600) % 24;
        const m = Math.floor((totalSec % 3600) / 60);
        const s = Math.floor(totalSec % 60);
        return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    // -------------------------------------------------------------------------
    // 13. Event Listeners
    // -------------------------------------------------------------------------
    els.btnPlayPause.addEventListener("click", togglePlayPause);
    els.btnRestart.addEventListener("click", () => seekToTime(0.0));
    els.btnStepBack.addEventListener("click", () => seekToTime(state.currentTime - 5.0));
    els.btnStepFwd.addEventListener("click", () => seekToTime(state.currentTime + 5.0));

    els.timeSlider.addEventListener("input", (e) => {
        seekToTime(parseFloat(e.target.value));
    });

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

    els.scenarioSelect.addEventListener("change", (e) => {
        loadTrajectories(e.target.value, state.period);
    });

    els.periodSelect.addEventListener("change", (e) => {
        loadTrajectories(state.scenario, e.target.value);
    });

    els.closeInspectorBtn.addEventListener("click", closeInspector);

    // Keyboard Shortcuts
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
    // 14. Bootstrap
    // -------------------------------------------------------------------------
    // Start with 3D Cesium (or 2.5D fallback)
    initCesium();
    initLeaflet();
    loadTrajectories(els.scenarioSelect.value, els.periodSelect.value);
});
