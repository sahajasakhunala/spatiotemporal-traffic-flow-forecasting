/**
 * Barapullah Corridor SUMO Vehicle-Level Microsimulation Player
 * -------------------------------------------------------------
 * Visualizes 100% genuine TraCI vehicle telemetry over real OpenStreetMap geometry.
 * Dynamic vehicle icons (cars, auto-rickshaws, city buses), speed-based coloring,
 * full playback controls (play, pause, scrub, speed multiplier), and vehicle inspector.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. State Management
    // -------------------------------------------------------------------------
    const state = {
        map: null,
        networkLayer: null,
        vehicleMarkers: new Map(), // vid -> L.marker
        selectedVehicleId: null,
        
        // Simulation dataset
        scenario: "ml_forecast",
        period: "morning_rush",
        trajectoryData: null,
        frames: {},
        summaryByStep: [],
        totalDuration: 3600,
        
        // Playback state
        isPlaying: false,
        currentTime: 0.0,      // float seconds
        speedMultiplier: 2.0,  // 1x, 2x, 5x, 10x
        lastAnimTimestamp: null,
    };

    // DOM Elements
    const els = {
        scenarioSelect: document.getElementById("scenarioSelect"),
        periodSelect: document.getElementById("periodSelect"),
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
        
        // KPIs
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
    // 2. Map Initialization
    // -------------------------------------------------------------------------
    function initMap() {
        // Centered on Barapullah Corridor origin
        state.map = L.map("simMap", {
            center: [28.58215, 77.24072],
            zoom: 14,
            minZoom: 12,
            maxZoom: 18,
            zoomControl: true,
        });

        // CartoDB DarkMatter Basemap Tiles
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
            subdomains: "abcd",
            maxZoom: 19,
        }).addTo(state.map);

        loadCorridorNetwork();
    }

    // -------------------------------------------------------------------------
    // 3. Load Road Network Geometry (218 Segments)
    // -------------------------------------------------------------------------
    async function loadCorridorNetwork() {
        try {
            const resp = await fetch("/api/simulation/network");
            if (!resp.ok) throw new Error("Failed to load corridor network");
            const geojson = await resp.json();

            if (state.networkLayer) {
                state.map.removeLayer(state.networkLayer);
            }

            state.networkLayer = L.geoJSON(geojson, {
                style: (feature) => {
                    const frc = feature.properties.frc || 2;
                    if (frc === 1) {
                        return { color: "#4f6880", weight: 6, opacity: 0.85 }; // Elevated Expressway
                    } else if (frc === 2) {
                        return { color: "#36454f", weight: 4, opacity: 0.75 }; // Arterials
                    } else {
                        return { color: "#28343e", weight: 3, opacity: 0.65 }; // Ramps
                    }
                },
                onEachFeature: (feature, layer) => {
                    layer.bindTooltip(
                        `<b>${feature.properties.street_name}</b><br>` +
                        `FRC ${feature.properties.frc} · ${feature.properties.lanes} lanes · ` +
                        `${feature.properties.speed_limit_kmh} km/h`,
                        { sticky: true, className: "road-tooltip" }
                    );
                }
            }).addTo(state.map);

            // Fit map to Barapullah bounds
            state.map.fitBounds(state.networkLayer.getBounds(), { padding: [30, 30] });
        } catch (err) {
            console.error("Error loading network GeoJSON:", err);
        }
    }

    // -------------------------------------------------------------------------
    // 4. Load Scenario Trajectory Data
    // -------------------------------------------------------------------------
    async function loadTrajectories(scenario, period) {
        pauseSimulation();
        state.scenario = scenario;
        state.period = period;
        clearVehicleMarkers();

        // Update UI status
        els.kpiDemandRate.textContent = "Loading TraCI data...";
        
        try {
            const url = `/api/simulation/trajectories?scenario=${encodeURIComponent(scenario)}&period=${encodeURIComponent(period)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}: Trajectory not found`);
            
            const data = await resp.json();
            state.trajectoryData = data;
            state.frames = data.frames || {};
            state.summaryByStep = data.summary_by_step || [];
            state.totalDuration = data.metadata ? data.metadata.duration_sec : 3600;

            els.timeSlider.max = state.totalDuration;
            els.timelineMax.textContent = formatClock(state.totalDuration);

            // Set demand rate label
            const vehCount = data.metadata ? data.metadata.total_unique_vehicles : 200;
            els.kpiDemandRate.textContent = `Demand: ~${vehCount} veh/h`;

            // Reset to current or start
            seekToTime(state.currentTime || 0.0);
            playSimulation();
        } catch (err) {
            console.error("Failed to load trajectory data:", err);
            els.kpiDemandRate.textContent = "Data unavailable";
        }
    }

    // -------------------------------------------------------------------------
    // 5. Vehicle SVG Glyph Generator (Speed-Colored & Oriented)
    // -------------------------------------------------------------------------
    function getSpeedColor(speedKmh) {
        if (speedKmh >= 35.0) return "#00E676"; // High speed (Green)
        if (speedKmh >= 15.0) return "#FFD600"; // Medium speed (Amber)
        return "#FF1744";                       // Low speed / Queued (Red)
    }

    function createVehicleIcon(vtype, speedKmh, headingDeg, isSelected) {
        const speedColor = getSpeedColor(speedKmh);
        const strokeColor = isSelected ? "#ffffff" : "#111418";
        const strokeWidth = isSelected ? "2.5" : "1.2";

        let svgContent = "";
        let iconSize = [20, 20];

        if (vtype === "bus") {
            // City Transit Bus: Longer rectangular body
            iconSize = [28, 28];
            svgContent = `
                <svg width="28" height="28" viewBox="0 0 28 28" class="vehicle-svg">
                    <g transform="rotate(${headingDeg} 14 14)">
                        <!-- Bus Body -->
                        <rect x="9" y="3" width="10" height="22" rx="2.5" fill="${speedColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>
                        <!-- Front Windshield -->
                        <rect x="10.5" y="4.5" width="7" height="3" rx="1" fill="#1e293b"/>
                        <!-- Windows -->
                        <rect x="10" y="9" width="8" height="1.5" fill="#1e293b"/>
                        <rect x="10" y="12" width="8" height="1.5" fill="#1e293b"/>
                        <rect x="10" y="15" width="8" height="1.5" fill="#1e293b"/>
                        <!-- Headlights -->
                        <circle cx="10.5" cy="4" r="0.8" fill="#ffffff"/>
                        <circle cx="17.5" cy="4" r="0.8" fill="#ffffff"/>
                    </g>
                </svg>`;
        } else if (vtype === "auto") {
            // Auto-Rickshaw: Compact three-wheeler silhouette
            iconSize = [20, 20];
            svgContent = `
                <svg width="20" height="20" viewBox="0 0 20 20" class="vehicle-svg">
                    <g transform="rotate(${headingDeg} 10 10)">
                        <!-- Yellow/Green Canopy -->
                        <polygon points="10,3 15,16 5,16" fill="${speedColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>
                        <!-- Front Wheel / Nose -->
                        <circle cx="10" cy="5" r="1.5" fill="#111"/>
                        <circle cx="10" cy="4" r="0.8" fill="#fff"/>
                        <!-- Rear Wheels -->
                        <circle cx="6" cy="15" r="1.2" fill="#111"/>
                        <circle cx="14" cy="15" r="1.2" fill="#111"/>
                    </g>
                </svg>`;
        } else {
            // Passenger Car: Sleek directional sedan
            iconSize = [22, 22];
            svgContent = `
                <svg width="22" height="22" viewBox="0 0 22 22" class="vehicle-svg">
                    <g transform="rotate(${headingDeg} 11 11)">
                        <!-- Car Body -->
                        <rect x="7.5" y="4" width="7" height="14" rx="2" fill="${speedColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>
                        <!-- Windshields -->
                        <rect x="8.5" y="6.5" width="5" height="2.5" rx="0.5" fill="#0f172a"/>
                        <rect x="8.5" y="13" width="5" height="2" rx="0.5" fill="#0f172a"/>
                        <!-- Headlights -->
                        <circle cx="8.5" cy="4.5" r="0.8" fill="#ffffff"/>
                        <circle cx="13.5" cy="4.5" r="0.8" fill="#ffffff"/>
                    </g>
                </svg>`;
        }

        return L.divIcon({
            html: svgContent,
            className: "vehicle-marker-icon",
            iconSize: iconSize,
            iconAnchor: [iconSize[0] / 2, iconSize[1] / 2],
        });
    }

    // -------------------------------------------------------------------------
    // 6. Frame Interpolation & Vehicle Rendering
    // -------------------------------------------------------------------------
    function updateVehiclePositions(simTime) {
        if (!state.frames) return;

        const baseSec = Math.floor(simTime);
        const frac = simTime - baseSec;

        const currentKey = `${baseSec}`;
        const nextKey = `${baseSec + 1}`;

        const currentVehicles = state.frames[currentKey] || [];
        const nextVehicles = state.frames[nextKey] || [];

        // Build index of next frame for smooth interpolation
        const nextMap = new Map();
        for (const v of nextVehicles) {
            nextMap.set(v.vehicle_id, v);
        }

        const activeVidSet = new Set();
        let totalSpeed = 0.0;
        let vehicleCount = 0;

        for (const v of currentVehicles) {
            activeVidSet.add(v.vehicle_id);
            vehicleCount++;
            totalSpeed += v.speed_kmh;

            let lat = v.latitude;
            let lon = v.longitude;
            let heading = v.heading_angle_deg;

            // Interpolate position if next frame exists for this vehicle
            const nextV = nextMap.get(v.vehicle_id);
            if (nextV && frac > 0.0) {
                lat = lat + (nextV.latitude - lat) * frac;
                lon = lon + (nextV.longitude - lon) * frac;
                // Avoid 360 wrap glitch
                let dAngle = nextV.heading_angle_deg - heading;
                if (dAngle > 180) dAngle -= 360;
                if (dAngle < -180) dAngle += 360;
                heading = heading + dAngle * frac;
            }

            const isSelected = (v.vehicle_id === state.selectedVehicleId);
            const icon = createVehicleIcon(v.vehicle_type, v.speed_kmh, heading, isSelected);

            if (state.vehicleMarkers.has(v.vehicle_id)) {
                const marker = state.vehicleMarkers.get(v.vehicle_id);
                marker.setLatLng([lat, lon]);
                marker.setIcon(icon);
                marker.telemetry = v;
            } else {
                const marker = L.marker([lat, lon], { icon: icon, zIndexOffset: 100 });
                marker.telemetry = v;
                marker.on("click", () => inspectVehicle(marker.telemetry));
                marker.addTo(state.map);
                state.vehicleMarkers.set(v.vehicle_id, marker);
            }
        }

        // Remove markers that are no longer active
        for (const [vid, marker] of state.vehicleMarkers.entries()) {
            if (!activeVidSet.has(vid)) {
                state.map.removeLayer(marker);
                state.vehicleMarkers.delete(vid);
            }
        }

        // Update KPIs
        updateKPIs(baseSec, vehicleCount, totalSpeed);

        // Update Inspector if selected vehicle is active
        if (state.selectedVehicleId) {
            const marker = state.vehicleMarkers.get(state.selectedVehicleId);
            if (marker && marker.telemetry) {
                inspectVehicle(marker.telemetry);
            }
        }
    }

    function clearVehicleMarkers() {
        for (const marker of state.vehicleMarkers.values()) {
            state.map.removeLayer(marker);
        }
        state.vehicleMarkers.clear();
    }

    // -------------------------------------------------------------------------
    // 7. KPI Metrics Update
    // -------------------------------------------------------------------------
    function updateKPIs(currentSec, activeCount, totalSpeed) {
        els.kpiSimTime.textContent = formatFullTime(currentSec);
        els.kpiStep.textContent = `Step: ${currentSec}s / ${state.totalDuration}s`;
        els.kpiActiveVehs.textContent = activeCount;

        const meanSpd = activeCount > 0 ? (totalSpeed / activeCount).toFixed(1) : "0.0";
        els.kpiMeanSpeed.textContent = `${meanSpd} km/h`;

        const density = (activeCount / 1.54).toFixed(1);
        els.kpiDensity.textContent = `${density} veh/km`;

        // Completed vehicles from step summary
        if (state.summaryByStep && state.summaryByStep[currentSec]) {
            const sumRow = state.summaryByStep[currentSec];
            els.kpiCompletedVehs.textContent = `Completed: ${sumRow.completed_vehicles || 0}`;
        }

        // Update timeline slider value without triggering change event
        els.timeSlider.value = currentSec;
        els.timelineCurrent.textContent = formatClock(currentSec);
    }

    // -------------------------------------------------------------------------
    // 8. Vehicle Telemetry Inspector
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
    // 9. Playback Engine (60 FPS Animation Loop)
    // -------------------------------------------------------------------------
    function animationLoop(timestamp) {
        if (!state.isPlaying) return;

        if (state.lastAnimTimestamp != null) {
            const dt = (timestamp - state.lastAnimTimestamp) / 1000.0; // delta seconds
            state.currentTime += dt * state.speedMultiplier;

            if (state.currentTime >= state.totalDuration) {
                state.currentTime = state.totalDuration;
                pauseSimulation();
            }

            updateVehiclePositions(state.currentTime);
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
        els.btnPlayPause.style.background = "#f85149"; // Red for pause
        requestAnimationFrame(animationLoop);
    }

    function pauseSimulation() {
        state.isPlaying = false;
        state.lastAnimTimestamp = null;
        els.playIcon.textContent = "▶";
        els.playText.textContent = "Play";
        els.btnPlayPause.style.background = "#3fb950"; // Green for play
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
        updateVehiclePositions(state.currentTime);
    }

    // -------------------------------------------------------------------------
    // 10. Formatters
    // -------------------------------------------------------------------------
    function formatClock(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    function formatFullTime(seconds) {
        // Base on 08:00 AM for Morning Rush
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
    // 11. Event Listeners
    // -------------------------------------------------------------------------
    els.btnPlayPause.addEventListener("click", togglePlayPause);
    
    els.btnRestart.addEventListener("click", () => {
        seekToTime(0.0);
    });

    els.btnStepBack.addEventListener("click", () => {
        seekToTime(state.currentTime - 5.0);
    });

    els.btnStepFwd.addEventListener("click", () => {
        seekToTime(state.currentTime + 5.0);
    });

    els.timeSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        seekToTime(val);
    });

    els.speedBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            els.speedBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            state.speedMultiplier = parseFloat(btn.getAttribute("data-speed"));
        });
    });

    els.scenarioSelect.addEventListener("change", (e) => {
        loadTrajectories(e.target.value, state.period);
    });

    els.periodSelect.addEventListener("change", (e) => {
        loadTrajectories(state.scenario, e.target.value);
    });

    els.closeInspectorBtn.addEventListener("click", closeInspector);

    // Keyboard Shortcuts (Space = Play/Pause, Left/Right = Seek)
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
    // 12. Bootstrap
    // -------------------------------------------------------------------------
    initMap();
    loadTrajectories(els.scenarioSelect.value, els.periodSelect.value);
});
