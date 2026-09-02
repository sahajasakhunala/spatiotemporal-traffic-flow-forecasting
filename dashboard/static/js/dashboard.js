document.addEventListener("DOMContentLoaded", function() {
    loadSegments();
    loadMetrics();
    loadCongestion();

    document.getElementById("prediction-form").addEventListener("submit", function(e) {
        e.preventDefault();
        runPrediction();
    });
});

async function loadSegments() {
    try {
        const res = await fetch("/api/segments");
        const data = await res.json();
        const select = document.getElementById("segment-select");
        select.innerHTML = "";
        
        if (data.segments && data.segments.length > 0) {
            data.segments.forEach(seg => {
                const opt = document.createElement("option");
                opt.value = seg.segment_id;
                opt.textContent = `${seg.street_name} (ID: ${seg.segment_id}, FRC ${seg.frc}, Mean Flow: ${seg.segment_mean_traffic.toFixed(1)})`;
                select.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("Failed to load segments", err);
    }
}

async function loadMetrics() {
    try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        
        const labels = Object.keys(data);
        const maeData = labels.map(l => data[l].test_metrics ? data[l].test_metrics.MAE : 0);
        const rmseData = labels.map(l => data[l].test_metrics ? data[l].test_metrics.RMSE : 0);

        const ctx = document.getElementById("modelComparisonChart").getContext("2d");
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Test MAE (Lower is better)",
                        data: maeData,
                        backgroundColor: "rgba(56, 189, 248, 0.75)",
                        borderColor: "rgba(56, 189, 248, 1)",
                        borderWidth: 1,
                        borderRadius: 4
                    },
                    {
                        label: "Test RMSE (Lower is better)",
                        data: rmseData,
                        backgroundColor: "rgba(244, 63, 94, 0.75)",
                        borderColor: "rgba(244, 63, 94, 1)",
                        borderWidth: 1,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: "#f8fafc", font: { weight: "bold" } }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: "#94a3b8",
                            maxRotation: 20,
                            minRotation: 10,
                            font: { size: 10 }
                        },
                        grid: { color: "rgba(255, 255, 255, 0.05)" }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: "#94a3b8" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" }
                    }
                }
            }
        });
    } catch (err) {
        console.error("Failed to load model metrics", err);
    }
}

async function loadCongestion() {
    try {
        const res = await fetch("/api/congestion");
        const data = await res.json();
        if (data.rush_hour) {
            if (data.rush_hour.morning_rush_hour) {
                document.getElementById("morning-cong").textContent = `${data.rush_hour.morning_rush_hour.congestion_level_percent}%`;
            }
            if (data.rush_hour.evening_rush_hour) {
                document.getElementById("evening-cong").textContent = `${data.rush_hour.evening_rush_hour.congestion_level_percent}%`;
            }
        }
    } catch (err) {
        console.error("Failed to load congestion", err);
    }
}

async function runPrediction() {
    const segmentId = document.getElementById("segment-select").value;
    const date = document.getElementById("pred-date").value;
    const hour = parseInt(document.getElementById("pred-hour").value);
    const model = document.getElementById("model-select").value;
    const lag1 = parseFloat(document.getElementById("lag1-input").value);

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                segment_id: segmentId,
                date: date,
                hour: hour,
                model: model,
                lag_1: lag1,
                lag_2: lag1 * 0.9,
                lag_3: lag1 * 0.8,
                lag_24: lag1 * 1.05
            })
        });

        const data = await res.json();
        if (data.status === "success") {
            const pred = data.prediction;
            document.getElementById("prediction-result").style.display = "block";
            document.getElementById("pred-flow-val").textContent = `${pred.predicted_next_hour_probe_flow} probe units`;
            document.getElementById("pred-comparison").innerHTML = `vs Naive Persistence (t-1: <b>${pred.naive_lag1_baseline}</b>): Delta <b>${pred.delta_vs_lag1 > 0 ? '+' : ''}${pred.delta_vs_lag1}</b> | Model: <i>${pred.model_used}</i>`;
        }
    } catch (err) {
        alert("Prediction failed: " + err);
    }
}
