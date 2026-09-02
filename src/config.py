"""
Central configuration module for Intelligent Urban Traffic Flow & Congestion Prediction System.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = Path(r"C:\Users\LENOVO\Downloads\archive_traffic\new_delhi_traffic_dataset")
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
DOCS_DIR = BASE_DIR / "docs"

# Probe count files directory
PROBE_COUNTS_GEOJSON_DIR = RAW_DATA_DIR / "probe_counts" / "geojson"
GLOBAL_METRICS_DIR = RAW_DATA_DIR / "global_metrics"
WEEKDAY_STATS_DIR = RAW_DATA_DIR / "weekday_stats"
FACILITY_DIR = RAW_DATA_DIR / "facility"

# Time mapping: timeSet 2 -> hour 0, timeSet 25 -> hour 23
TIMESET_TO_HOUR_OFFSET = 2

# Functional Road Class (FRC) labels
FRC_MAP = {
    1: "Motorway / Major Highway",
    2: "Major Arterial",
    3: "Secondary Arterial",
    4: "Collector Road",
    5: "Local Connecting Road",
    6: "Local Residential Street"
}

# Rush hours definition for New Delhi
MORNING_RUSH_HOURS = [8, 9, 10]
EVENING_RUSH_HOURS = [17, 18, 19, 20]

# Chronological split configuration
TRAIN_START_DATE = "2024-08-11"
TRAIN_END_DATE = "2024-08-26"
TEST_START_DATE = "2024-08-27"
TEST_END_DATE = "2024-08-30"

# Cultural / Festival dates in August 2024
SPECIAL_DATES = {
    "2024-08-15": "Independence Day",
    "2024-08-19": "Rakshabandhan",
    "2024-08-26": "Janmashtami"
}
