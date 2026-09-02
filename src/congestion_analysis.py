"""
Congestion & Macro Traffic Pattern Analytics Module.
Ingests and analyzes contextual urban benchmarks from:
- global_metrics/
- weekday_stats/

Contextual grounding:
- Rush hour speed degradation (morning 23.6 km/h, evening 19.9 km/h)
- Congestion intensity percent (morning 43%, evening 69%)
- Weekday-by-weekday congestion and speed heatmaps
- 2024 vs 2023 YoY mobility comparisons
"""
from pathlib import Path
import json
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import GLOBAL_METRICS_DIR, WEEKDAY_STATS_DIR, VISUALIZATIONS_DIR


def parse_percentage(val: Any) -> float:
    """Safely converts string percentage '43%' to float 43.0."""
    if isinstance(val, str):
        val = val.replace("%", "").strip()
        return float(val)
    return float(val) if pd.notnull(val) else np.nan


def parse_speed(val: Any) -> float:
    """Safely converts '35 km/h' to float 35.0."""
    if isinstance(val, str):
        val = val.replace("km/h", "").strip()
        return float(val)
    return float(val) if pd.notnull(val) else np.nan


def run_congestion_analysis(global_dir: Path = GLOBAL_METRICS_DIR,
                            weekday_dir: Path = WEEKDAY_STATS_DIR,
                            output_dir: Path = VISUALIZATIONS_DIR) -> Dict[str, Any]:
    """Extracts macro statistics and generates analytical heatmaps and comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    
    # 1. Load city and urban rush hour metrics
    city_rush_path = global_dir / "2024_city_rush_hour.json"
    if city_rush_path.exists():
        with open(city_rush_path, "r") as f:
            report["city_rush_hour"] = json.load(f)
            
    urban_rush_path = global_dir / "2024_urban_rush_hour.json"
    if urban_rush_path.exists():
        with open(urban_rush_path, "r") as f:
            report["urban_rush_hour"] = json.load(f)
            
    # 2. Weekday Congestion Heatmap
    cong_csv_path = weekday_dir / "2024_week_day_congestion_city.csv"
    if cong_csv_path.exists():
        cong_df = pd.read_csv(cong_csv_path)
        days = [c for c in cong_df.columns if c != "Time"]
        for col in days:
            cong_df[col] = cong_df[col].apply(parse_percentage)
            
        cong_heatmap = cong_df.set_index("Time")[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cong_heatmap, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={'label': 'Congestion Level (%)'}, ax=ax)
        ax.set_title("14. New Delhi Diurnal Weekday Congestion Heatmap (2024)")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Time of Day")
        plt.tight_layout()
        plt.savefig(output_dir / "14_weekday_congestion_heatmap.png", dpi=300)
        plt.close()
        report["weekday_congestion"] = cong_heatmap.to_dict()
        
    # 3. Weekday Speed Heatmap
    speed_csv_path = weekday_dir / "2024_week_day_speed_city.csv"
    if speed_csv_path.exists():
        speed_df = pd.read_csv(speed_csv_path)
        days = [c for c in speed_df.columns if c != "Time"]
        for col in days:
            speed_df[col] = speed_df[col].apply(parse_speed)
            
        speed_heatmap = speed_df.set_index("Time")[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(speed_heatmap, annot=True, fmt=".1f", cmap="coolwarm_r", cbar_kws={'label': 'Average Speed (km/h)'}, ax=ax)
        ax.set_title("15. New Delhi Hourly Weekday Speed Heatmap (2024)")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Time of Day")
        plt.tight_layout()
        plt.savefig(output_dir / "15_weekday_speed_heatmap.png", dpi=300)
        plt.close()
        report["weekday_speed"] = speed_heatmap.to_dict()
        
    # 4. Monthly Congestion Trend 2024 vs 2023
    city_traffic_path = global_dir / "new_delhi_2024_city_traffic.json"
    if city_traffic_path.exists():
        with open(city_traffic_path, "r") as f:
            city_traffic = json.load(f)
            monthly = city_traffic.get("monthly_congestion_level", [])
            if monthly:
                m_df = pd.DataFrame(monthly)
                fig, ax = plt.subplots(figsize=(11, 5))
                x = np.arange(len(m_df))
                width = 0.35
                ax.bar(x - width/2, m_df["2023"], width, label="2023", color="#a6cee3")
                ax.bar(x + width/2, m_df["2024"], width, label="2024", color="#1f78b4")
                ax.set_title("16. Monthly Congestion Level Benchmark: 2024 vs 2023")
                ax.set_xlabel("Month")
                ax.set_ylabel("Congestion Level (%)")
                ax.set_xticks(x)
                ax.set_xticklabels(m_df["month"])
                ax.legend()
                plt.tight_layout()
                plt.savefig(output_dir / "16_monthly_congestion_comparison.png", dpi=300)
                plt.close()
                report["monthly_trends"] = monthly
                
    print(f"Congestion analysis completed. Visualizations saved to {output_dir}")
    return report
