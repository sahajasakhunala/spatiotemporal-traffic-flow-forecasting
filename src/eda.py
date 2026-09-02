"""
EDA Module for Intelligent Urban Traffic Flow & Congestion Prediction System.
Produces 10 rigorous, high-resolution visual analyses:
1. Traffic flow proxy distribution (log scale + boxplot)
2. Hourly traffic flow profile (24-hour diurnal cycle)
3. Traffic flow by day of the week
4. Weekday vs Weekend pattern comparison
5. Rush hour vs Off-peak traffic intensity
6. Road segment traffic variation (FRC functional classes)
7. Time-series trend across the 20 days (highlighting festivals)
8. Speed limit vs Traffic volume relationships
9. Spatial/Geographic traffic density distribution
10. Festival anomaly analysis (Independence Day, Rakshabandhan, Janmashtami)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import VISUALIZATIONS_DIR, FRC_MAP, SPECIAL_DATES

# Set aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


def run_comprehensive_eda(df: pd.DataFrame, output_dir: Path = VISUALIZATIONS_DIR):
    """Executes the full 10-point EDA analysis and exports figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating comprehensive EDA visualizations...")
    
    # Pre-calculate auxiliary columns if missing
    if "day_of_week" not in df.columns:
        dt = pd.to_datetime(df["date"])
        df["day_of_week"] = dt.dt.day_name()
        df["day_of_week_num"] = dt.dt.dayofweek
        df["is_weekend"] = df["day_of_week_num"].isin([5, 6])
        df["is_rush_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19, 20])
    
    # 1. Traffic flow distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["probe_count"], bins=60, kde=True, ax=axes[0], color="#1f77b4", log_scale=(False, True))
    axes[0].set_title("1. Traffic Flow Distribution (Log-Scale Counts)")
    axes[0].set_xlabel("Hourly Probe Count (Traffic Flow Proxy)")
    axes[0].set_ylabel("Log Frequency")
    
    sns.boxplot(x=df["probe_count"], ax=axes[1], color="#ff7f0e", fliersize=2)
    axes[1].set_title("Traffic Flow Boxplot & Outlier Spread")
    axes[1].set_xlabel("Probe Count")
    plt.tight_layout()
    plt.savefig(output_dir / "01_traffic_flow_distribution.png", dpi=300)
    plt.close()
    
    # 2. Hourly Diurnal Profile
    hourly_stats = df.groupby("hour")["probe_count"].agg(["mean", "median", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_stats["hour"], hourly_stats["mean"], marker="o", color="#2ca02c", linewidth=2.5, label="Mean Probe Count")
    ax.plot(hourly_stats["hour"], hourly_stats["median"], marker="s", linestyle="--", color="#d62728", label="Median Probe Count")
    ax.fill_between(hourly_stats["hour"], hourly_stats["mean"] - hourly_stats["std"]/2, hourly_stats["mean"] + hourly_stats["std"]/2, color="#2ca02c", alpha=0.15, label="0.5 Std Dev")
    ax.set_title("2. Diurnal Traffic Cycle across Delhi NCR (24-Hour Profile)")
    ax.set_xlabel("Hour of Day (0:00 - 23:00)")
    ax.set_ylabel("Average Probe Count per Segment")
    ax.set_xticks(range(0, 24))
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "02_hourly_traffic_profile.png", dpi=300)
    plt.close()
    
    # 3. Day of Week Profile
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = df.groupby("day_of_week")["probe_count"].mean().reindex(days_order).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=dow_stats, x="day_of_week", y="probe_count", palette="Blues_d", ax=ax)
    ax.set_title("3. Average Traffic Flow by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Probe Count")
    plt.tight_layout()
    plt.savefig(output_dir / "03_day_of_week_traffic.png", dpi=300)
    plt.close()
    
    # 4. Weekday vs Weekend Hourly Profile
    w_hourly = df.groupby(["hour", "is_weekend"])["probe_count"].mean().reset_index()
    w_hourly["Day Type"] = w_hourly["is_weekend"].map({False: "Weekday (Mon-Fri)", True: "Weekend (Sat-Sun)"})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=w_hourly, x="hour", y="probe_count", hue="Day Type", marker="o", palette=["#1f77b4", "#e377c2"], ax=ax, linewidth=2.5)
    ax.set_title("4. Weekday vs Weekend Traffic Flow Dynamics")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Probe Count")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(output_dir / "04_weekday_vs_weekend.png", dpi=300)
    plt.close()
    
    # 5. Rush Hour vs Off-Peak
    rush_stats = df.groupby("is_rush_hour")["probe_count"].mean().reset_index()
    rush_stats["Period"] = rush_stats["is_rush_hour"].map({False: "Off-Peak Hours", True: "Rush Hours (8-10 AM, 5-8 PM)"})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=rush_stats, x="Period", y="probe_count", palette=["#7f7f7f", "#d62728"], ax=ax)
    ax.set_title("5. Peak Rush Hours vs Off-Peak Traffic Intensity")
    ax.set_ylabel("Mean Probe Count")
    plt.tight_layout()
    plt.savefig(output_dir / "05_rush_vs_offpeak.png", dpi=300)
    plt.close()
    
    # 6. Functional Road Class (FRC) Variation
    df["frc_name"] = df["frc"].map(FRC_MAP).fillna("Other")
    frc_stats = df.groupby("frc_name")["probe_count"].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frc_stats, x="probe_count", y="frc_name", palette="viridis", ax=ax)
    ax.set_title("6. Traffic Flow Proxy Variation across Functional Road Classes (FRC)")
    ax.set_xlabel("Mean Probe Count")
    ax.set_ylabel("Road Classification")
    plt.tight_layout()
    plt.savefig(output_dir / "06_frc_traffic_variation.png", dpi=300)
    plt.close()
    
    # 7. Date Trend & Festival Impact
    date_stats = df.groupby("date")["probe_count"].mean().reset_index()
    date_stats["date_str"] = date_stats["date"].astype(str)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(date_stats["date_str"], date_stats["probe_count"], marker="o", color="#8c564b", linewidth=2)
    ax.set_title("7. Daily Traffic Flow Trend across Delhi NCR (August 1130, 2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Daily Probe Count")
    plt.xticks(rotation=45)
    
    # Annotate festivals
    for date_k, name in SPECIAL_DATES.items():
        if date_k in date_stats["date_str"].values:
            val = date_stats.loc[date_stats["date_str"] == date_k, "probe_count"].values[0]
            ax.annotate(f"{name}\n({date_k[5:]})", xy=(date_k, val), xytext=(date_k, val + 0.3),
                        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                        fontsize=9, fontweight="bold", ha="center")
    plt.tight_layout()
    plt.savefig(output_dir / "07_date_trend_and_festivals.png", dpi=300)
    plt.close()
    
    # 8. Speed Limit vs Traffic Volume
    df_sl = df[df["speed_limit"].notnull() & (df["speed_limit"] > 0)]
    sl_stats = df_sl.groupby("speed_limit")["probe_count"].mean().reset_index()
    sl_stats = sl_stats[sl_stats["speed_limit"].isin([20, 30, 40, 45, 50, 60, 70, 80])]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=sl_stats, x="speed_limit", y="probe_count", palette="mako", ax=ax)
    ax.set_title("8. Traffic Volume by Posted Speed Limit (km/h)")
    ax.set_xlabel("Speed Limit (km/h)")
    ax.set_ylabel("Mean Probe Count")
    plt.tight_layout()
    plt.savefig(output_dir / "08_speed_limit_vs_traffic.png", dpi=300)
    plt.close()
    
    # 9. Spatial Density (Sample of Centroids)
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    sample_valid = sample_df[sample_df["longitude"].notnull() & sample_df["latitude"].notnull()]
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(sample_valid["longitude"], sample_valid["latitude"], c=sample_valid["probe_count"],
                    cmap="inferno", s=3, alpha=0.6, vmax=sample_valid["probe_count"].quantile(0.95))
    plt.colorbar(sc, ax=ax, label="Probe Count (p95 capped)")
    ax.set_title("9. Spatial Traffic Density Map (Delhi NCR Probe Centroids)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_dir / "09_spatial_density_map.png", dpi=300)
    plt.close()
    
    # 10. Festival vs Normal Days Traffic Comparison
    df["day_category"] = df["date"].astype(str).map(lambda d: SPECIAL_DATES.get(d, "Regular Day"))
    fest_stats = df.groupby(["hour", "day_category"])["probe_count"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=fest_stats, x="hour", y="probe_count", hue="day_category", marker="o", ax=ax)
    ax.set_title("10. Cultural Event & Festival Traffic Anomaly Comparison")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Probe Count")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(output_dir / "10_festival_traffic_anomalies.png", dpi=300)
    plt.close()
    
    print(f"Successfully generated all 10 EDA visualizations in {output_dir}")
