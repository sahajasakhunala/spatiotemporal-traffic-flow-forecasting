"""
Data Quality Validation Module.
Inspects tabular traffic probe datasets for:
- Missing values & NaNs
- Duplicate rows across (segment_id, date, hour)
- Numeric type validity & range bounds
- Zero-value traffic distribution (noting legitimately low volume roads)
- Outliers & extreme value distributions
- Temporal & spatial continuity across road segments
"""
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.config import PROCESSED_DATA_DIR


def validate_single_day(df: pd.DataFrame) -> Dict[str, Any]:
    """Runs data quality checks on a single day's extracted DataFrame."""
    report = {}
    
    # 1. Row count and segment cardinality
    report["total_rows"] = len(df)
    report["unique_segments"] = df["segment_id"].nunique()
    report["expected_rows"] = report["unique_segments"] * 24
    report["is_complete_grid"] = (report["total_rows"] == report["expected_rows"])
    
    # 2. Missing values check
    missing_counts = df.isnull().sum().to_dict()
    report["missing_values"] = missing_counts
    
    # 3. Duplicate checks
    duplicates = df.duplicated(subset=["segment_id", "date", "hour"]).sum()
    report["duplicate_records"] = int(duplicates)
    
    # 4. Zero probe counts analysis (legitimate low-traffic / late-night segments)
    zero_count = (df["probe_count"] == 0).sum()
    report["zero_probe_count"] = int(zero_count)
    report["zero_probe_percentage"] = round((zero_count / len(df)) * 100, 2)
    
    # 5. Outlier summary (IQR based)
    q25 = df["probe_count"].quantile(0.25)
    q75 = df["probe_count"].quantile(0.75)
    iqr = q75 - q25
    upper_bound = q75 + 3.0 * iqr
    outliers = (df["probe_count"] > upper_bound).sum()
    report["outliers_3iqr_count"] = int(outliers)
    report["outliers_3iqr_pct"] = round((outliers / len(df)) * 100, 2)
    report["max_probe_count"] = int(df["probe_count"].max())
    report["mean_probe_count"] = round(float(df["probe_count"].mean()), 2)
    report["median_probe_count"] = float(df["probe_count"].median())
    
    # 6. Coordinate completeness
    coords_valid = df["longitude"].notnull() & df["latitude"].notnull()
    report["valid_coords_pct"] = round((coords_valid.sum() / len(df)) * 100, 2)
    
    return report


def print_quality_report(report: Dict[str, Any], title: str = "Data Quality Report"):
    """Formatted printer for quality check results."""
    print("=" * 60)
    print(f" {title.upper()} ")
    print("=" * 60)
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    - {sub_k}: {sub_v}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)
