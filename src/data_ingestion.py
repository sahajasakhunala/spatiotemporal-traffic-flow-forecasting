"""
Data ingestion module.
Converts nested road segment GeoJSON files into a flattened, memory-efficient tabular format.
Processes daily files incrementally and saves them as partitioned Parquet datasets.
"""
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import PROBE_COUNTS_GEOJSON_DIR, PROCESSED_DATA_DIR, TIMESET_TO_HOUR_OFFSET


def extract_date_from_filename(filename: str) -> str:
    """Extract YYYY-MM-DD from filename pattern like new_delhi__2024-08-11_to_2024-08-11_.geojson"""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if not match:
        raise ValueError(f"Could not extract date from filename: {filename}")
    return match.group(1)


def parse_single_geojson(filepath: Path) -> pd.DataFrame:
    """
    Parses a single GeoJSON probe count file into a clean, flat pandas DataFrame.
    
    Nested structure:
      feature -> properties:
        segmentId (int)
        newSegmentId (str)
        streetName (str)
        speedLimit (float/int)
        frc (int)
        distance (float)
        segmentProbeCounts: list of dicts with {timeSet, dateRange, probeCount}
      feature -> geometry:
        LineString coordinates -> compute representative centroid (lon, lat)
        
    Returns:
      pd.DataFrame with schema:
      [segment_id, new_segment_id, street_name, speed_limit, frc, distance,
       date, hour, probe_count, longitude, latitude]
    """
    date_str = extract_date_from_filename(filepath.name)
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    records = []
    
    features = data.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        spc = props.get("segmentProbeCounts", [])
        
        # Skip metadata header features that don't represent road segments
        if not spc or "segmentId" not in props:
            continue
            
        segment_id = props.get("segmentId")
        new_segment_id = props.get("newSegmentId")
        street_name = props.get("streetName") or "Unknown"
        speed_limit = props.get("speedLimit")
        frc = props.get("frc")
        distance = props.get("distance")
        
        # Representative centroid coordinate for spatial analysis
        geom = feat.get("geometry")
        lon, lat = np.nan, np.nan
        if geom and geom.get("type") == "LineString" and geom.get("coordinates"):
            coords = np.array(geom["coordinates"])
            lon = float(np.mean(coords[:, 0]))
            lat = float(np.mean(coords[:, 1]))
            
        for entry in spc:
            time_set = entry.get("timeSet")
            probe_count = entry.get("probeCount", 0)
            
            # Map timeSet to 0-23 hour
            if time_set is not None:
                hour = int(time_set) - TIMESET_TO_HOUR_OFFSET
                records.append({
                    "segment_id": segment_id,
                    "new_segment_id": new_segment_id,
                    "street_name": street_name,
                    "speed_limit": speed_limit,
                    "frc": frc,
                    "distance": distance,
                    "date": date_str,
                    "hour": hour,
                    "probe_count": probe_count,
                    "longitude": lon,
                    "latitude": lat
                })
                
    df = pd.DataFrame(records)
    
    # Enforce correct data types for optimal memory usage
    if not df.empty:
        df["segment_id"] = df["segment_id"].astype("int64")
        df["new_segment_id"] = df["new_segment_id"].astype("string")
        df["street_name"] = df["street_name"].astype("string")
        df["speed_limit"] = df["speed_limit"].astype("float32")
        df["frc"] = df["frc"].astype("int8")
        df["distance"] = df["distance"].astype("float32")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["hour"] = df["hour"].astype("int8")
        df["probe_count"] = df["probe_count"].astype("int32")
        df["longitude"] = df["longitude"].astype("float32")
        df["latitude"] = df["latitude"].astype("float32")
        
    return df


def save_daily_partition(df: pd.DataFrame, date_str: str, output_dir: Path) -> Path:
    """Saves a daily DataFrame to a partitioned Parquet directory."""
    partition_dir = output_dir / f"date={date_str}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    file_path = partition_dir / "data.parquet"
    df.to_parquet(file_path, index=False, engine="pyarrow", compression="snappy")
    return file_path


def process_all_geojson_files(raw_dir: Path = PROBE_COUNTS_GEOJSON_DIR,
                              output_dir: Path = PROCESSED_DATA_DIR) -> List[Path]:
    """
    Incrementally processes all daily GeoJSON files and stores them as partitioned Parquet.
    """
    raw_files = sorted(list(raw_dir.glob("*.geojson")))
    print(f"Found {len(raw_files)} GeoJSON files in {raw_dir}")
    
    processed_files = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(raw_files, start=1):
        date_str = extract_date_from_filename(file_path.name)
        print(f"[{i}/{len(raw_files)}] Ingesting {file_path.name} (Date: {date_str})...")
        
        df = parse_single_geojson(file_path)
        print(f"  -> Extracted {len(df):,} rows for {df['segment_id'].nunique():,} unique road segments.")
        
        saved_path = save_daily_partition(df, date_str, output_dir)
        processed_files.append(saved_path)
        print(f"  -> Saved partition to {saved_path}")
        
    print(f"Completed ingestion for all {len(processed_files)} dates.")
    return processed_files


if __name__ == "__main__":
    import sys
    # Verify single file parsing
    test_files = sorted(list(PROBE_COUNTS_GEOJSON_DIR.glob("*.geojson")))
    if not test_files:
        print("Error: No GeoJSON files found in configured raw directory!")
        sys.exit(1)
        
    print("Testing single GeoJSON ingestion on:", test_files[0].name)
    df_sample = parse_single_geojson(test_files[0])
    print("Sample DataFrame Info:")
    print(df_sample.info())
    print("\nHead (5 rows):")
    print(df_sample.head())
