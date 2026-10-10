"""
data_preprocessing.py
---------------------
Loads and cleans the Nepal Flood & Weather dataset.
Derives flood-risk labels using per-location discharge percentiles
(p90 = HIGH, p75-p90 = MEDIUM, <p75 = LOW) to avoid data leakage.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ── Robust path resolution ─────────────────────────────────────────────────
# Works on: local Windows, local Linux, Streamlit Cloud (/mount/src/...)
# Repo root = parent of the src/ folder that contains this file
_REPO_ROOT = Path(__file__).resolve().parent.parent

_CSV_NAME = "CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv"

# Search every possible folder name the CSV might be in
_CANDIDATE_DIRS = ["data", "Data", "Dataset", "dataset", "datasets", "."]
CSV_FILE = None
for _d in _CANDIDATE_DIRS:
    _candidate = _REPO_ROOT / _d / _CSV_NAME
    if _candidate.exists():
        CSV_FILE = _candidate
        break

# Fallback: scan the whole repo root for the file
if CSV_FILE is None:
    for _p in _REPO_ROOT.rglob(_CSV_NAME):
        CSV_FILE = _p
        break

if CSV_FILE is None:
    raise FileNotFoundError(
        f"Cannot find {_CSV_NAME}. "
        f"Searched under: {_REPO_ROOT}. "
        f"Please place the CSV in the repo root or a subfolder."
    )

DATA_DIR = CSV_FILE.parent

# ── Flood-risk thresholds (derived from dataset percentiles) ───────────────
FLOOD_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
FLOOD_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}


# ── Public helpers ─────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Return the raw CSV as a DataFrame."""
    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_location_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = (
        df.groupby("location")["river_discharge_m3s"]
        .agg(
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    return thresholds


def assign_flood_risk(df: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(thresholds, on="location", how="left")
    discharge = df["river_discharge_m3s"]
    conditions = [
        discharge >= df["p90"],
        (discharge >= df["p75"]) & (discharge < df["p90"]),
    ]
    choices = [2, 1]
    df["flood_risk"] = np.select(conditions, choices, default=0)
    df.drop(columns=["p75", "p90"], inplace=True)
    return df


def preprocess(df: pd.DataFrame) -> tuple:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.sort_values(["location", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    for col in ["precipitation_mm", "rain_mm", "wind_speed_max_kmh", "wind_gusts_max_kmh"]:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q3)

    thresholds = compute_location_thresholds(df)
    df = assign_flood_risk(df, thresholds)

    return df, thresholds


def get_feature_columns() -> list:
    return [
        "precipitation_mm", "rain_mm", "soil_moisture_0_100cm_m3m3",
        "temperature_mean_c", "dew_point_mean_c", "precipitation_hours",
        "relative_humidity_mean_pct", "wind_speed_max_kmh", "wind_gusts_max_kmh",
        "wind_direction_dominant_deg", "elevation_m", "month_sin", "month_cos", "day_of_year",
    ]


def get_display_feature_meta() -> dict:
    return {
        "precipitation_mm":            {"label": "Precipitation (mm)",    "min": 0.0,  "max": 150.0, "default": 5.0,  "step": 0.1},
        "rain_mm":                     {"label": "Rainfall (mm)",          "min": 0.0,  "max": 150.0, "default": 5.0,  "step": 0.1},
        "soil_moisture_0_100cm_m3m3":  {"label": "Soil Moisture (m³/m³)", "min": 0.1,  "max": 0.6,   "default": 0.32, "step": 0.001},
        "temperature_mean_c":          {"label": "Mean Temperature (°C)",  "min": -10.0,"max": 40.0,  "default": 22.0, "step": 0.1},
        "dew_point_mean_c":            {"label": "Dew Point (°C)",         "min": -20.0,"max": 30.0,  "default": 15.0, "step": 0.1},
        "precipitation_hours":         {"label": "Precipitation Hours",    "min": 0,    "max": 24,    "default": 3,    "step": 1},
        "relative_humidity_mean_pct":  {"label": "Relative Humidity (%)",  "min": 0,    "max": 100,   "default": 70,   "step": 1},
        "wind_speed_max_kmh":          {"label": "Max Wind Speed (km/h)",  "min": 0.0,  "max": 120.0, "default": 15.0, "step": 0.1},
        "wind_gusts_max_kmh":          {"label": "Max Wind Gusts (km/h)",  "min": 0.0,  "max": 180.0, "default": 30.0, "step": 0.1},
        "wind_direction_dominant_deg": {"label": "Wind Direction (°)",     "min": 0,    "max": 360,   "default": 180,  "step": 1},
        "elevation_m":                 {"label": "Elevation (m)",           "min": 50,   "max": 5000,  "default": 500,  "step": 10},
    }
