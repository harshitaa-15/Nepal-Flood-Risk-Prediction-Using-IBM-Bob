"""
utils.py
--------
Shared utility functions used by app.py and other modules.
"""

import os
import json
import pandas as pd
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Dataset statistics (cached) ─────────────────────────────────────────────

def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return a dict of summary statistics for the dashboard."""
    return {
        "total_records":     len(df),
        "locations":         df["location"].nunique(),
        "rivers":            df["river"].nunique(),
        "date_range_start":  str(df["date"].min().date()),
        "date_range_end":    str(df["date"].max().date()),
        "avg_precipitation": round(df["precipitation_mm"].mean(), 2),
        "max_discharge":     round(df["river_discharge_m3s"].max(), 2),
        "high_risk_pct":     round((df["flood_risk"] == 2).mean() * 100, 1),
        "medium_risk_pct":   round((df["flood_risk"] == 1).mean() * 100, 1),
        "low_risk_pct":      round((df["flood_risk"] == 0).mean() * 100, 1),
    }


# ── Model artifacts ──────────────────────────────────────────────────────────

def load_metrics() -> dict:
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_feature_importance() -> dict:
    path = os.path.join(MODELS_DIR, "feature_importance.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_best_model_name() -> str:
    path = os.path.join(MODELS_DIR, "best_model_name.txt")
    if not os.path.exists(path):
        return "Unknown"
    with open(path) as f:
        return f.read().strip()


def models_trained() -> bool:
    """Check whether training artifacts exist."""
    return os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))


# ── Human-readable feature names ────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    "precipitation_mm":              "Precipitation (mm)",
    "rain_mm":                       "Rainfall (mm)",
    "soil_moisture_0_100cm_m3m3":    "Soil Moisture",
    "temperature_mean_c":            "Temperature (°C)",
    "dew_point_mean_c":              "Dew Point (°C)",
    "precipitation_hours":           "Precipitation Hours",
    "relative_humidity_mean_pct":    "Relative Humidity (%)",
    "wind_speed_max_kmh":            "Max Wind Speed (km/h)",
    "wind_gusts_max_kmh":            "Max Wind Gusts (km/h)",
    "wind_direction_dominant_deg":   "Wind Direction (°)",
    "elevation_m":                   "Elevation (m)",
    "month_sin":                     "Month (sin)",
    "month_cos":                     "Month (cos)",
    "day_of_year":                   "Day of Year",
    "precipitation_mm_roll3_mean":   "Precip 3-day Avg",
    "precipitation_mm_roll7_mean":   "Precip 7-day Avg",
    "precipitation_mm_roll3_sum":    "Precip 3-day Total",
    "precipitation_mm_roll7_sum":    "Precip 7-day Total",
    "rain_mm_roll3_mean":            "Rain 3-day Avg",
    "rain_mm_roll7_mean":            "Rain 7-day Avg",
    "rain_mm_roll3_sum":             "Rain 3-day Total",
    "rain_mm_roll7_sum":             "Rain 7-day Total",
    "relative_humidity_mean_pct_roll3_mean": "Humidity 3-day Avg",
    "relative_humidity_mean_pct_roll7_mean": "Humidity 7-day Avg",
    "relative_humidity_mean_pct_roll3_sum":  "Humidity 3-day Sum",
    "relative_humidity_mean_pct_roll7_sum":  "Humidity 7-day Sum",
    "precip_roll7_max":              "Precip 7-day Max",
    "saturation_index":              "Saturation Index",
    "heavy_rain_flag":               "Heavy Rain Flag",
    "temp_dew_spread":               "Temp–Dew Spread (°C)",
    "location_enc":                  "Location (encoded)",
}


def pretty_feature(name: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


# ── Number formatting ────────────────────────────────────────────────────────

def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def fmt_num(val: float, decimals: int = 2) -> str:
    return f"{val:,.{decimals}f}"


# ── Season helper ────────────────────────────────────────────────────────────

def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"
