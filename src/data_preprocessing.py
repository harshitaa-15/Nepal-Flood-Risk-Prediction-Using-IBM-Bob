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

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_FILE = os.path.join(DATA_DIR, "CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv")

# ── Flood-risk thresholds (derived from dataset percentiles) ───────────────
# Using per-location percentiles so that different river scales
# are treated fairly.  Thresholds are computed on TRAINING rows only;
# for inference we rely on saved percentile tables.
FLOOD_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
FLOOD_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}


# ── Public helpers ─────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Return the raw CSV as a DataFrame."""
    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_location_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with per-location discharge thresholds:
        p75  -> LOW/MEDIUM boundary
        p90  -> MEDIUM/HIGH boundary
    """
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
    """
    Assign integer flood-risk label (0=LOW, 1=MEDIUM, 2=HIGH) based on
    per-location discharge thresholds.
    """
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


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    clean_df : processed DataFrame with flood_risk label
    thresholds : per-location discharge thresholds (for inference)
    """
    df = df.copy()

    # ── 1. Remove duplicates ──────────────────────────────────────────────
    df.drop_duplicates(inplace=True)

    # ── 2. Sort by location + date ────────────────────────────────────────
    df.sort_values(["location", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 3. Date/time features ─────────────────────────────────────────────
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    # Cyclical encoding for month (captures seasonality)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    # ── 4. Missing-value handling (none present but kept for robustness) ──
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── 5. Outlier capping (IQR × 3 per numerical column) ─────────────────
    # Use a wide multiplier so extreme-but-real flood events are preserved.
    for col in ["precipitation_mm", "rain_mm", "wind_speed_max_kmh",
                "wind_gusts_max_kmh"]:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q3)

    # ── 6. Flood-risk labels from per-location thresholds ─────────────────
    thresholds = compute_location_thresholds(df)
    df = assign_flood_risk(df, thresholds)

    return df, thresholds


def get_feature_columns() -> list[str]:
    """
    Return the list of feature columns used for modelling.
    river_discharge_m3s is the source of the label, so it is excluded
    from features to prevent data leakage.
    """
    return [
        "precipitation_mm",
        "rain_mm",
        "soil_moisture_0_100cm_m3m3",
        "temperature_mean_c",
        "dew_point_mean_c",
        "precipitation_hours",
        "relative_humidity_mean_pct",
        "wind_speed_max_kmh",
        "wind_gusts_max_kmh",
        "wind_direction_dominant_deg",
        "elevation_m",
        "month_sin",
        "month_cos",
        "day_of_year",
    ]


def get_display_feature_meta() -> dict:
    """Human-readable metadata for Streamlit input widgets."""
    return {
        "precipitation_mm": {
            "label": "Precipitation (mm)", "min": 0.0, "max": 150.0,
            "default": 5.0, "step": 0.1,
        },
        "rain_mm": {
            "label": "Rainfall (mm)", "min": 0.0, "max": 150.0,
            "default": 5.0, "step": 0.1,
        },
        "soil_moisture_0_100cm_m3m3": {
            "label": "Soil Moisture (m³/m³)", "min": 0.1, "max": 0.6,
            "default": 0.32, "step": 0.001,
        },
        "temperature_mean_c": {
            "label": "Mean Temperature (°C)", "min": -10.0, "max": 40.0,
            "default": 22.0, "step": 0.1,
        },
        "dew_point_mean_c": {
            "label": "Dew Point (°C)", "min": -20.0, "max": 30.0,
            "default": 15.0, "step": 0.1,
        },
        "precipitation_hours": {
            "label": "Precipitation Hours", "min": 0, "max": 24,
            "default": 3, "step": 1,
        },
        "relative_humidity_mean_pct": {
            "label": "Relative Humidity (%)", "min": 0, "max": 100,
            "default": 70, "step": 1,
        },
        "wind_speed_max_kmh": {
            "label": "Max Wind Speed (km/h)", "min": 0.0, "max": 120.0,
            "default": 15.0, "step": 0.1,
        },
        "wind_gusts_max_kmh": {
            "label": "Max Wind Gusts (km/h)", "min": 0.0, "max": 180.0,
            "default": 30.0, "step": 0.1,
        },
        "wind_direction_dominant_deg": {
            "label": "Wind Direction (°)", "min": 0, "max": 360,
            "default": 180, "step": 1,
        },
        "elevation_m": {
            "label": "Elevation (m)", "min": 50, "max": 5000,
            "default": 500, "step": 10,
        },
    }


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")
    clean, thresh = preprocess(raw)
    print(f"Processed shape: {clean.shape}")
    print("Flood risk distribution:")
    print(clean["flood_risk"].value_counts().sort_index())
    print("\nPer-location thresholds:")
    print(thresh)
