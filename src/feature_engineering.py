"""
feature_engineering.py
----------------------
Adds domain-relevant features for flood-risk modelling.
All transformations are leakage-free (use only current/past values).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling-window statistics per location.
    Windows: 3-day, 7-day.
    Only precipitation and rain are aggregated (most predictive for floods).
    """
    df = df.copy()
    df = df.sort_values(["location", "date"])

    for window in [3, 7]:
        for col in ["precipitation_mm", "rain_mm", "relative_humidity_mean_pct"]:
            df[f"{col}_roll{window}_mean"] = (
                df.groupby("location")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            df[f"{col}_roll{window}_sum"] = (
                df.groupby("location")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
            )

    # 7-day max precipitation (antecedent wet spell)
    df["precip_roll7_max"] = (
        df.groupby("location")["precipitation_mm"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())
    )

    # Soil saturation proxy: soil moisture × relative humidity
    df["saturation_index"] = (
        df["soil_moisture_0_100cm_m3m3"] * df["relative_humidity_mean_pct"] / 100
    )

    # High-precipitation flag (>20 mm = heavy rain threshold)
    df["heavy_rain_flag"] = (df["precipitation_mm"] > 20).astype(int)

    # Temperature-dew point spread (low spread → high humidity / fog / rain)
    df["temp_dew_spread"] = df["temperature_mean_c"] - df["dew_point_mean_c"]

    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True,
                         encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode location, river, basin.

    Parameters
    ----------
    fit : bool
        True during training; False during inference (use saved encoders).
    encoders : dict | None
        Pre-fitted LabelEncoders when fit=False.

    Returns
    -------
    df : DataFrame with encoded columns
    encoders : fitted LabelEncoder dict
    """
    if encoders is None:
        encoders = {}

    for col in ["location", "river", "basin"]:
        if fit:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[f"{col}_enc"] = le.transform(df[col].astype(str))

    return df, encoders


def get_all_feature_columns() -> list[str]:
    """
    Full feature set including engineered features.
    """
    base = [
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
    rolling = []
    for window in [3, 7]:
        for col in ["precipitation_mm", "rain_mm", "relative_humidity_mean_pct"]:
            rolling.append(f"{col}_roll{window}_mean")
            rolling.append(f"{col}_roll{window}_sum")
    rolling.append("precip_roll7_max")

    engineered = [
        "saturation_index",
        "heavy_rain_flag",
        "temp_dew_spread",
        "location_enc",
    ]

    return base + rolling + engineered


def prepare_features(df: pd.DataFrame, fit: bool = True,
                     encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Full feature-engineering pipeline:
      1. Rolling features
      2. Categorical encoding
      3. Return feature matrix
    """
    df = add_rolling_features(df)
    df, encoders = encode_categoricals(df, fit=fit, encoders=encoders)
    return df, encoders


def scale_features(X_train, X_test=None, fit: bool = True,
                   scaler=None):
    """
    Standard scaling.  Returns scaled arrays + fitted scaler.
    For tree-based models scaling is not needed but kept for SVM / LR.
    """
    if fit:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
    else:
        X_train_s = scaler.transform(X_train)

    if X_test is not None:
        X_test_s = scaler.transform(X_test)
        return X_train_s, X_test_s, scaler
    return X_train_s, scaler
