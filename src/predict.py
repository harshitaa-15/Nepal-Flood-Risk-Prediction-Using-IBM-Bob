"""
predict.py
----------
Inference module: given raw input values, returns flood risk level,
probability, and top contributing features.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

SRC_DIR    = os.path.dirname(__file__)
ROOT_DIR   = os.path.join(SRC_DIR, "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

sys.path.insert(0, SRC_DIR)
from feature_engineering import prepare_features

RISK_LABELS  = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
RISK_COLORS  = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
RISK_EMOJIS  = {0: "✅", 1: "⚠️", 2: "🚨"}


def load_artifacts():
    """Load all persisted model artifacts."""
    model         = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    encoders      = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    feature_cols  = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    thresholds    = pd.read_csv(os.path.join(MODELS_DIR, "location_thresholds.csv"))
    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()
    return model, encoders, feature_cols, thresholds, model_name


def build_input_row(user_inputs: dict, location: str, month: int,
                    day_of_year: int, encoders: dict) -> pd.DataFrame:
    """
    Convert flat user-input dict into a single-row DataFrame with all
    engineered features set to sensible defaults (rolling features → 0).

    Parameters
    ----------
    user_inputs : dict of raw feature values
    location    : location string (must be in the encoder's classes)
    month       : calendar month (1–12)
    day_of_year : day of year (1–366)
    encoders    : fitted LabelEncoder dict from training
    """
    row = user_inputs.copy()

    # Date-derived cyclical features
    row["month_sin"]   = np.sin(2 * np.pi * month / 12)
    row["month_cos"]   = np.cos(2 * np.pi * month / 12)
    row["day_of_year"] = day_of_year
    row["month"]       = month

    # Rolling features default to current values (best approximation for
    # single-point inference where history is unknown)
    for window in [3, 7]:
        for col in ["precipitation_mm", "rain_mm", "relative_humidity_mean_pct"]:
            row[f"{col}_roll{window}_mean"] = user_inputs.get(col, 0)
            row[f"{col}_roll{window}_sum"]  = user_inputs.get(col, 0) * window

    row["precip_roll7_max"]  = user_inputs.get("precipitation_mm", 0)
    row["saturation_index"]  = (
        user_inputs.get("soil_moisture_0_100cm_m3m3", 0.3)
        * user_inputs.get("relative_humidity_mean_pct", 70) / 100
    )
    row["heavy_rain_flag"]   = 1 if user_inputs.get("precipitation_mm", 0) > 20 else 0
    row["temp_dew_spread"]   = (
        user_inputs.get("temperature_mean_c", 20)
        - user_inputs.get("dew_point_mean_c", 15)
    )

    # Location encoding
    le = encoders["location"]
    known_classes = list(le.classes_)
    if location in known_classes:
        row["location_enc"] = le.transform([location])[0]
    else:
        # Unknown location: use median encoding
        row["location_enc"] = len(known_classes) // 2

    df_row = pd.DataFrame([row])
    return df_row


def predict(user_inputs: dict, location: str, month: int,
            day_of_year: int) -> dict:
    """
    Main inference function.

    Returns
    -------
    dict with keys:
        risk_level    : "LOW" | "MEDIUM" | "HIGH"
        risk_class    : 0 | 1 | 2
        probability   : float (0–1), probability of predicted class
        probabilities : list[float], probs for [LOW, MEDIUM, HIGH]
        color         : hex color string
        emoji         : emoji string
        top_factors   : list of (feature_name, importance_value) tuples
    """
    model, encoders, feature_cols, thresholds, _ = load_artifacts()

    df_row = build_input_row(user_inputs, location, month, day_of_year, encoders)

    # Align columns
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_cols]

    pred_class = int(model.predict(df_row)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_row)[0].tolist()
        # Ensure list is length 3 (LOW, MED, HIGH)
        while len(proba) < 3:
            proba.append(0.0)
    else:
        proba = [0.0, 0.0, 0.0]
        proba[pred_class] = 1.0

    prob_pred = proba[pred_class]

    # Top contributing features (from model importances)
    top_factors = []
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        top_factors = imp.sort_values(ascending=False).head(6).items()
        top_factors = [(k, round(float(v), 4)) for k, v in top_factors]

    return {
        "risk_level":    RISK_LABELS[pred_class],
        "risk_class":    pred_class,
        "probability":   round(prob_pred, 4),
        "probabilities": [round(p, 4) for p in proba[:3]],
        "color":         RISK_COLORS[pred_class],
        "emoji":         RISK_EMOJIS[pred_class],
        "top_factors":   top_factors,
    }


def get_known_locations() -> list:
    """Return list of locations the model was trained on."""
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    return list(encoders["location"].classes_)


if __name__ == "__main__":
    # Quick smoke test
    sample_inputs = {
        "precipitation_mm": 35.0,
        "rain_mm": 34.5,
        "soil_moisture_0_100cm_m3m3": 0.42,
        "temperature_mean_c": 28.0,
        "dew_point_mean_c": 22.0,
        "precipitation_hours": 10,
        "relative_humidity_mean_pct": 88,
        "wind_speed_max_kmh": 25.0,
        "wind_gusts_max_kmh": 55.0,
        "wind_direction_dominant_deg": 180,
        "elevation_m": 300,
    }
    result = predict(sample_inputs, "Chatara", month=8, day_of_year=225)
    print(result)
