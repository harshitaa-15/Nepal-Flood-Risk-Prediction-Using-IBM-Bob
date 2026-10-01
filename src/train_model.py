"""
train_model.py
--------------
End-to-end training pipeline:
  1. Load & preprocess data
  2. Feature engineering
  3. Train/test split (time-aware)
  4. Handle class imbalance with SMOTE
  5. Train multiple classifiers
  6. Evaluate and compare
  7. Select best model (F1-macro weighted)
  8. Persist model artifacts to models/
"""

import os
import sys
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
SRC_DIR    = os.path.dirname(__file__)
ROOT_DIR   = os.path.join(SRC_DIR, "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)
from data_preprocessing import load_raw, preprocess
from feature_engineering import prepare_features, get_all_feature_columns

TARGET = "flood_risk"
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════
# 1 · Load & preprocess
# ══════════════════════════════════════════════════════════════════════════
def load_and_prepare():
    raw_df = load_raw()
    clean_df, thresholds = preprocess(raw_df)

    # Feature engineering
    feat_df, encoders = prepare_features(clean_df, fit=True)

    # Feature set
    feature_cols = get_all_feature_columns()
    # Drop any columns not present after engineering
    feature_cols = [c for c in feature_cols if c in feat_df.columns]

    X = feat_df[feature_cols]
    y = feat_df[TARGET]

    return feat_df, X, y, thresholds, encoders, feature_cols


# ══════════════════════════════════════════════════════════════════════════
# 2 · Train / test split  (time-aware: last 20% chronologically)
# ══════════════════════════════════════════════════════════════════════════
def time_split(feat_df, X, y):
    # Sort by date for temporal split
    sort_idx = feat_df["date"].argsort()
    X_sorted = X.iloc[sort_idx]
    y_sorted = y.iloc[sort_idx]

    split = int(len(X_sorted) * 0.80)
    X_train, X_test = X_sorted.iloc[:split], X_sorted.iloc[split:]
    y_train, y_test = y_sorted.iloc[:split], y_sorted.iloc[split:]
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════
# 3 · Handle class imbalance with SMOTE
# ══════════════════════════════════════════════════════════════════════════
def resample(X_train, y_train):
    # Fill any NaN introduced by rolling windows (first rows have no history)
    X_train = X_train.fillna(0)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_res).value_counts().sort_index().to_dict()}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════════════════════
# 4 · Model definitions
# ══════════════════════════════════════════════════════════════════════════
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# 5 · Evaluation helper
# ══════════════════════════════════════════════════════════════════════════
def evaluate(model, X_test, y_test):
    X_test = X_test.fillna(0)
    y_pred = model.predict(X_test)
    classes = sorted(y_test.unique())

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (one-vs-rest, weighted)
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = model.decision_function(X_test)
            # normalize for multiclass
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
        y_bin = label_binarize(y_test, classes=classes)
        auc = roc_auc_score(y_bin, y_prob[:, :len(classes)],
                            multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
    }


# ══════════════════════════════════════════════════════════════════════════
# 6 · Save artifacts
# ══════════════════════════════════════════════════════════════════════════
def save_artifacts(best_model, best_name, thresholds, encoders,
                   feature_cols, metrics_dict, X_test, y_test, feat_df):
    # Model
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    # Thresholds
    thresholds.to_csv(os.path.join(MODELS_DIR, "location_thresholds.csv"), index=False)
    # Encoders
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    # Feature columns
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    # Best model name
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)
    # All model metrics
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)
    # Confusion matrix
    _save_confusion_matrix(best_model, X_test, y_test, best_name)
    # Feature importance
    _save_feature_importance(best_model, feature_cols)
    # EDA plots from feat_df
    _save_eda_plots(feat_df)
    print("All artifacts saved to models/")


def _save_confusion_matrix(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["LOW", "MEDIUM", "HIGH"]
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()


def _save_feature_importance(model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="#3b82d4")
    ax.set_title("Top-20 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    # Also save as JSON for Streamlit
    imp_dict = imp.sort_values(ascending=False).head(15).to_dict()
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(imp_dict, f, indent=2)


def _save_eda_plots(df: pd.DataFrame):
    eda_dir = os.path.join(MODELS_DIR, "eda_plots")
    os.makedirs(eda_dir, exist_ok=True)
    label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    df = df.copy()
    df["risk_label"] = df["flood_risk"].map(label_map)

    # ── a) Risk distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["risk_label"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"])
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_title("Flood Risk Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "risk_distribution.png"), dpi=120)
    plt.close()

    # ── b) Monthly average precipitation
    monthly = df.groupby("month")["precipitation_mm"].mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 13), monthly.values, color="#3b82d4", edgecolor="white")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_title("Monthly Average Precipitation (mm)")
    ax.set_ylabel("mm")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "monthly_precipitation.png"), dpi=120)
    plt.close()

    # ── c) Correlation heatmap (numerical features)
    num_cols = [
        "precipitation_mm", "rain_mm", "soil_moisture_0_100cm_m3m3",
        "temperature_mean_c", "relative_humidity_mean_pct",
        "wind_speed_max_kmh", "river_discharge_m3s", "flood_risk",
    ]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "correlation_heatmap.png"), dpi=120)
    plt.close()

    # ── d) Monthly flood-risk distribution
    monthly_risk = df.groupby(["month", "risk_label"]).size().unstack(fill_value=0)
    monthly_risk = monthly_risk.reindex(columns=["LOW", "MEDIUM", "HIGH"], fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_risk.plot(kind="bar", ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"],
                      edgecolor="white", stacked=True)
    ax.set_xticklabels(month_names, rotation=45, ha="right")
    ax.set_title("Monthly Flood Risk Distribution")
    ax.set_ylabel("Count")
    ax.legend(title="Risk Level")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "monthly_risk.png"), dpi=120)
    plt.close()

    # ── e) Location-based average discharge
    loc_discharge = df.groupby("location")["river_discharge_m3s"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    loc_discharge.plot(kind="barh", ax=ax, color="#7c5cd8", edgecolor="white")
    ax.set_title("Average River Discharge by Location (m³/s)")
    ax.set_xlabel("m³/s")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "location_discharge.png"), dpi=120)
    plt.close()

    # ── f) Temperature trends by month
    fig, ax = plt.subplots(figsize=(8, 4))
    temp_monthly = df.groupby("month")["temperature_mean_c"].mean()
    ax.plot(range(1, 13), temp_monthly.values, marker="o", color="#e74c3c", linewidth=2)
    ax.fill_between(range(1, 13), temp_monthly.values, alpha=0.15, color="#e74c3c")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_title("Monthly Mean Temperature (°C)")
    ax.set_ylabel("°C")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "temperature_trend.png"), dpi=120)
    plt.close()

    # ── g) Humidity trends by month
    fig, ax = plt.subplots(figsize=(8, 4))
    hum_monthly = df.groupby("month")["relative_humidity_mean_pct"].mean()
    ax.plot(range(1, 13), hum_monthly.values, marker="s", color="#3498db", linewidth=2)
    ax.fill_between(range(1, 13), hum_monthly.values, alpha=0.15, color="#3498db")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_title("Monthly Mean Relative Humidity (%)")
    ax.set_ylabel("%")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "humidity_trend.png"), dpi=120)
    plt.close()

    # ── h) Discharge over time (all locations)
    fig, ax = plt.subplots(figsize=(12, 5))
    for loc in df["location"].unique():
        sub = df[df["location"] == loc].sort_values("date")
        ax.plot(sub["date"], sub["river_discharge_m3s"], alpha=0.6,
                linewidth=0.8, label=loc)
    ax.set_title("River Discharge Over Time by Location (m³/s)")
    ax.set_ylabel("m³/s")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "discharge_over_time.png"), dpi=120)
    plt.close()

    print("EDA plots saved.")


# ══════════════════════════════════════════════════════════════════════════
# 7 · Main training routine
# ══════════════════════════════════════════════════════════════════════════
def train():
    print("=" * 60)
    print("Nepal Flood Risk — Training Pipeline")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    feat_df, X, y, thresholds, encoders, feature_cols = load_and_prepare()
    print(f"Dataset: {X.shape[0]} rows × {X.shape[1]} features")
    print("Class distribution (before SMOTE):")
    print(y.value_counts().sort_index().rename({0: "LOW", 1: "MEDIUM", 2: "HIGH"}))

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = time_split(feat_df, X, y)
    print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── Resample ──────────────────────────────────────────────────────────
    X_res, y_res = resample(X_train, y_train)

    # ── Train & evaluate all models ───────────────────────────────────────
    models = get_models()
    metrics_dict = {}
    best_f1  = -1
    best_name  = None
    best_model = None

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_res, y_res)
        m = evaluate(model, X_test, y_test)
        metrics_dict[name] = m
        print(f"  Accuracy={m['accuracy']:.4f}  F1={m['f1_weighted']:.4f}  "
              f"AUC={m['roc_auc']}")
        if m["f1_weighted"] > best_f1:
            best_f1   = m["f1_weighted"]
            best_name = name
            best_model = model

    print(f"\nBest model: {best_name}  (F1={best_f1:.4f})")
    print("\nClassification Report:")
    y_pred = best_model.predict(X_test.fillna(0))
    print(classification_report(y_test, y_pred,
                                 target_names=["LOW", "MEDIUM", "HIGH"]))

    # ── Save ──────────────────────────────────────────────────────────────
    save_artifacts(best_model, best_name, thresholds, encoders,
                   feature_cols, metrics_dict, X_test.fillna(0), y_test, feat_df)

    return best_model, metrics_dict


if __name__ == "__main__":
    train()
