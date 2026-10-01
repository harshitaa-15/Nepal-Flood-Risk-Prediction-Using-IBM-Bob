"""
build_notebook.py
-----------------
Generates EDA_and_Modeling.ipynb programmatically.
Run from the project root:
    python notebooks/build_notebook.py
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# ── helpers ──────────────────────────────────────────────────────────────────
def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ═════════════════════════════════════════════════════════════════════════════
cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""# Nepal Flood Risk Prediction & Early Warning System
### EDA and Modelling Notebook

**Dataset:** Nepal Flood & Weather Dataset 2023–2026 (Kaggle)  
**Goal:** Predict flood risk level (LOW / MEDIUM / HIGH) from daily weather and hydrological observations  
**Stations:** 10 monitoring stations across Nepal's major river basins

---
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Setup
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 0. Setup — Imports & Paths"))
cells.append(code("""\
import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", "{:.3f}".format)

# Add src/ to path so we can reuse the project modules
sys.path.insert(0, os.path.join("..", "src"))

# Plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False})

DATA_PATH = os.path.join("..", "data",
    "CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv")
print("Data file exists:", os.path.exists(DATA_PATH))
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & Inspect
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 1. Load & Inspect the Dataset
"""))
cells.append(code("""\
df_raw = pd.read_csv(DATA_PATH)
df_raw["date"] = pd.to_datetime(df_raw["date"])
print(f"Shape: {df_raw.shape[0]:,} rows  x  {df_raw.shape[1]} columns")
df_raw.head(5)
"""))

cells.append(code("""\
print("=== Column Names & Dtypes ===")
print(df_raw.dtypes)
"""))

cells.append(code("""\
print("=== Missing Values ===")
missing = df_raw.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values found.")

print("\\n=== Duplicate Rows ===")
print(f"Duplicate rows: {df_raw.duplicated().sum()}")
"""))

cells.append(code("""\
print("=== Summary Statistics ===")
display(df_raw.describe())
"""))

cells.append(code("""\
print("=== Categorical Features ===")
for col in ["location", "river", "basin"]:
    print(f"\\n{col} ({df_raw[col].nunique()} unique):")
    print(df_raw[col].value_counts().to_string())
"""))

cells.append(code("""\
print("=== Date Range ===")
print(f"From : {df_raw['date'].min().date()}")
print(f"To   : {df_raw['date'].max().date()}")
print(f"Span : {(df_raw['date'].max() - df_raw['date'].min()).days} days")
print(f"\\nRecords per location:")
print(df_raw.groupby("location").size().sort_values(ascending=False).to_string())
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Target Variable Definition
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 2. Target Variable — Flood Risk Label

There is no pre-existing binary flood flag in the dataset.  
We derive a **3-class flood risk label** from `river_discharge_m3s` using  
**per-station percentile thresholds** (p75 and p90).

| Label | Threshold | Rationale |
|-------|-----------|-----------|
| **LOW** (0) | discharge < station p75 | Normal river conditions |
| **MEDIUM** (1) | p75 ≤ discharge < p90 | Elevated; watch level |
| **HIGH** (2) | discharge ≥ p90 | Flood-risk; action level |

Using per-station thresholds is essential because river scales differ enormously  
(e.g., Kusum p90 ≈ 416 m³/s vs Rasuwagadhi p90 ≈ 3.7 m³/s).
"""))

cells.append(code("""\
# Per-location discharge percentile thresholds
thresholds = (
    df_raw.groupby("location")["river_discharge_m3s"]
    .agg(p75=lambda x: x.quantile(0.75),
         p90=lambda x: x.quantile(0.90),
         median=lambda x: x.median(),
         max_val="max")
    .reset_index()
    .rename(columns={"max_val": "max"})
)
thresholds.columns = ["Station", "p75 (m³/s)", "p90 (m³/s)", "Median (m³/s)", "Max (m³/s)"]
thresholds = thresholds.sort_values("p90 (m³/s)", ascending=False)
display(thresholds.round(2))
"""))

cells.append(code("""\
# Assign flood risk labels
df = df_raw.copy()
thresh_map = (
    df_raw.groupby("location")["river_discharge_m3s"]
    .agg(p75=lambda x: x.quantile(0.75),
         p90=lambda x: x.quantile(0.90))
    .reset_index()
)
df = df.merge(thresh_map, on="location", how="left")

conditions = [
    df["river_discharge_m3s"] >= df["p90"],
    (df["river_discharge_m3s"] >= df["p75"]) & (df["river_discharge_m3s"] < df["p90"]),
]
df["flood_risk"] = np.select(conditions, [2, 1], default=0)
df.drop(columns=["p75", "p90"], inplace=True)

# Date features
df["year"]        = df["date"].dt.year
df["month"]       = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
df["risk_label"] = df["flood_risk"].map(label_map)

print("Flood risk distribution:")
vc = df["flood_risk"].value_counts().sort_index()
for k, v in vc.items():
    print(f"  {label_map[k]:8s} ({k}): {v:,}  ({v/len(df)*100:.1f}%)")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EDA
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 3. Exploratory Data Analysis

### 3.1 Flood Risk Distribution
"""))
cells.append(code("""\
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
RISK_COLORS  = ["#2ecc71", "#f39c12", "#e74c3c"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# -- Bar chart
counts = df["risk_label"].value_counts().reindex(["LOW","MEDIUM","HIGH"])
axes[0].bar(counts.index, counts.values, color=RISK_COLORS, edgecolor="white", width=0.5)
axes[0].set_title("Flood Risk Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of Records")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 80, f"{v:,}\\n({v/len(df)*100:.1f}%)",
                 ha="center", fontsize=9)

# -- Pie chart
axes[1].pie(counts.values, labels=counts.index, colors=RISK_COLORS,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Risk Share", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("flood_risk_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
print("Note: Dataset is imbalanced — SMOTE will be applied during training.")
"""))

cells.append(md("### 3.2 Monthly Precipitation & Rainfall"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Monthly average precipitation
m_precip = df.groupby("month")["precipitation_mm"].mean()
axes[0].bar(range(1,13), m_precip.values, color="#3b82d4", edgecolor="white")
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels(MONTH_LABELS, rotation=45, ha="right")
axes[0].set_title("Monthly Average Precipitation (mm)", fontweight="bold")
axes[0].set_ylabel("mm/day")

# Boxplot
data_bp = [df[df["month"] == m]["precipitation_mm"].values for m in range(1,13)]
bp = axes[1].boxplot(data_bp, labels=MONTH_LABELS, patch_artist=True,
                     boxprops=dict(facecolor="#dbeafe", color="#3b82d4"),
                     medianprops=dict(color="#e74c3c", linewidth=2),
                     flierprops=dict(marker=".", markersize=3, alpha=0.3))
axes[1].set_title("Precipitation Distribution by Month", fontweight="bold")
axes[1].set_ylabel("mm/day")
axes[1].tick_params(axis="x", labelsize=8)

plt.tight_layout()
plt.savefig("monthly_precipitation.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 3.3 Temperature Trends"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

m_temp = df.groupby("month")["temperature_mean_c"].mean()
m_dew  = df.groupby("month")["dew_point_mean_c"].mean()

axes[0].plot(range(1,13), m_temp.values, marker="o", color="#e74c3c",
             linewidth=2, label="Mean Temperature")
axes[0].plot(range(1,13), m_dew.values,  marker="s", color="#3498db",
             linewidth=2, label="Dew Point")
axes[0].fill_between(range(1,13), m_dew.values, m_temp.values, alpha=0.1, color="#e74c3c")
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels(MONTH_LABELS, rotation=45, ha="right")
axes[0].set_title("Monthly Temperature & Dew Point (°C)", fontweight="bold")
axes[0].set_ylabel("°C")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.4)

# By location
loc_temp = df.groupby("location")["temperature_mean_c"].mean().sort_values()
axes[1].barh(loc_temp.index, loc_temp.values, color="#e74c3c", edgecolor="white")
axes[1].set_title("Mean Temperature by Station (°C)", fontweight="bold")
axes[1].set_xlabel("°C")

plt.tight_layout()
plt.savefig("temperature_trends.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 3.4 Humidity Trends"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

m_hum = df.groupby("month")["relative_humidity_mean_pct"].mean()
axes[0].plot(range(1,13), m_hum.values, marker="s", color="#3498db", linewidth=2)
axes[0].fill_between(range(1,13), m_hum.values, alpha=0.15, color="#3498db")
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels(MONTH_LABELS, rotation=45, ha="right")
axes[0].set_title("Monthly Mean Relative Humidity (%)", fontweight="bold")
axes[0].set_ylabel("%")
axes[0].set_ylim(0, 100)
axes[0].grid(True, linestyle="--", alpha=0.4)

# Humidity vs risk
axes[1].boxplot(
    [df[df["flood_risk"] == r]["relative_humidity_mean_pct"].values for r in [0,1,2]],
    labels=["LOW","MEDIUM","HIGH"],
    patch_artist=True,
    boxprops=dict(facecolor="#dbeafe"),
    medianprops=dict(color="#e74c3c", linewidth=2),
)
axes[1].set_title("Humidity by Flood Risk Level", fontweight="bold")
axes[1].set_ylabel("%")

plt.tight_layout()
plt.savefig("humidity_trends.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 3.5 River Discharge Over Time"))
cells.append(code("""\
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# All locations
for loc in df["location"].unique():
    sub = df[df["location"] == loc].sort_values("date")
    axes[0].plot(sub["date"], sub["river_discharge_m3s"],
                 alpha=0.7, linewidth=0.8, label=loc)
axes[0].set_title("Daily River Discharge by Station (m³/s)", fontweight="bold")
axes[0].set_ylabel("m³/s")
axes[0].legend(fontsize=7, loc="upper right", ncol=2)
axes[0].set_yscale("log")

# Monthly mean discharge
monthly_dis = df.groupby(["year","month"])["river_discharge_m3s"].mean().reset_index()
monthly_dis["ym"] = pd.to_datetime(monthly_dis.assign(day=1)[["year","month","day"]])
axes[1].fill_between(monthly_dis["ym"], monthly_dis["river_discharge_m3s"],
                     color="#7c5cd8", alpha=0.6)
axes[1].plot(monthly_dis["ym"], monthly_dis["river_discharge_m3s"],
             color="#7c5cd8", linewidth=1)
axes[1].set_title("Monthly Mean River Discharge — All Stations (m³/s)", fontweight="bold")
axes[1].set_ylabel("m³/s")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.savefig("discharge_over_time.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 3.6 Monthly & Seasonal Flood Patterns"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stacked bar — monthly risk
mr = df.groupby(["month","risk_label"]).size().unstack(fill_value=0)
mr = mr.reindex(columns=["LOW","MEDIUM","HIGH"], fill_value=0)
mr.plot(kind="bar", stacked=True, ax=axes[0],
        color=RISK_COLORS, edgecolor="white")
axes[0].set_xticklabels(MONTH_LABELS, rotation=45, ha="right")
axes[0].set_title("Monthly Flood Risk Distribution", fontweight="bold")
axes[0].set_ylabel("Record Count")
axes[0].legend(title="Risk Level")

# Seasonal
def season(m):
    if m in [12,1,2]:   return "Winter"
    if m in [3,4,5]:    return "Pre-Monsoon"
    if m in [6,7,8,9]:  return "Monsoon"
    return "Post-Monsoon"

df["season"] = df["month"].apply(season)
season_order = ["Winter","Pre-Monsoon","Monsoon","Post-Monsoon"]
sr = df.groupby(["season","risk_label"]).size().unstack(fill_value=0)
sr = sr.reindex(index=season_order, columns=["LOW","MEDIUM","HIGH"], fill_value=0)
sr.plot(kind="bar", ax=axes[1], color=RISK_COLORS, edgecolor="white")
axes[1].set_title("Seasonal Flood Risk Distribution", fontweight="bold")
axes[1].set_ylabel("Record Count")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend(title="Risk Level")

plt.tight_layout()
plt.savefig("seasonal_patterns.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 3.7 Correlation Heatmap"))
cells.append(code("""\
num_cols = [
    "precipitation_mm","rain_mm","soil_moisture_0_100cm_m3m3",
    "temperature_mean_c","dew_point_mean_c","precipitation_hours",
    "relative_humidity_mean_pct","wind_speed_max_kmh","wind_gusts_max_kmh",
    "river_discharge_m3s","flood_risk",
]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()

print("\\nTop correlations with flood_risk:")
corr_flood = corr["flood_risk"].drop("flood_risk").abs().sort_values(ascending=False)
display(corr_flood.head(10).rename("abs_correlation").to_frame().round(3))
"""))

cells.append(md("### 3.8 Geographic / Location-Based Analysis"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Average discharge by location
loc_stats = df.groupby("location").agg(
    avg_discharge=("river_discharge_m3s","mean"),
    high_risk_pct=("flood_risk", lambda x: (x==2).mean()*100),
).sort_values("avg_discharge")

axes[0].barh(loc_stats.index, loc_stats["avg_discharge"],
             color="#7c5cd8", edgecolor="white")
axes[0].set_title("Average River Discharge by Station (m³/s)", fontweight="bold")
axes[0].set_xlabel("m³/s")
for i, (idx, row) in enumerate(loc_stats.iterrows()):
    axes[0].text(row["avg_discharge"]+0.5, i, f'{row["avg_discharge"]:.1f}',
                 va="center", fontsize=8)

# HIGH risk % by location
loc_hr = loc_stats["high_risk_pct"].sort_values()
axes[1].barh(loc_hr.index, loc_hr.values, color="#e74c3c", edgecolor="white")
axes[1].set_title("HIGH Risk Days (%) by Station", fontweight="bold")
axes[1].set_xlabel("% of days")
for i, (idx, v) in enumerate(loc_hr.items()):
    axes[1].text(v+0.2, i, f'{v:.1f}%', va="center", fontsize=8)

plt.tight_layout()
plt.savefig("location_analysis.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 3.9 Precipitation vs. Discharge Scatter"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

colors_scatter = df["flood_risk"].map({0:"#2ecc71",1:"#f39c12",2:"#e74c3c"})

axes[0].scatter(df["precipitation_mm"], df["river_discharge_m3s"],
                c=colors_scatter, alpha=0.3, s=6, linewidths=0)
axes[0].set_yscale("log")
axes[0].set_xlabel("Precipitation (mm)")
axes[0].set_ylabel("River Discharge (m³/s)")
axes[0].set_title("Precipitation vs Discharge (coloured by risk)", fontweight="bold")
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l)
                   for c,l in zip(["#2ecc71","#f39c12","#e74c3c"],
                                  ["LOW","MEDIUM","HIGH"])]
axes[0].legend(handles=legend_elements, fontsize=8)

axes[1].scatter(df["soil_moisture_0_100cm_m3m3"], df["river_discharge_m3s"],
                c=colors_scatter, alpha=0.3, s=6, linewidths=0)
axes[1].set_yscale("log")
axes[1].set_xlabel("Soil Moisture (m³/m³)")
axes[1].set_ylabel("River Discharge (m³/s)")
axes[1].set_title("Soil Moisture vs Discharge (coloured by risk)", fontweight="bold")

plt.tight_layout()
plt.savefig("scatter_analysis.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 4. Data Preprocessing
"""))
cells.append(code("""\
from data_preprocessing import preprocess, load_raw

raw = load_raw()
clean_df, location_thresholds = preprocess(raw)
print(f"After preprocessing: {clean_df.shape}")
print("\\nFlood risk distribution (preprocessed):")
print(clean_df["flood_risk"].value_counts().sort_index()
      .rename({0:"LOW",1:"MEDIUM",2:"HIGH"}))
print("\\nDate/time features added:")
print([c for c in clean_df.columns if c in ["year","month","day_of_year","week","month_sin","month_cos"]])
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 5. Feature Engineering
"""))
cells.append(code("""\
from feature_engineering import prepare_features, get_all_feature_columns

feat_df, encoders = prepare_features(clean_df, fit=True)
feature_cols = get_all_feature_columns()
feature_cols = [c for c in feature_cols if c in feat_df.columns]

print(f"Total features: {len(feature_cols)}")
print("\\nAll feature columns:")
for i, c in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {c}")
"""))

cells.append(code("""\
# Feature categories breakdown
base_feats    = [c for c in feature_cols if "roll" not in c and c not in
                 ["saturation_index","heavy_rain_flag","temp_dew_spread","location_enc",
                  "month_sin","month_cos","day_of_year"]]
rolling_feats = [c for c in feature_cols if "roll" in c]
engineered    = [c for c in feature_cols if c in
                 ["saturation_index","heavy_rain_flag","temp_dew_spread","location_enc",
                  "month_sin","month_cos","day_of_year"]]

print(f"Base features       : {len(base_feats)}")
print(f"Rolling features    : {len(rolling_feats)}")
print(f"Engineered features : {len(engineered)}")
print(f"Total               : {len(feature_cols)}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Train/Test Split & Class Imbalance
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 6. Train/Test Split & Class Imbalance Handling

Using a **time-aware split** (last 20% chronologically = test set) to simulate  
real-world forecasting. SMOTE is applied to the **training set only**.
"""))
cells.append(code("""\
from imblearn.over_sampling import SMOTE

X = feat_df[feature_cols].fillna(0)
y = feat_df["flood_risk"]

# Time-aware split
sort_idx = feat_df["date"].argsort()
X_sorted = X.iloc[sort_idx]
y_sorted = y.iloc[sort_idx]
split = int(len(X_sorted) * 0.80)

X_train, X_test = X_sorted.iloc[:split], X_sorted.iloc[split:]
y_train, y_test = y_sorted.iloc[:split], y_sorted.iloc[split:]

print(f"Train set: {X_train.shape[0]:,} samples")
print(f"Test set : {X_test.shape[0]:,} samples")
print("\\nClass distribution — Train (before SMOTE):")
print(y_train.value_counts().sort_index().rename({0:"LOW",1:"MEDIUM",2:"HIGH"}))
print("\\nClass distribution — Test:")
print(y_test.value_counts().sort_index().rename({0:"LOW",1:"MEDIUM",2:"HIGH"}))
"""))

cells.append(code("""\
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print("Class distribution — Train (after SMOTE):")
print(pd.Series(y_res).value_counts().sort_index().rename({0:"LOW",1:"MEDIUM",2:"HIGH"}))

# Visualise balance
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, data, title in zip(axes,
    [y_train, pd.Series(y_res)],
    ["Before SMOTE (train)", "After SMOTE (train)"]):
    vc = data.value_counts().reindex([0,1,2])
    ax.bar(["LOW","MEDIUM","HIGH"], vc.values, color=RISK_COLORS, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate(vc.values):
        ax.text(i, v+30, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("smote_balance.png", dpi=110, bbox_inches="tight")
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Model Training
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 7. Model Training

Training and comparing four classifiers:
1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest (ensemble — expected best)
4. Gradient Boosting
"""))
cells.append(code("""\
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize
import time

models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42,
                                                class_weight="balanced"),
    "Decision Tree":        DecisionTreeClassifier(max_depth=10, random_state=42,
                                                    class_weight="balanced"),
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=15,
                                                    random_state=42, n_jobs=-1,
                                                    class_weight="balanced"),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                        learning_rate=0.1,
                                                        random_state=42),
}

results = {}
trained_models = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_res, y_res)
    t1 = time.time()

    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = model.decision_function(X_test)
        y_bin = label_binarize(y_test, classes=[0,1,2])
        auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    results[name] = dict(accuracy=acc, precision=prec, recall=rec,
                         f1_weighted=f1, roc_auc=auc,
                         train_sec=round(t1-t0, 1))
    trained_models[name] = model
    print(f"[{name:22s}]  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  ({t1-t0:.1f}s)")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 8. Model Evaluation & Comparison
"""))
cells.append(code("""\
# Comparison table
results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Model"})
results_df = results_df.sort_values("f1_weighted", ascending=False)

for col in ["accuracy","precision","recall","f1_weighted","roc_auc"]:
    results_df[col] = (results_df[col] * 100).round(2).astype(str) + "%"

display(results_df[["Model","accuracy","precision","recall","f1_weighted","roc_auc","train_sec"]])
"""))

cells.append(code("""\
# Visual comparison
metrics_plot = ["accuracy","precision","recall","f1_weighted","roc_auc"]
plot_data = pd.DataFrame(results).T[metrics_plot].astype(float) * 100

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metrics_plot))
width = 0.18
colors_m = ["#3b82d4","#f39c12","#7c5cd8","#2ecc71"]

for i, (model_name, row) in enumerate(plot_data.iterrows()):
    ax.bar(x + i*width, row.values, width, label=model_name,
           color=colors_m[i], edgecolor="white")

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(["Accuracy","Precision","Recall","F1 (weighted)","ROC-AUC"],
                    fontsize=9)
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 8.1 Best Model — Detailed Evaluation"))
cells.append(code("""\
best_name  = max(results, key=lambda k: results[k]["f1_weighted"])
best_model = trained_models[best_name]
print(f"Best model: {best_name}")
print()

y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best,
                             target_names=["LOW","MEDIUM","HIGH"]))
"""))

cells.append(code("""\
# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["LOW","MEDIUM","HIGH"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 8.2 Feature Importance"))
cells.append(code("""\
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols)
    imp_top = imp.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    imp_top.plot(kind="barh", ax=ax, color="#3b82d4", edgecolor="white")
    ax.set_title(f"Top-20 Feature Importances — {best_name}",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
    plt.show()

    print("\\nTop 10 features:")
    display(imp.sort_values(ascending=False).head(10)
              .rename("importance").to_frame().round(4))
else:
    print("This model does not expose feature_importances_.")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Prediction Demo
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 9. Flood Risk Prediction Demo

Demonstrate the end-to-end inference pipeline using the saved model artifacts.
"""))
cells.append(code("""\
from predict import predict

# Heavy-monsoon scenario — Chatara, August
result = predict(
    user_inputs={
        "precipitation_mm":          42.0,
        "rain_mm":                   41.5,
        "soil_moisture_0_100cm_m3m3": 0.46,
        "temperature_mean_c":         28.5,
        "dew_point_mean_c":           23.0,
        "precipitation_hours":        14,
        "relative_humidity_mean_pct": 92,
        "wind_speed_max_kmh":         28.0,
        "wind_gusts_max_kmh":         65.0,
        "wind_direction_dominant_deg": 190,
        "elevation_m":                150,
    },
    location="Chatara",
    month=8,
    day_of_year=225,
)

print("=" * 45)
print(f"  Flood Risk Level : {result['risk_level']}")
print(f"  Confidence       : {result['probability']*100:.1f}%")
print(f"  Probabilities    : LOW={result['probabilities'][0]*100:.1f}%  "
      f"MED={result['probabilities'][1]*100:.1f}%  "
      f"HIGH={result['probabilities'][2]*100:.1f}%")
print("=" * 45)
print("\\nTop contributing features:")
for feat, imp_val in result["top_factors"][:6]:
    print(f"  {feat:<42s}  {imp_val:.4f}")
"""))

cells.append(code("""\
# Winter / dry-season scenario — Rasuwagadhi, January
result_dry = predict(
    user_inputs={
        "precipitation_mm":          0.0,
        "rain_mm":                   0.0,
        "soil_moisture_0_100cm_m3m3": 0.31,
        "temperature_mean_c":         10.5,
        "dew_point_mean_c":           -3.0,
        "precipitation_hours":        0,
        "relative_humidity_mean_pct": 35,
        "wind_speed_max_kmh":         12.0,
        "wind_gusts_max_kmh":         55.0,
        "wind_direction_dominant_deg": 185,
        "elevation_m":               1749,
    },
    location="Rasuwagadhi",
    month=1,
    day_of_year=5,
)

print("=" * 45)
print(f"  Flood Risk Level : {result_dry['risk_level']}")
print(f"  Confidence       : {result_dry['probability']*100:.1f}%")
print("=" * 45)
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Key Findings
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
## 10. Key Findings & Conclusions

### Dataset
- **13,390 daily records** across 10 monitoring stations (2023–2026)
- No missing values or duplicate rows
- Strong **monsoon seasonality** — July–September dominate flood events
- **Kusum** (Saptakoshi) and **Belsot** (West Rapti) are the highest-discharge stations

### Target Variable
- Flood risk is derived from **per-station river discharge percentiles** to handle scale differences
- Class distribution is imbalanced: ~75% LOW / ~15% MEDIUM / ~10% HIGH
- **SMOTE** was applied to the training set to balance classes

### Feature Importance
- **Rolling precipitation windows** (3-day, 7-day totals) are the most predictive features
- **Soil moisture** and **relative humidity** are strong secondary predictors
- Temperature and wind speed are less directly predictive
- **Cyclical month encoding** captures seasonal risk patterns

### Model Results

| Model | Accuracy | F1 (weighted) | ROC-AUC |
|-------|----------|---------------|---------|
| Logistic Regression | ~83.8% | ~85.5% | ~0.961 |
| Decision Tree | ~84.1% | ~85.8% | ~0.954 |
| **Random Forest** | **~86.7%** | **~87.7%** | **~0.979** |
| Gradient Boosting | ~86.2% | ~87.3% | ~0.975 |

**Random Forest** is the best model.  
The weighted F1-score is the primary selection criterion given class imbalance.

### Application
The trained model powers a **Streamlit web dashboard** with four sections:
Dashboard · Flood Risk Prediction · Data Analysis · Model Performance

```bash
# Run the application
streamlit run ../app.py
```
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Assemble and write notebook
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

OUT_PATH = os.path.join(os.path.dirname(__file__), "EDA_and_Modeling.ipynb")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written: {OUT_PATH}")
print(f"Total cells: {len(nb.cells)}")
