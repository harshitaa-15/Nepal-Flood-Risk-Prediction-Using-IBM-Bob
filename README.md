# Nepal Flood Risk Prediction & Early Warning System

A complete end-to-end Machine Learning web application that predicts **flood risk levels** across Nepal's major river monitoring stations using daily weather and hydrological data (2023–2026).

---

## Demo

| Dashboard | Prediction | Data Analysis | Model Performance |
|-----------|-----------|---------------|-------------------|
| Dataset KPIs, risk distribution, station summary | Input weather params → instant risk assessment | Interactive charts, seasonal trends, correlations | Confusion matrix, feature importances, model comparison |

---

## Dataset

**Source:** [Nepal Flood & Weather Dataset 2023–2026 — Kaggle](https://www.kaggle.com/datasets/tejal5kunjir/nepal-flood-and-weather-dataset-2023-2026)

| Property | Value |
|----------|-------|
| Records | 13,390 |
| Features | 19 raw + 12 engineered |
| Locations | 10 monitoring stations |
| Rivers | 9 (Bhote Koshi, Bagmati, Narayani, Saptakoshi, Kamala, West Rapti, Babai, Karnali, Chamelia) |
| Date Range | 2023-01-01 → 2026-08-31 |

---

## Project Structure

```
nepal-flood-risk-prediction/
│
├── data/
│   └── CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv
│
├── src/
│   ├── data_preprocessing.py   # Load, clean, label flood risk
│   ├── feature_engineering.py  # Rolling features, encoding
│   ├── train_model.py          # Full training pipeline
│   ├── predict.py              # Inference module
│   └── utils.py                # Shared helpers
│
├── models/
│   ├── best_model.pkl          # Trained classifier
│   ├── encoders.pkl            # LabelEncoders
│   ├── feature_columns.pkl     # Feature list
│   ├── location_thresholds.csv # Per-station p75/p90 thresholds
│   ├── metrics.json            # All model metrics
│   ├── feature_importance.json # Feature importances
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── eda_plots/              # EDA visualisations
│
├── app.py                      # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd nepal-flood-risk-prediction
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

This will:
- Preprocess the dataset
- Engineer rolling and cyclical features
- Apply SMOTE to handle class imbalance
- Train and compare 4 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Save the best model and all evaluation artifacts to `models/`

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Machine Learning Pipeline

### Target Variable

Flood risk is a **3-class classification** problem derived from per-station river discharge percentiles:

| Label | Threshold | Class |
|-------|-----------|-------|
| LOW | discharge < station p75 | 0 |
| MEDIUM | p75 ≤ discharge < p90 | 1 |
| HIGH | discharge ≥ p90 | 2 |

Using per-station thresholds ensures rivers of very different scales (e.g., Chamelia vs. Bhote Koshi) are treated fairly.

### Features

**Raw:**
- `precipitation_mm`, `rain_mm`, `soil_moisture_0_100cm_m3m3`
- `temperature_mean_c`, `dew_point_mean_c`, `dew_point_mean_c`
- `precipitation_hours`, `relative_humidity_mean_pct`
- `wind_speed_max_kmh`, `wind_gusts_max_kmh`, `wind_direction_dominant_deg`
- `elevation_m`

**Engineered:**
- 3-day and 7-day rolling means/sums for precipitation, rain, humidity
- 7-day rolling max precipitation
- Cyclical month encoding (`month_sin`, `month_cos`)
- Saturation index (soil moisture × humidity)
- Temperature–dew point spread
- Location label encoding

### Class Imbalance

The dataset is significantly imbalanced (75% LOW, 15% MEDIUM, 10% HIGH). SMOTE is applied to the training set only to produce balanced classes.

### Model Results

| Model | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC |
|-------|----------|-----------|--------|---------------|---------|
| Logistic Regression | 83.8% | — | — | 85.5% | 0.961 |
| Decision Tree | 84.1% | — | — | 85.8% | 0.954 |
| **Random Forest** | **86.7%** | — | — | **87.7%** | **0.979** |
| Gradient Boosting | 86.2% | — | — | 87.3% | 0.975 |

**Best model: Random Forest** (time-aware test split, SMOTE-balanced training)

---

## Key Findings

- **Monsoon seasonality** drives most flood events (July–September)
- **Precipitation rolling windows** and **soil moisture** are the most predictive features
- **Kusum (Saptakoshi)** and **Belsot (West Rapti)** stations show the highest discharge volumes
- Temperature and dew point are less predictive than precipitation and soil saturation

---

## Application Sections

### Dashboard
- Project overview and dataset statistics
- Flood risk distribution chart
- Monthly risk pattern
- Station-level summary table

### Flood Risk Prediction
- Select station and date
- Enter weather parameters with units
- Get instant risk level (LOW / MEDIUM / HIGH) with confidence percentage
- View top contributing factors

### Data Analysis
- Interactive filtering by station and year
- Tabs: Precipitation, Temperature & Humidity, River Discharge, Seasonal Patterns, Correlations

### Model Performance
- Selected model metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Model comparison table and chart
- Confusion matrix
- Feature importance chart

---

## Technologies

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML | scikit-learn, imbalanced-learn |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Frontend | Streamlit |
| Persistence | joblib |

---

## License

MIT — free to use for portfolio, academic, and non-commercial purposes.

---

*Nepal Flood Risk Prediction & Early Warning System — Built for ML/Data Analytics Portfolio*



