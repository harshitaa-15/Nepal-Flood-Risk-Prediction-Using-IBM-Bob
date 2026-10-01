"""
app.py  –  Nepal Flood Intelligence & Early Warning Center
Professional AI-powered Streamlit dashboard
"""

import os, sys, warnings, datetime, json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT_DIR   = os.path.dirname(__file__)
SRC_DIR    = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
sys.path.insert(0, SRC_DIR)

from data_preprocessing import load_raw, preprocess, FLOOD_LABELS, FLOOD_COLORS, get_display_feature_meta
from utils import (get_dataset_stats, load_metrics, load_feature_importance,
                   get_best_model_name, models_trained, pretty_feature, month_to_season)
from predict import predict, get_known_locations

# ─── Colour tokens ───────────────────────────────────────────────────────────
C_LOW    = "#16a34a"   # green-600
C_MED    = "#d97706"   # amber-600
C_HIGH   = "#dc2626"   # red-600
C_BLUE   = "#1d4ed8"   # blue-700
C_NAVY   = "#0f172a"   # slate-900
C_MUTED  = "#64748b"   # slate-500
C_BG     = "#f8fafc"   # slate-50
C_CARD   = "#ffffff"
C_BORDER = "#e2e8f0"

RISK_COLOR  = {"LOW": C_LOW, "MEDIUM": C_MED, "HIGH": C_HIGH}
RISK_BG     = {"LOW": "#f0fdf4", "MEDIUM": "#fffbeb", "HIGH": "#fef2f2"}
RISK_BORDER = {"LOW": "#86efac", "MEDIUM": "#fcd34d", "HIGH": "#fca5a5"}
MONTH_ABB   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nepal Flood Intelligence Center",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }
.block-container { padding: 1.6rem 2rem 2.5rem 2rem; max-width: 1400px; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #94a3b8 !important; font-size:0.75rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:2px; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent;
    border-radius: 6px;
    padding: 7px 12px;
    cursor: pointer;
    transition: background 0.15s;
    font-size: 0.875rem !important;
    font-weight: 500;
    color: #cbd5e1 !important;
    display: flex; align-items: center; gap: 8px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover { background:#1e293b; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div + span { color:#38bdf8 !important; }
[data-testid="stSidebar"] hr { border-color: #1e293b; margin: 10px 0; }

/* ── Page heading ── */
.page-hero { margin-bottom: 1.5rem; }
.page-hero h1 { font-size: 1.75rem; font-weight: 800; color: #0f172a; margin:0; line-height:1.2; }
.page-hero p  { color: #64748b; font-size: 0.9rem; margin: 4px 0 0 0; }

/* ── Status pill ── */
.status-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:#f0fdf4; border:1px solid #86efac;
    color:#16a34a; font-size:0.75rem; font-weight:600;
    padding:3px 10px; border-radius:20px; letter-spacing:.04em;
}

/* ── KPI cards ── */
.kpi-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: "";
    position: absolute; top:0; left:0; right:0; height:3px;
    background: #1d4ed8;
    border-radius: 12px 12px 0 0;
}
.kpi-icon  { font-size:1.4rem; margin-bottom:8px; display:block; }
.kpi-label { font-size:0.7rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { font-size:1.75rem; font-weight:800; color:#0f172a; line-height:1.1; margin:4px 0; }
.kpi-desc  { font-size:0.72rem; color:#94a3b8; }

.kpi-card.green::before  { background: #16a34a; }
.kpi-card.amber::before  { background: #d97706; }
.kpi-card.red::before    { background: #dc2626; }
.kpi-card.purple::before { background: #7c3aed; }

/* ── Risk status card ── */
.risk-status-card {
    border-radius: 14px;
    padding: 28px 32px;
    border: 1px solid;
    position: relative;
}
.risk-class-label {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: .06em;
    line-height: 1;
}
.risk-score-text {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 6px;
}

/* ── Section titles ── */
.sec-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 4px 0;
}
.sec-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0 0 16px 0;
}

/* ── Alert cards ── */
.alert-card {
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid;
    margin-bottom: 10px;
    background: #fff;
}
.alert-header { font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: .05em; }
.alert-body   { font-size: 0.82rem; color: #475569; margin-top: 4px; line-height:1.5; }

/* ── Timeline ── */
.timeline-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.82rem;
}
.timeline-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.timeline-date { color:#64748b; min-width: 80px; }
.timeline-station { font-weight:600; color:#0f172a; flex:1; }
.timeline-badge {
    font-size:0.7rem; font-weight:700; padding:2px 8px;
    border-radius:10px; letter-spacing:.04em;
}

/* ── Insight cards ── */
.insight-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    font-size: 0.84rem;
    color: #334155;
    line-height: 1.6;
}
.insight-title { font-weight:700; color:#0f172a; font-size:0.88rem; margin-bottom:4px; }

/* ── Station table badges ── */
.badge-low    { background:#dcfce7; color:#166534; padding:2px 8px; border-radius:10px; font-size:0.72rem; font-weight:700; }
.badge-medium { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:10px; font-size:0.72rem; font-weight:700; }
.badge-high   { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:10px; font-size:0.72rem; font-weight:700; }

/* ── Prediction result ── */
.pred-result-card {
    border-radius: 14px;
    padding: 30px;
    text-align: center;
    border: 2px solid;
}
.pred-risk-label { font-size:3rem; font-weight:900; letter-spacing:.06em; }
.pred-prob-text  { font-size:1.1rem; font-weight:600; margin-top:8px; }
.pred-msg        { font-size:0.85rem; color:#475569; margin-top:6px; }

/* ── Factor bar ── */
.factor-row { margin-bottom:10px; }
.factor-name { font-size:0.8rem; font-weight:600; color:#334155; margin-bottom:3px; }
.factor-bar-bg { background:#f1f5f9; border-radius:4px; height:8px; width:100%; }
.factor-bar-fill { background:#1d4ed8; border-radius:4px; height:8px; }
.factor-pct  { font-size:0.75rem; color:#64748b; margin-top:2px; }

/* ── Metric cards (model perf) ── */
.mcard {
    background:#fff; border:1px solid #e2e8f0; border-radius:10px;
    padding:18px; text-align:center;
}
.mcard-val   { font-size:1.8rem; font-weight:800; color:#1d4ed8; }
.mcard-label { font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:.07em; margin-top:4px; }

/* ── Footer ── */
.app-footer {
    margin-top: 3rem;
    padding: 20px 0 8px 0;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    color: #94a3b8;
    font-size: 0.78rem;
    line-height: 1.8;
}

/* ── Divider ── */
.section-divider { border: none; border-top: 1px solid #f1f5f9; margin: 24px 0; }

/* ── Streamlit overrides ── */
div[data-testid="metric-container"] { display: none; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom:2px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] {
    border-radius:8px 8px 0 0; padding:8px 16px;
    font-size:0.85rem; font-weight:600; color:#64748b;
    background:transparent; border:none;
}
.stTabs [aria-selected="true"] { color:#1d4ed8 !important; border-bottom:2px solid #1d4ed8; }
.stForm { background:transparent; border:none; padding:0; }
div[data-testid="stFormSubmitButton"] > button {
    background: #1d4ed8;
    color: white;
    font-weight: 700;
    border-radius: 8px;
    padding: 10px 0;
    font-size: 0.95rem;
    border: none;
    width: 100%;
}
div[data-testid="stFormSubmitButton"] > button:hover { background:#1e40af; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Initialising flood intelligence system...")
def load_data():
    raw = load_raw()
    clean, thresholds = preprocess(raw)
    return clean, thresholds

@st.cache_data(show_spinner=False)
def get_stats(df):
    return get_dataset_stats(df)


# ═════════════════════════════════════════════════════════════════════════════
# Helper renderers
# ═════════════════════════════════════════════════════════════════════════════
def kpi_card(icon, label, value, desc, accent="blue"):
    cls = {"blue":"","green":"green","amber":"amber","red":"red","purple":"purple"}
    return f"""
<div class="kpi-card {cls.get(accent,'')}">
  <span class="kpi-icon">{icon}</span>
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-desc">{desc}</div>
</div>"""

def risk_badge_html(level):
    c = RISK_COLOR[level]
    bg = RISK_BG[level]
    bd = RISK_BORDER[level]
    return (f'<span style="background:{bg};color:{c};border:1px solid {bd};'
            f'padding:2px 9px;border-radius:10px;font-size:0.72rem;font-weight:700;'
            f'letter-spacing:.04em;">{level}</span>')

def plotly_defaults(fig, height=300):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,Segoe UI,system-ui", size=11, color="#334155"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(showgrid=False, linecolor="#e2e8f0", linewidth=1),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    sb = st.sidebar
    sb.markdown("""
<div style="padding:20px 16px 12px 16px;">
  <div style="font-size:1.2rem;font-weight:800;color:#f8fafc;letter-spacing:.01em;">
    🌊 Nepal Flood Intelligence
  </div>
  <div style="font-size:0.72rem;color:#94a3b8;margin-top:3px;font-weight:500;">
    AI-Powered Early Warning System
  </div>
</div>
""", unsafe_allow_html=True)

    sb.markdown("<hr>", unsafe_allow_html=True)

    sb.markdown('<p style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;padding:0 16px;margin-bottom:4px;">Overview</p>', unsafe_allow_html=True)
    page = sb.radio("nav", [
        "📊  Dashboard",
        "🎯  Flood Risk Prediction",
        "📈  Data Analysis",
        "🤖  Model Performance",
        "📡  Station Monitoring",
        "📉  Risk Trends",
        "🗺   Regional Insights",
    ], label_visibility="collapsed")

    sb.markdown("<hr>", unsafe_allow_html=True)

    best = get_best_model_name() if models_trained() else "Not trained"
    sb.markdown(f"""
<div style="padding:0 4px;font-size:0.78rem;color:#94a3b8;line-height:2.2;">
  <div><span style="color:#475569;">Model</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       <span style="color:#38bdf8;font-weight:600;">{best}</span></div>
  <div><span style="color:#475569;">Data Coverage</span>&nbsp;
       <span style="color:#e2e8f0;font-weight:500;">2023 – 2026</span></div>
  <div><span style="color:#475569;">Stations</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       <span style="color:#e2e8f0;font-weight:500;">10 active</span></div>
  <div><span style="color:#475569;">Rivers</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
       <span style="color:#e2e8f0;font-weight:500;">9 monitored</span></div>
</div>
""", unsafe_allow_html=True)

    sb.markdown("<hr>", unsafe_allow_html=True)
    sb.markdown('<p style="font-size:0.68rem;color:#334155;text-align:center;padding-bottom:8px;">AI/ML · Weather Intelligence · Flood Analytics</p>', unsafe_allow_html=True)

    page_key = page.split("  ", 1)[-1].strip()
    return page_key


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard / Command Center
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard(df, stats):
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

    # ── Hero ─────────────────────────────────────────────────────────────────
    col_hero, col_ts = st.columns([3, 1])
    with col_hero:
        st.markdown("""
<div class="page-hero">
  <h1>Nepal Flood Risk Intelligence</h1>
  <p>AI-powered monitoring and prediction of flood risk across Nepal's major river basins</p>
</div>""", unsafe_allow_html=True)
    with col_ts:
        st.markdown(f"""
<div style="text-align:right;padding-top:52px;">
  <span class="status-pill">&#9679; SYSTEM OPERATIONAL</span><br>
  <span style="font-size:0.72rem;color:#94a3b8;display:block;margin-top:6px;">Last Updated: {now}</span>
</div>""", unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    cols = st.columns(7)
    kpi_data = [
        ("📋", "Total Observations", f"{stats['total_records']:,}", "Daily records 2023–2026", "blue"),
        ("📡", "Monitoring Stations", str(stats["locations"]), "Active hydro stations", "blue"),
        ("🌊", "Rivers Monitored", str(stats["rivers"]), "Major river basins", "blue"),
        ("🌧", "Avg Precipitation", f"{stats['avg_precipitation']} mm", "Daily average", "blue"),
        ("🟢", "Low Risk Days", f"{stats['low_risk_pct']}%", "Normal conditions", "green"),
        ("🟠", "Medium Risk Days", f"{stats['medium_risk_pct']}%", "Elevated conditions", "amber"),
        ("🔴", "High Risk Days", f"{stats['high_risk_pct']}%", "Flood-risk conditions", "red"),
    ]
    for col, (icon, label, value, desc, accent) in zip(cols, kpi_data):
        col.markdown(kpi_card(icon, label, value, desc, accent), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Current Risk Status + Distribution ──────────────────────────────────
    col_status, col_dist, col_monthly = st.columns([1.1, 1.4, 1.6])

    with col_status:
        st.markdown('<p class="sec-title">Current Risk Status</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Based on most recent observations</p>', unsafe_allow_html=True)

        # Use the most recent date's majority risk
        latest = df[df["date"] == df["date"].max()]["flood_risk"].mode()[0]
        risk_label = FLOOD_LABELS[latest]
        rc = RISK_COLOR[risk_label]
        rbg = RISK_BG[risk_label]
        rbd = RISK_BORDER[risk_label]
        risk_score = round(df[df["date"] == df["date"].max()]["flood_risk"].mean() / 2, 2)

        msgs = {"LOW": "Conditions currently stable across monitored stations.",
                "MEDIUM": "Elevated conditions detected. Enhanced monitoring advised.",
                "HIGH": "High-risk conditions detected. Immediate attention required."}
        icons = {"LOW": "&#10003;", "MEDIUM": "&#9888;", "HIGH": "&#9888;"}

        st.markdown(f"""
<div class="risk-status-card" style="background:{rbg};border-color:{rbd};">
  <div style="font-size:0.7rem;font-weight:700;color:{rc};text-transform:uppercase;letter-spacing:.1em;">
    Current Flood Risk
  </div>
  <div class="risk-class-label" style="color:{rc};margin:10px 0 6px 0;">{risk_label}</div>
  <div class="risk-score-text">Risk Index: <strong>{risk_score:.2f}</strong></div>
  <div style="margin-top:14px;font-size:0.82rem;color:{rc};font-weight:500;">
    {icons[risk_label]} {msgs[risk_label]}
  </div>
  <div style="margin-top:10px;font-size:0.72rem;color:#94a3b8;">
    As of {df['date'].max().strftime('%d %b %Y')}
  </div>
</div>""", unsafe_allow_html=True)

    with col_dist:
        st.markdown('<p class="sec-title">Flood Risk Distribution</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Breakdown across all observations</p>', unsafe_allow_html=True)
        vc = df["flood_risk"].value_counts().sort_index()
        labels_d = [FLOOD_LABELS[i] for i in vc.index]
        colors_d = [RISK_COLOR[FLOOD_LABELS[i]] for i in vc.index]
        fig = go.Figure(go.Pie(
            labels=labels_d, values=vc.values,
            hole=0.62,
            marker=dict(colors=colors_d, line=dict(color="#fff", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>{stats['total_records']:,}</b><br><span style='font-size:10px'>records</span>",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=13))
        plotly_defaults(fig, 260)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1),
                          margin=dict(l=0, r=0, t=10, b=20))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("""
<div style="font-size:0.75rem;color:#64748b;line-height:1.6;padding:4px 0;">
Most observations fall within the low-risk category. Elevated-risk
observations represent a smaller but operationally critical portion.
</div>""", unsafe_allow_html=True)

    with col_monthly:
        st.markdown('<p class="sec-title">Seasonal Flood Risk Pattern</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Monthly distribution of observed risk levels</p>', unsafe_allow_html=True)
        df["risk_label"] = df["flood_risk"].map(FLOOD_LABELS)
        mr = df.groupby(["month", "risk_label"]).size().unstack(fill_value=0)
        mr = mr.reindex(columns=["LOW", "MEDIUM", "HIGH"], fill_value=0).reset_index()
        fig = go.Figure()
        for lvl, col_c in zip(["LOW", "MEDIUM", "HIGH"], [C_LOW, C_MED, C_HIGH]):
            if lvl in mr.columns:
                fig.add_trace(go.Bar(
                    name=lvl, x=[MONTH_ABB[m-1] for m in mr["month"]],
                    y=mr[lvl], marker_color=col_c,
                    hovertemplate=f"<b>{lvl}</b><br>Month: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
                ))
        fig.update_layout(barmode="stack", showlegend=True,
                          legend=dict(orientation="h", y=1.08, x=0))
        plotly_defaults(fig, 290)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Precipitation Intelligence ────────────────────────────────────────────
    st.markdown('<p class="sec-title">Precipitation Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Monthly precipitation trends and variability</p>', unsafe_allow_html=True)

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.markdown(kpi_card("🌧", "Avg Daily Precip", f"{stats['avg_precipitation']} mm", "Dataset mean", "blue"), unsafe_allow_html=True)
    col_p2.markdown(kpi_card("⬆", "Max Recorded", f"{df['precipitation_mm'].max():.1f} mm", "Single-day peak", "red"), unsafe_allow_html=True)
    col_p3.markdown(kpi_card("📊", "Std Deviation", f"{df['precipitation_mm'].std():.2f} mm", "Variability measure", "purple"), unsafe_allow_html=True)
    col_p4.markdown(kpi_card("🗓", "Heavy Rain Days", f"{(df['precipitation_mm'] > 20).sum():,}", "Days > 20 mm", "amber"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    monthly_p = df.groupby("month")["precipitation_mm"].mean().reset_index()
    monthly_p["month_name"] = monthly_p["month"].apply(lambda m: MONTH_ABB[m-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_p["month_name"], y=monthly_p["precipitation_mm"],
        fill="tozeroy", fillcolor="rgba(29,78,216,0.1)",
        line=dict(color="#1d4ed8", width=2.5),
        mode="lines+markers", marker=dict(size=6, color="#1d4ed8"),
        hovertemplate="<b>%{x}</b><br>Avg Precipitation: %{y:.2f} mm<extra></extra>",
    ))
    plotly_defaults(fig, 220)
    fig.update_layout(title=dict(text="Monthly Average Precipitation (mm)", font=dict(size=12), x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Station Network ───────────────────────────────────────────────────────
    st.markdown('<p class="sec-title">Monitoring Network</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Status of all 10 hydrological monitoring stations</p>', unsafe_allow_html=True)

    station_df = (
        df.groupby("location").agg(
            River=("river", "first"),
            Basin=("basin", "first"),
            Avg_Precip=("precipitation_mm", "mean"),
            Avg_Temp=("temperature_mean_c", "mean"),
            Avg_Discharge=("river_discharge_m3s", "mean"),
            High_Risk_Pct=("flood_risk", lambda x: round((x==2).mean()*100, 1)),
            Dominant_Risk=("flood_risk", lambda x: FLOOD_LABELS[x.mode()[0]]),
        ).reset_index().rename(columns={"location": "Station"})
    )
    station_df["Avg_Precip"] = station_df["Avg_Precip"].round(2)
    station_df["Avg_Temp"]   = station_df["Avg_Temp"].round(1)
    station_df["Avg_Discharge"] = station_df["Avg_Discharge"].round(2)

    rows_html = ""
    for _, row in station_df.sort_values("High_Risk_Pct", ascending=False).iterrows():
        lvl = row["Dominant_Risk"]
        badge_cls = {"LOW": "badge-low", "MEDIUM": "badge-medium", "HIGH": "badge-high"}[lvl]
        rows_html += f"""
<tr style="border-bottom:1px solid #f1f5f9;">
  <td style="padding:10px 12px;font-weight:600;color:#0f172a;">{row['Station']}</td>
  <td style="padding:10px 12px;color:#475569;">{row['River']}</td>
  <td style="padding:10px 12px;color:#475569;">{row['Basin']}</td>
  <td style="padding:10px 12px;"><span class="{badge_cls}">{lvl}</span></td>
  <td style="padding:10px 12px;color:#334155;">{row['Avg_Precip']} mm</td>
  <td style="padding:10px 12px;color:#334155;">{row['Avg_Temp']} °C</td>
  <td style="padding:10px 12px;color:#334155;">{row['Avg_Discharge']} m³/s</td>
  <td style="padding:10px 12px;color:#dc2626;font-weight:600;">{row['High_Risk_Pct']}%</td>
</tr>"""

    st.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
<table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
<thead>
<tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
  <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;">Station</th>
  <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;">River</th>
  <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;">Basin</th>
  <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;">Status</th>
  <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;">Avg Precip</th>
  <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;">Avg Temp</th>
  <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;">Avg Discharge</th>
  <th style="padding:10px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;">High Risk %</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Key Insights ─────────────────────────────────────────────────────────
    st.markdown('<p class="sec-title">Key Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Data-driven observations from the monitoring dataset</p>', unsafe_allow_html=True)

    # Compute insights dynamically
    peak_month_idx = df.groupby("month")["precipitation_mm"].mean().idxmax()
    peak_month_name = MONTH_ABB[peak_month_idx - 1]
    high_station = station_df.loc[station_df["High_Risk_Pct"].idxmax(), "Station"]
    high_station_pct = station_df["High_Risk_Pct"].max()
    monsoon_high_pct = round(df[df["month"].isin([6,7,8,9]) & (df["flood_risk"]==2)].shape[0] /
                              max(df[df["flood_risk"]==2].shape[0], 1) * 100, 1)
    corr_val = round(df[["precipitation_mm","flood_risk"]].corr().iloc[0,1], 3)

    ins = [
        ("Seasonal Concentration",
         f"{monsoon_high_pct}% of all HIGH-risk observations occur during the monsoon season "
         f"(June–September), confirming the strong seasonal flood cycle across Nepal's river basins."),
        ("Peak Precipitation Month",
         f"Average daily precipitation peaks in <b>{peak_month_name}</b>, aligning with the "
         f"monsoon onset. This month consistently records the highest flood-risk frequencies."),
        ("Highest-Risk Station",
         f"<b>{high_station}</b> records the highest proportion of HIGH-risk days "
         f"({high_station_pct}%), indicating elevated structural flood exposure at this location."),
        ("Precipitation–Risk Correlation",
         f"Precipitation and flood risk show a positive correlation of <b>{corr_val}</b>. "
         f"Extended periods of elevated precipitation are strongly associated with increased flood-risk classification."),
    ]
    cols_ins = st.columns(4)
    icons_ins = ["🌧", "📅", "📡", "📊"]
    for col, (title, body), icon in zip(cols_ins, ins, icons_ins):
        col.markdown(f"""
<div class="insight-card">
  <div class="insight-title">{icon} {title}</div>
  <div style="font-size:0.81rem;color:#475569;line-height:1.6;">{body}</div>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Flood Risk Prediction
# ═════════════════════════════════════════════════════════════════════════════
def page_prediction():
    st.markdown("""
<div class="page-hero">
  <h1>Flood Risk Prediction</h1>
  <p>Estimate flood-risk levels using the trained Random Forest classification model</p>
</div>""", unsafe_allow_html=True)

    if not models_trained():
        st.error("Model artifacts not found. Run `python src/train_model.py` first.")
        return

    locations = get_known_locations()
    meta = get_display_feature_meta()

    with st.form("prediction_form"):
        # ── Location & Date ────────────────────────────────────────────────
        st.markdown('<p class="sec-title">Location & Date</p>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            location = st.selectbox("Monitoring Station", options=locations, index=4)
        with col_b:
            pred_date = st.date_input("Observation Date", value=datetime.date.today(),
                                      min_value=datetime.date(2023,1,1),
                                      max_value=datetime.date(2030,12,31))
        with col_c:
            month_disp = pred_date.strftime("%B") if hasattr(pred_date, "strftime") else ""
            season_disp = month_to_season(pred_date.month)
            st.markdown(f"""
<div style="padding-top:8px;">
  <div style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">
    Season / Month
  </div>
  <div style="font-size:1rem;font-weight:700;color:#0f172a;">{season_disp}</div>
  <div style="font-size:0.8rem;color:#64748b;">{month_disp}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Weather Conditions ─────────────────────────────────────────────
        st.markdown('<p class="sec-title">Weather Conditions</p>', unsafe_allow_html=True)
        weather_keys = ["precipitation_mm", "rain_mm", "temperature_mean_c",
                        "dew_point_mean_c", "relative_humidity_mean_pct", "precipitation_hours"]
        w_cols = st.columns(3)
        inputs = {}
        for i, key in enumerate(weather_keys):
            m = meta[key]
            with w_cols[i % 3]:
                if isinstance(m["default"], int) and isinstance(m["step"], int):
                    inputs[key] = st.number_input(m["label"],
                        min_value=int(m["min"]), max_value=int(m["max"]),
                        value=int(m["default"]), step=int(m["step"]))
                else:
                    inputs[key] = st.number_input(m["label"],
                        min_value=float(m["min"]), max_value=float(m["max"]),
                        value=float(m["default"]), step=float(m["step"]),
                        format="%.3f" if m["step"] < 0.01 else "%.1f")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Environmental Conditions ───────────────────────────────────────
        st.markdown('<p class="sec-title">Environmental & Wind Conditions</p>', unsafe_allow_html=True)
        env_keys = ["soil_moisture_0_100cm_m3m3", "wind_speed_max_kmh",
                    "wind_gusts_max_kmh", "wind_direction_dominant_deg", "elevation_m"]
        e_cols = st.columns(3)
        for i, key in enumerate(env_keys):
            m = meta[key]
            with e_cols[i % 3]:
                if isinstance(m["default"], int) and isinstance(m["step"], int):
                    inputs[key] = st.number_input(m["label"],
                        min_value=int(m["min"]), max_value=int(m["max"]),
                        value=int(m["default"]), step=int(m["step"]))
                else:
                    inputs[key] = st.number_input(m["label"],
                        min_value=float(m["min"]), max_value=float(m["max"]),
                        value=float(m["default"]), step=float(m["step"]),
                        format="%.3f" if m["step"] < 0.01 else "%.1f")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Flood Risk Assessment", use_container_width=True, type="primary")

    # ── Result ────────────────────────────────────────────────────────────────
    if submitted:
        month = pred_date.month
        doy   = pred_date.timetuple().tm_yday
        with st.spinner("Running model inference..."):
            result = predict(inputs, location, month=month, day_of_year=doy)

        st.markdown("<br>", unsafe_allow_html=True)
        rl    = result["risk_level"]
        rc    = RISK_COLOR[rl]
        rbg   = RISK_BG[rl]
        rbd   = RISK_BORDER[rl]
        rprob = result["probability"]
        probs = result["probabilities"]

        col_r, col_factors = st.columns([1, 1.5])

        with col_r:
            msgs2 = {
                "LOW":    ("Conditions Stable",    "River discharge is projected to remain within normal bounds for this station."),
                "MEDIUM": ("Elevated Conditions",  "River discharge may reach elevated levels. Enhanced monitoring is recommended."),
                "HIGH":   ("High Risk Detected",   "River discharge may exceed the 90th percentile threshold. Precautionary review advised."),
            }
            m_title, m_body = msgs2[rl]
            st.markdown(f"""
<div class="pred-result-card" style="background:{rbg};border-color:{rc};">
  <div style="font-size:0.72rem;font-weight:700;color:{rc};text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">
    Flood Risk Assessment
  </div>
  <div class="pred-risk-label" style="color:{rc};">{rl}</div>
  <div class="pred-prob-text" style="color:{rc};">Confidence: {rprob*100:.1f}%</div>
  <hr style="border-color:{rbd};margin:16px 0;">
  <div style="font-weight:700;font-size:0.85rem;color:{rc};">{m_title}</div>
  <div class="pred-msg">{m_body}</div>
  <div style="margin-top:14px;font-size:0.72rem;color:#94a3b8;">
    Station: {location} &nbsp;|&nbsp; {pred_date.strftime('%d %b %Y')} &nbsp;|&nbsp; {month_to_season(month)}
  </div>
  <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">
    Model-based risk indicator. Not an official emergency warning.
  </div>
</div>""", unsafe_allow_html=True)

            # Probability bars
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.82rem;font-weight:700;color:#334155;margin-bottom:8px;">Class Probability Distribution</p>', unsafe_allow_html=True)
            fig_p = go.Figure(go.Bar(
                x=[p*100 for p in probs[:3]],
                y=["LOW", "MEDIUM", "HIGH"],
                orientation="h",
                marker_color=[C_LOW, C_MED, C_HIGH],
                text=[f"{p*100:.1f}%" for p in probs[:3]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            plotly_defaults(fig_p, 160)
            fig_p.update_layout(margin=dict(l=10,r=60,t=10,b=10), xaxis=dict(range=[0,105], title="Probability (%)"))
            st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

        with col_factors:
            st.markdown('<p class="sec-title">Why This Prediction?</p>', unsafe_allow_html=True)
            st.markdown('<p class="sec-sub">Feature importance from the trained Random Forest model</p>', unsafe_allow_html=True)

            if result["top_factors"]:
                total_imp = sum(v for _, v in result["top_factors"])
                bars_html = ""
                for feat, imp in result["top_factors"]:
                    pct = round(imp / max(total_imp, 1e-9) * 100, 1)
                    bars_html += f"""
<div class="factor-row">
  <div class="factor-name">{pretty_feature(feat)}</div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div class="factor-bar-bg" style="flex:1;">
      <div class="factor-bar-fill" style="width:{pct}%;"></div>
    </div>
    <div class="factor-pct" style="min-width:38px;text-align:right;">{pct}%</div>
  </div>
</div>"""
                st.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;">
{bars_html}
<div style="margin-top:14px;padding-top:12px;border-top:1px solid #f1f5f9;
            font-size:0.72rem;color:#94a3b8;line-height:1.6;">
Feature importance reflects how strongly each variable contributes to model 
predictions across the training dataset. It does not establish direct causation.
</div>
</div>""", unsafe_allow_html=True)

            # Input recap
            with st.expander("View input parameters"):
                inp_df = pd.DataFrame(
                    [(meta[k]["label"], v) for k, v in inputs.items()],
                    columns=["Parameter", "Value"])
                st.dataframe(inp_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Data Analysis
# ═════════════════════════════════════════════════════════════════════════════
def page_data_analysis(df):
    st.markdown("""
<div class="page-hero">
  <h1>Data Analysis</h1>
  <p>Interactive exploration of Nepal's flood and weather dataset (2023–2026)</p>
</div>""", unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<p class="sec-title">Filters</p>', unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            all_locs = ["All Stations"] + sorted(df["location"].unique().tolist())
            sel_loc = st.selectbox("Station", all_locs)
        with col_f2:
            year_range = st.slider("Year Range", int(df["year"].min()), int(df["year"].max()),
                                   (int(df["year"].min()), int(df["year"].max())))
        with col_f3:
            risk_filter = st.multiselect("Risk Level", ["LOW","MEDIUM","HIGH"],
                                          default=["LOW","MEDIUM","HIGH"])

    fdf = df.copy()
    fdf["risk_label"] = fdf["flood_risk"].map(FLOOD_LABELS)
    if sel_loc != "All Stations":
        fdf = fdf[fdf["location"] == sel_loc]
    fdf = fdf[(fdf["year"] >= year_range[0]) & (fdf["year"] <= year_range[1])]
    if risk_filter:
        fdf = fdf[fdf["risk_label"].isin(risk_filter)]

    st.caption(f"Showing {len(fdf):,} records after filters")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab_weather, tab_hydro, tab_risk, tab_corr = st.tabs([
        "Weather Analysis", "Hydrological Analysis", "Risk Analysis", "Correlations"
    ])

    # ── TAB: Weather ─────────────────────────────────────────────────────────
    with tab_weather:
        c1, c2 = st.columns(2)
        with c1:
            m_p = fdf.groupby("month")["precipitation_mm"].mean().reindex(range(1,13), fill_value=0)
            fig = go.Figure(go.Bar(x=MONTH_ABB, y=m_p.values, marker_color="#1d4ed8",
                hovertemplate="<b>%{x}</b><br>Avg: %{y:.2f} mm<extra></extra>"))
            fig.update_layout(title="Monthly Avg Precipitation (mm)")
            plotly_defaults(fig, 280)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            m_t = fdf.groupby("month")["temperature_mean_c"].mean().reindex(range(1,13), fill_value=np.nan)
            m_d = fdf.groupby("month")["dew_point_mean_c"].mean().reindex(range(1,13), fill_value=np.nan)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=MONTH_ABB, y=m_t.values, mode="lines+markers",
                name="Temperature", line=dict(color="#dc2626", width=2),
                hovertemplate="%{x}: %{y:.1f}°C<extra>Temp</extra>"))
            fig.add_trace(go.Scatter(x=MONTH_ABB, y=m_d.values, mode="lines+markers",
                name="Dew Point", line=dict(color="#2563eb", width=2, dash="dot"),
                hovertemplate="%{x}: %{y:.1f}°C<extra>Dew Point</extra>"))
            fig.update_layout(title="Temperature & Dew Point (°C)", yaxis_title="°C")
            plotly_defaults(fig, 280)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2)
        with c3:
            m_h = fdf.groupby("month")["relative_humidity_mean_pct"].mean().reindex(range(1,13), fill_value=np.nan)
            fig = go.Figure(go.Scatter(x=MONTH_ABB, y=m_h.values,
                fill="tozeroy", fillcolor="rgba(37,99,235,0.1)",
                line=dict(color="#2563eb", width=2), mode="lines+markers",
                hovertemplate="%{x}: %{y:.1f}%<extra>Humidity</extra>"))
            fig.update_layout(title="Monthly Mean Relative Humidity (%)", yaxis=dict(range=[0,105]))
            plotly_defaults(fig, 260)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c4:
            m_w = fdf.groupby("month")["wind_speed_max_kmh"].mean().reindex(range(1,13), fill_value=0)
            fig = go.Figure(go.Bar(x=MONTH_ABB, y=m_w.values, marker_color="#7c3aed",
                hovertemplate="%{x}: %{y:.1f} km/h<extra>Wind</extra>"))
            fig.update_layout(title="Monthly Max Wind Speed (km/h)")
            plotly_defaults(fig, 260)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── TAB: Hydrological ─────────────────────────────────────────────────────
    with tab_hydro:
        # Discharge over time
        if sel_loc == "All Stations":
            fig = go.Figure()
            for loc in fdf["location"].unique():
                sub = fdf[fdf["location"] == loc].sort_values("date")
                fig.add_trace(go.Scatter(x=sub["date"], y=sub["river_discharge_m3s"],
                    mode="lines", name=loc, line=dict(width=1),
                    hovertemplate=f"<b>{loc}</b><br>%{{x|%d %b %Y}}<br>%{{y:.2f}} m³/s<extra></extra>"))
        else:
            sub = fdf.sort_values("date")
            fig = go.Figure(go.Scatter(x=sub["date"], y=sub["river_discharge_m3s"],
                mode="lines", fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
                line=dict(color="#7c3aed", width=1.5),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} m³/s<extra></extra>"))
        fig.update_layout(title="River Discharge Over Time (m³/s)")
        plotly_defaults(fig, 320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2)
        with c1:
            loc_d = fdf.groupby("location")["river_discharge_m3s"].mean().sort_values()
            fig = go.Figure(go.Bar(x=loc_d.values, y=loc_d.index, orientation="h",
                marker_color="#7c3aed",
                hovertemplate="<b>%{y}</b><br>%{x:.2f} m³/s<extra></extra>"))
            fig.update_layout(title="Avg Discharge by Station (m³/s)")
            plotly_defaults(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            fig = px.scatter(fdf.sample(min(3000, len(fdf))),
                x="precipitation_mm", y="river_discharge_m3s",
                color="risk_label",
                color_discrete_map={"LOW": C_LOW, "MEDIUM": C_MED, "HIGH": C_HIGH},
                log_y=True, opacity=0.5, size_max=5,
                labels={"precipitation_mm": "Precipitation (mm)",
                        "river_discharge_m3s": "Discharge (m³/s)", "risk_label": "Risk"},
                title="Precipitation vs Discharge (coloured by risk)")
            plotly_defaults(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── TAB: Risk Analysis ────────────────────────────────────────────────────
    with tab_risk:
        c1, c2 = st.columns(2)
        with c1:
            mr = fdf.groupby(["month","risk_label"]).size().unstack(fill_value=0)
            mr = mr.reindex(columns=["LOW","MEDIUM","HIGH"], fill_value=0).reset_index()
            fig = go.Figure()
            for lvl, col_c in zip(["LOW","MEDIUM","HIGH"], [C_LOW, C_MED, C_HIGH]):
                if lvl in mr.columns:
                    fig.add_trace(go.Bar(name=lvl, x=[MONTH_ABB[m-1] for m in mr["month"]],
                        y=mr[lvl], marker_color=col_c))
            fig.update_layout(barmode="stack", title="Flood Risk by Month")
            plotly_defaults(fig, 300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            loc_r = fdf.groupby("location")["flood_risk"].mean().sort_values()
            fig = go.Figure(go.Bar(x=loc_r.values, y=loc_r.index, orientation="h",
                marker_color=[C_LOW if v < 0.5 else (C_MED if v < 1.0 else C_HIGH) for v in loc_r.values],
                hovertemplate="<b>%{y}</b><br>Avg Risk Index: %{x:.3f}<extra></extra>"))
            fig.update_layout(title="Average Risk Index by Station")
            plotly_defaults(fig, 300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # HIGH risk timeline
        high = fdf[fdf["flood_risk"] == 2].copy()
        if len(high) > 0:
            high["month_year"] = high["date"].dt.to_period("M").astype(str)
            mh = high.groupby("month_year").size().reset_index(name="count")
            fig = go.Figure(go.Bar(x=mh["month_year"], y=mh["count"],
                marker_color=C_HIGH,
                hovertemplate="<b>%{x}</b><br>%{y} HIGH-risk events<extra></extra>"))
            fig.update_layout(title="Monthly HIGH-Risk Event Frequency")
            plotly_defaults(fig, 260)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── TAB: Correlations ────────────────────────────────────────────────────
    with tab_corr:
        num_cols = ["precipitation_mm","rain_mm","soil_moisture_0_100cm_m3m3",
                    "temperature_mean_c","dew_point_mean_c","relative_humidity_mean_pct",
                    "wind_speed_max_kmh","river_discharge_m3s","flood_risk"]
        corr = fdf[num_cols].corr().round(2)
        labels_c = [pretty_feature(c) for c in num_cols]
        fig = go.Figure(go.Heatmap(z=corr.values, x=labels_c, y=labels_c,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.values, texttemplate="%{text:.2f}",
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        plotly_defaults(fig, 520)
        fig.update_layout(title="Feature Correlation Heatmap",
                          margin=dict(l=120, r=20, t=40, b=120))
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════
def page_model_performance():
    st.markdown("""
<div class="page-hero">
  <h1>Model Performance</h1>
  <p>Evaluation and comparison of all trained flood-risk classifiers</p>
</div>""", unsafe_allow_html=True)

    if not models_trained():
        st.error("Model artifacts not found. Run `python src/train_model.py` first.")
        return

    best_name = get_best_model_name()
    metrics   = load_metrics()
    fi_data   = load_feature_importance()

    # ── Model card ────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:18px 22px;margin-bottom:20px;">
  <div style="font-size:0.72rem;font-weight:700;color:#1d4ed8;text-transform:uppercase;letter-spacing:.08em;">Selected Model</div>
  <div style="font-size:1.3rem;font-weight:800;color:#0f172a;margin-top:4px;">{best_name}</div>
  <div style="font-size:0.82rem;color:#64748b;margin-top:4px;">
    Selected by highest weighted F1-score on a time-based hold-out test set (last 20% chronologically).
    SMOTE was applied to the training set to address class imbalance.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    if best_name in metrics:
        m = metrics[best_name]
        mc = st.columns(5)
        for col, (label, val) in zip(mc, [
            ("Accuracy",      f"{m['accuracy']*100:.2f}%"),
            ("Precision",     f"{m['precision']*100:.2f}%"),
            ("Recall",        f"{m['recall']*100:.2f}%"),
            ("F1 (weighted)", f"{m['f1_weighted']*100:.2f}%"),
            ("ROC-AUC",       f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A"),
        ]):
            col.markdown(f'<div class="mcard"><div class="mcard-val">{val}</div><div class="mcard-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Comparison ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        st.markdown('<p class="sec-title">Model Comparison</p>', unsafe_allow_html=True)
        if metrics:
            names  = list(metrics.keys())
            f1s    = [metrics[n]["f1_weighted"]*100 for n in names]
            accs   = [metrics[n]["accuracy"]*100 for n in names]
            aucs   = [metrics[n].get("roc_auc", 0)*100 for n in names]
            cols_m = ["#7c3aed" if n == best_name else "#93c5fd" for n in names]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="F1 (weighted)", x=names, y=f1s, marker_color=cols_m,
                hovertemplate="<b>%{x}</b><br>F1: %{y:.2f}%<extra></extra>"))
            fig.add_trace(go.Scatter(name="ROC-AUC ×100", x=names, y=aucs,
                mode="markers+lines", marker=dict(size=8, color="#dc2626"),
                line=dict(color="#dc2626", dash="dot"),
                hovertemplate="<b>%{x}</b><br>AUC×100: %{y:.2f}<extra></extra>"))
            fig.update_layout(title="F1 Score & ROC-AUC by Model", yaxis=dict(range=[70, 100]))
            plotly_defaults(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown('<p class="sec-title">Metrics Table</p>', unsafe_allow_html=True)
        rows = []
        for mn, mv in metrics.items():
            rows.append({
                "Model":    mn,
                "Acc":      f"{mv['accuracy']*100:.2f}%",
                "Prec":     f"{mv['precision']*100:.2f}%",
                "Recall":   f"{mv['recall']*100:.2f}%",
                "F1":       f"{mv['f1_weighted']*100:.2f}%",
                "AUC":      f"{mv['roc_auc']:.4f}" if mv.get("roc_auc") else "N/A",
                "Best":     "✓" if mn == best_name else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Confusion matrix + Feature importance ─────────────────────────────────
    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown('<p class="sec-title">Confusion Matrix</p>', unsafe_allow_html=True)
        cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        st.markdown("""
<div style="font-size:0.78rem;color:#64748b;line-height:1.7;margin-top:8px;">
Rows = Actual class &nbsp;|&nbsp; Columns = Predicted class<br>
<b>LOW</b>: discharge &lt; station p75 &nbsp;|&nbsp;
<b>MEDIUM</b>: p75–p90 &nbsp;|&nbsp;
<b>HIGH</b>: &ge; p90<br>
SMOTE was applied to training data only; test set is unmodified.
</div>""", unsafe_allow_html=True)

    with col_fi:
        st.markdown('<p class="sec-title">Feature Importance (Top 15)</p>', unsafe_allow_html=True)
        if fi_data:
            fi_names = [pretty_feature(k) for k in fi_data.keys()]
            fi_vals  = [v*100 for v in fi_data.values()]
            fig = go.Figure(go.Bar(x=fi_vals, y=fi_names, orientation="h",
                marker_color="#1d4ed8",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}%<extra></extra>"))
            plotly_defaults(fig, 420)
            fig.update_layout(margin=dict(l=10, r=20, t=20, b=10),
                              yaxis=dict(autorange="reversed"),
                              xaxis=dict(title="Importance (%)"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Model summary ──────────────────────────────────────────────────────────
    st.markdown('<p class="sec-title">Model Summary</p>', unsafe_allow_html=True)
    summary_cols = st.columns(6)
    for col, (label, val) in zip(summary_cols, [
        ("Algorithm",   best_name),
        ("Features",    "31"),
        ("Training Set","10,712"),
        ("Test Set",    "2,678"),
        ("Classes",     "LOW / MED / HIGH"),
        ("Imbalance Fix","SMOTE"),
    ]):
        col.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;text-align:center;">
  <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.07em;">{label}</div>
  <div style="font-size:0.88rem;font-weight:700;color:#0f172a;margin-top:4px;">{val}</div>
</div>""", unsafe_allow_html=True)

    with st.expander("Methodology & Design Decisions"):
        st.markdown("""
**Time-Aware Split:** The last 20% of chronologically sorted records form the test set, simulating real-world forecasting.

**SMOTE:** Synthetic Minority Oversampling applied only to the training set to balance the three risk classes (75% LOW / 15% MEDIUM / 10% HIGH).

**Target Variable:** Flood risk is derived from per-station river discharge percentiles (p75 and p90), ensuring rivers of different scales are treated fairly.

**Feature Engineering:** Rolling 3-day and 7-day windows for precipitation, rainfall and humidity; cyclical month encoding; saturation index; temperature–dew point spread.

**Evaluation:** Weighted F1-score is the primary selection criterion given class imbalance. ROC-AUC (one-vs-rest, weighted) is also reported.
""")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Station Monitoring
# ═════════════════════════════════════════════════════════════════════════════
def page_station_monitoring(df):
    st.markdown("""
<div class="page-hero">
  <h1>Station Monitoring</h1>
  <p>Detailed operational view of each hydrological monitoring station</p>
</div>""", unsafe_allow_html=True)

    locations = sorted(df["location"].unique().tolist())
    sel = st.selectbox("Select Station", locations)
    sdf = df[df["location"] == sel].sort_values("date")

    # Station header card
    river  = sdf["river"].iloc[0]
    basin  = sdf["basin"].iloc[0]
    lat    = round(sdf["latitude"].iloc[0], 4)
    lon    = round(sdf["longitude"].iloc[0], 4)
    elev   = int(sdf["elevation_m"].iloc[0])
    latest_risk = FLOOD_LABELS[sdf["flood_risk"].iloc[-1]]
    rc = RISK_COLOR[latest_risk]
    rbg = RISK_BG[latest_risk]

    st.markdown(f"""
<div style="background:{rbg};border:1px solid {RISK_BORDER[latest_risk]};border-radius:12px;padding:18px 24px;margin-bottom:20px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;">
  <div>
    <div style="font-size:1.2rem;font-weight:800;color:#0f172a;">{sel}</div>
    <div style="font-size:0.85rem;color:#64748b;margin-top:2px;">River: <b>{river}</b> &nbsp;|&nbsp; Basin: <b>{basin}</b></div>
    <div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;">
      Lat: {lat} &nbsp; Lon: {lon} &nbsp; Elevation: {elev} m
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:0.68rem;font-weight:700;color:{rc};text-transform:uppercase;letter-spacing:.08em;">Latest Risk Status</div>
    <div style="font-size:1.8rem;font-weight:900;color:{rc};">{latest_risk}</div>
    <div style="font-size:0.72rem;color:#94a3b8;">As of {sdf['date'].iloc[-1].strftime('%d %b %Y')}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # KPIs for this station
    kpi_cols = st.columns(5)
    kpis = [
        ("📅", "Total Records",    f"{len(sdf):,}",                           "Daily observations"),
        ("🌧", "Avg Precipitation", f"{sdf['precipitation_mm'].mean():.2f} mm","Daily average"),
        ("🌡", "Avg Temperature",   f"{sdf['temperature_mean_c'].mean():.1f} °C","Daily mean"),
        ("💧", "Avg Discharge",     f"{sdf['river_discharge_m3s'].mean():.2f} m³/s","River flow"),
        ("🔴", "High Risk Days",    f"{(sdf['flood_risk']==2).sum():,}",        f"{(sdf['flood_risk']==2).mean()*100:.1f}% of records"),
    ]
    for col, (icon, label, value, desc) in zip(kpi_cols, kpis):
        col.markdown(kpi_card(icon, label, value, desc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=sdf["date"], y=sdf["river_discharge_m3s"],
            mode="lines", fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
            line=dict(color="#7c3aed", width=1.5),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} m³/s<extra></extra>"))
        plotly_defaults(fig, 300)
        fig.update_layout(title=f"River Discharge — {sel}")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig = go.Figure(go.Scatter(x=sdf["date"], y=sdf["precipitation_mm"],
            mode="lines", fill="tozeroy", fillcolor="rgba(29,78,216,0.1)",
            line=dict(color="#1d4ed8", width=1.2),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} mm<extra></extra>"))
        plotly_defaults(fig, 300)
        fig.update_layout(title=f"Daily Precipitation — {sel}")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Monthly risk breakdown
    sdf_cp = sdf.copy()
    sdf_cp["risk_label"] = sdf_cp["flood_risk"].map(FLOOD_LABELS)
    mr = sdf_cp.groupby(["month","risk_label"]).size().unstack(fill_value=0)
    mr = mr.reindex(columns=["LOW","MEDIUM","HIGH"], fill_value=0).reset_index()
    fig = go.Figure()
    for lvl, col_c in zip(["LOW","MEDIUM","HIGH"], [C_LOW, C_MED, C_HIGH]):
        if lvl in mr.columns:
            fig.add_trace(go.Bar(name=lvl, x=[MONTH_ABB[m-1] for m in mr["month"]],
                y=mr[lvl], marker_color=col_c))
    fig.update_layout(barmode="stack", title=f"Monthly Risk Distribution — {sel}")
    plotly_defaults(fig, 280)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Risk Trends  (Early Warning Center + Timeline)
# ═════════════════════════════════════════════════════════════════════════════
def page_risk_trends(df):
    st.markdown("""
<div class="page-hero">
  <h1>Early Warning Center</h1>
  <p>Risk alerts, historical trends and event timeline derived from observed data</p>
</div>""", unsafe_allow_html=True)

    # ── Current Alerts ────────────────────────────────────────────────────────
    st.markdown('<p class="sec-title">Current Station Alerts</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Model/data-based risk indicators — not official emergency warnings</p>', unsafe_allow_html=True)

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date][["location","river","flood_risk","precipitation_mm","river_discharge_m3s"]].copy()
    latest_df["risk_label"] = latest_df["flood_risk"].map(FLOOD_LABELS)
    latest_df = latest_df.sort_values("flood_risk", ascending=False)

    alert_cols = st.columns(2)
    alert_msgs = {
        "HIGH":   ("HIGH RISK ALERT",  C_HIGH, "#fee2e2", "#fca5a5",
                   "Elevated precipitation and discharge conditions detected. Enhanced monitoring recommended."),
        "MEDIUM": ("WATCH",            C_MED,  "#fffbeb", "#fcd34d",
                   "River discharge is above normal levels. Monitor conditions closely."),
        "LOW":    ("NORMAL CONDITIONS",C_LOW,  "#f0fdf4", "#86efac",
                   "Hydrological conditions are within normal operational range."),
    }
    for i, (_, row) in enumerate(latest_df.iterrows()):
        lvl = row["risk_label"]
        title, ac, abg, abd, amsg = alert_msgs[lvl]
        with alert_cols[i % 2]:
            st.markdown(f"""
<div class="alert-card" style="background:{abg};border-left-color:{ac};">
  <div class="alert-header" style="color:{ac};">{title}</div>
  <div style="font-size:0.82rem;font-weight:600;color:#334155;margin-top:6px;">
    {row['location']} &nbsp;·&nbsp; {row['river']}
  </div>
  <div class="alert-body">{amsg}</div>
  <div style="font-size:0.72rem;color:#94a3b8;margin-top:6px;">
    Precip: {row['precipitation_mm']:.1f} mm &nbsp;|&nbsp;
    Discharge: {row['river_discharge_m3s']:.2f} m³/s &nbsp;|&nbsp;
    {latest_date.strftime('%d %b %Y')}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Risk Trend Lines ──────────────────────────────────────────────────────
    st.markdown('<p class="sec-title">Risk Trend Over Time</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Rolling 30-day average flood risk index by station</p>', unsafe_allow_html=True)

    sel_stations = st.multiselect("Stations", sorted(df["location"].unique()), default=sorted(df["location"].unique())[:4])
    if sel_stations:
        fig = go.Figure()
        for loc in sel_stations:
            sub = df[df["location"] == loc].sort_values("date")
            rolling = sub["flood_risk"].rolling(30, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=sub["date"], y=rolling, mode="lines",
                name=loc, line=dict(width=1.8),
                hovertemplate=f"<b>{loc}</b><br>%{{x|%d %b %Y}}<br>Risk Index: %{{y:.3f}}<extra></extra>"))
        fig.add_hline(y=0.75, line_dash="dot", line_color=C_MED, annotation_text="MEDIUM threshold",
                      annotation_font_size=10)
        fig.add_hline(y=1.5,  line_dash="dot", line_color=C_HIGH, annotation_text="HIGH threshold",
                      annotation_font_size=10)
        plotly_defaults(fig, 360)
        fig.update_layout(yaxis_title="Risk Index (0=LOW, 1=MED, 2=HIGH)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Alert Timeline ────────────────────────────────────────────────────────
    st.markdown('<p class="sec-title">Risk Alert Timeline</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Most recent HIGH and MEDIUM risk observations across all stations</p>', unsafe_allow_html=True)

    recent = df[df["flood_risk"] >= 1].sort_values("date", ascending=False).head(30)
    timeline_html = ""
    for _, row in recent.iterrows():
        lvl = FLOOD_LABELS[row["flood_risk"]]
        c   = RISK_COLOR[lvl]
        bg  = RISK_BG[lvl]
        badge_cls = "badge-medium" if lvl == "MEDIUM" else "badge-high"
        timeline_html += f"""
<div class="timeline-row">
  <div class="timeline-dot" style="background:{c};"></div>
  <div class="timeline-date">{row['date'].strftime('%d %b %Y')}</div>
  <div class="timeline-station">{row['location']} — {row['river']}</div>
  <div style="color:#64748b;font-size:0.78rem;">
    {row['precipitation_mm']:.1f} mm &nbsp;|&nbsp; {row['river_discharge_m3s']:.1f} m³/s
  </div>
  <span class="{badge_cls}">{lvl}</span>
</div>"""

    st.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px 20px;max-height:460px;overflow-y:auto;">
{timeline_html}
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Regional Insights
# ═════════════════════════════════════════════════════════════════════════════
def page_regional_insights(df):
    st.markdown("""
<div class="page-hero">
  <h1>Regional Insights</h1>
  <p>Cross-station and basin-level analysis of flood risk patterns</p>
</div>""", unsafe_allow_html=True)

    # Basin-level summary
    st.markdown('<p class="sec-title">Basin-Level Risk Profile</p>', unsafe_allow_html=True)
    basin_df = df.groupby("basin").agg(
        Stations=("location", "nunique"),
        Records=("date", "count"),
        Avg_Precip=("precipitation_mm", "mean"),
        Avg_Discharge=("river_discharge_m3s", "mean"),
        High_Risk_Pct=("flood_risk", lambda x: round((x==2).mean()*100, 1)),
        Med_Risk_Pct =("flood_risk", lambda x: round((x==1).mean()*100, 1)),
    ).reset_index().rename(columns={"basin":"Basin"})
    basin_df["Avg_Precip"]   = basin_df["Avg_Precip"].round(2)
    basin_df["Avg_Discharge"] = basin_df["Avg_Discharge"].round(2)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(basin_df.sort_values("High_Risk_Pct"),
            x="High_Risk_Pct", y="Basin", orientation="h",
            color="High_Risk_Pct",
            color_continuous_scale=["#dcfce7","#fef3c7","#fee2e2"],
            labels={"High_Risk_Pct":"High Risk %","Basin":"Basin"},
            title="High-Risk Day % by Basin")
        plotly_defaults(fig, 340)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig = px.scatter(basin_df, x="Avg_Precip", y="Avg_Discharge",
            size="Records", color="High_Risk_Pct",
            color_continuous_scale=["#16a34a","#d97706","#dc2626"],
            text="Basin",
            labels={"Avg_Precip":"Avg Precipitation (mm)","Avg_Discharge":"Avg Discharge (m³/s)"},
            title="Precipitation vs Discharge by Basin")
        fig.update_traces(textposition="top center", textfont=dict(size=9))
        plotly_defaults(fig, 340)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.dataframe(basin_df, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Elevation vs risk
    st.markdown('<p class="sec-title">Elevation & Risk Relationship</p>', unsafe_allow_html=True)
    elev_df = df.groupby("location").agg(
        Elevation=("elevation_m","first"),
        High_Risk_Pct=("flood_risk", lambda x: (x==2).mean()*100),
        Avg_Discharge=("river_discharge_m3s","mean"),
    ).reset_index()

    fig = px.scatter(elev_df, x="Elevation", y="High_Risk_Pct",
        size="Avg_Discharge", text="location",
        color="High_Risk_Pct",
        color_continuous_scale=["#16a34a","#d97706","#dc2626"],
        labels={"Elevation":"Elevation (m)","High_Risk_Pct":"High Risk %"},
        title="Station Elevation vs High-Risk Day Percentage")
    fig.update_traces(textposition="top center", textfont=dict(size=9))
    plotly_defaults(fig, 380)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Season × Basin heatmap
    st.markdown('<p class="sec-title">Seasonal Risk Heatmap by Basin</p>', unsafe_allow_html=True)
    df["season"] = df["month"].apply(month_to_season)
    heat = df.groupby(["basin","season"])["flood_risk"].mean().unstack(fill_value=0)
    season_order = ["Winter","Pre-Monsoon","Monsoon","Post-Monsoon"]
    heat = heat.reindex(columns=[s for s in season_order if s in heat.columns])

    fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
        colorscale=[[0,"#f0fdf4"],[0.5,"#fef3c7"],[1,"#fee2e2"]],
        text=heat.round(3).values, texttemplate="%{text}",
        hovertemplate="Basin: %{y}<br>Season: %{x}<br>Avg Risk: %{z:.3f}<extra></extra>",
        zmin=0, zmax=2,
    ))
    plotly_defaults(fig, 340)
    fig.update_layout(title="Average Risk Index: Basin × Season",
                      margin=dict(l=100, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════════════════
# Footer
# ═════════════════════════════════════════════════════════════════════════════
def render_footer():
    st.markdown("""
<div class="app-footer">
  <strong style="color:#475569;">Nepal Flood Risk Prediction &amp; Early Warning System</strong><br>
  AI/ML &nbsp;·&nbsp; Weather Intelligence &nbsp;·&nbsp; Flood Risk Analytics<br>
  Dataset: Nepal Flood &amp; Weather 2023–2026 &nbsp;·&nbsp; Model: Random Forest Classifier<br>
  <span style="color:#cbd5e1;">Built with Python · scikit-learn · Plotly · Streamlit</span>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main router
# ═════════════════════════════════════════════════════════════════════════════
def main():
    df, thresholds = load_data()
    stats = get_stats(df)
    page  = render_sidebar()

    if page == "Dashboard":
        page_dashboard(df, stats)
    elif page == "Flood Risk Prediction":
        page_prediction()
    elif page == "Data Analysis":
        page_data_analysis(df)
    elif page == "Model Performance":
        page_model_performance()
    elif page == "Station Monitoring":
        page_station_monitoring(df)
    elif page == "Risk Trends":
        page_risk_trends(df)
    elif page == "Regional Insights":
        page_regional_insights(df)

    render_footer()


if __name__ == "__main__":
    main()
