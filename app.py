"""
PCA-Based Compression of Multivariate Solar-Pump Sensor Streams
for Mining Site Operational Monitoring
Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import json
import math
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SolarPCA Monitor | Mining Site Operations",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — WHITE & YELLOW THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ---- Root overrides ---- */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* ---- Main background ---- */
.stApp { background: #FAFAF8; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #FFFDE7 100%) !important;
    border-right: 1px solid #FFF176;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #3D3C38 !important;
    font-weight: 800;
}

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E5E4DF;
    border-radius: 16px;
    padding: 16px 20px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    transition: all 0.25s ease;
}
[data-testid="metric-container"]:hover {
    border-color: #FFD700;
    box-shadow: 0 4px 20px rgba(255,215,0,0.2);
    transform: translateY(-2px);
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    color: #1A1A17 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    color: #9E9D96 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}
[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 100px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #E5E4DF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 100px !important;
    padding: 8px 20px !important;
    font-weight: 600;
    font-size: 0.9rem;
    color: #9E9D96 !important;
    background: transparent !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #FFD700 !important;
    color: #1A1A17 !important;
    box-shadow: 0 2px 10px rgba(255,215,0,0.4) !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: #FFD700 !important;
    color: #1A1A17 !important;
    border: none !important;
    border-radius: 100px !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 10px 28px !important;
    box-shadow: 0 4px 16px rgba(255,215,0,0.3) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: #FFB300 !important;
    box-shadow: 0 6px 24px rgba(255,215,0,0.45) !important;
    transform: translateY(-2px) !important;
}

/* ---- Sliders ---- */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: #FFD700 !important;
}
[data-testid="stSlider"] [data-baseweb="thumb"] {
    background: #FFD700 !important;
    border-color: #FFD700 !important;
}

/* ---- Selectbox / Number Input ---- */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div {
    border-radius: 10px !important;
    border-color: #E5E4DF !important;
}

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden;
    border: 1px solid #E5E4DF;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

/* ---- Progress bar ---- */
.stProgress > div > div {
    background: linear-gradient(90deg, #FFD700, #FFE57F) !important;
    border-radius: 100px !important;
}

/* ---- Section headers ---- */
.section-header {
    background: linear-gradient(135deg, #FFFDE7 0%, #FFFFFF 100%);
    border: 1px solid #FFF176;
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.section-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FFD700, #FFF176, #FFD700);
}
.section-tag {
    display: inline-block;
    background: #FFD700;
    color: #1A1A17;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 100px;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #1A1A17;
    margin: 0;
}
.section-desc {
    color: #9E9D96;
    font-size: 0.9rem;
    margin: 0.3rem 0 0;
}

/* ---- Regime badges ---- */
.badge-normal   { background:#DCFCE7; color:#16A34A; padding:3px 12px; border-radius:100px; font-weight:700; font-size:0.78rem; border:1px solid #BBF7D0; }
.badge-strained { background:#FEF3C7; color:#D97706; padding:3px 12px; border-radius:100px; font-weight:700; font-size:0.78rem; border:1px solid #FDE68A; }
.badge-anomaly  { background:#FEE2E2; color:#DC2626; padding:3px 12px; border-radius:100px; font-weight:700; font-size:0.78rem; border:1px solid #FCA5A5; }

/* ---- Info/warning boxes ---- */
.info-box {
    background: #FFFDE7;
    border: 1px solid #FFF176;
    border-left: 4px solid #FFD700;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #706F69;
}
.warn-box {
    background: #FEF3C7;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #706F69;
}
.danger-box {
    background: #FEE2E2;
    border: 1px solid #FCA5A5;
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #706F69;
}

/* ---- Dividers ---- */
hr { border-color: #FFF176 !important; }

/* ---- Plotly chart containers ---- */
[data-testid="stPlotlyChart"] > div {
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font_family="Outfit, sans-serif",
    paper_bgcolor="white",
    plot_bgcolor="#FAFAF8",
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=["#FFD700", "#94A3B8", "#60A5FA", "#4ADE80", "#F87171", "#C084FC"],
)

YELLOW_SCALE = [[0, "#FFFFFF"], [0.5, "#FFE57F"], [1.0, "#FFD700"]]

REGIME_COLORS = {
    "normal":   "#4ADE80",
    "strained": "#FBBF24",
    "anomaly":  "#F87171",
}

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_sensor_data(n_samples: int, noise_level: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 2 * np.pi, n_samples)
    solar_factor = 0.7 * np.sin(1.5 * t) + 0.3 * np.sin(4 * t)
    load_factor  = -0.5 * np.cos(t)      + 0.4 * np.sin(3 * t)
    water_factor = 0.4  * np.sin(0.8 * t + 1.2)

    strained_mask = (np.arange(n_samples) >= int(n_samples * 0.40)) & \
                    (np.arange(n_samples) <  int(n_samples * 0.55))
    anomaly_mask  = (np.arange(n_samples) >= int(n_samples * 0.78)) & \
                    (np.arange(n_samples) <  int(n_samples * 0.84))

    strain_boost = np.where(anomaly_mask, 1.4, np.where(strained_mask, 0.6, 0.0))

    def make(base, sf, lf_or_wf, factor, noise_amp, strain_sign, low, high):
        v = base + factor * lf_or_wf + rng.standard_normal(n_samples) * noise_amp * noise_level
        v += strain_sign * strain_boost
        return np.clip(v, low, high)

    df = pd.DataFrame({
        "Sample":          np.arange(1, n_samples + 1),
        "Irradiance_W_m2": np.clip(700 + 200*solar_factor + rng.standard_normal(n_samples)*80*noise_level - strain_boost*300, 100, 1050),
        "PV_Voltage_V":    np.clip(210 + 20*solar_factor  + rng.standard_normal(n_samples)*8*noise_level  - strain_boost*40,  100, 260),
        "PV_Current_A":    np.clip(10  + 3*solar_factor   + rng.standard_normal(n_samples)*1.2*noise_level- strain_boost*4,   1,   18),
        "Motor_RPM":       np.clip(2000+ 500*load_factor  + rng.standard_normal(n_samples)*120*noise_level+ strain_boost*400, 500, 3500),
        "Flow_Rate_L_min": np.clip(50  + 20*load_factor   + rng.standard_normal(n_samples)*5*noise_level  - strain_boost*20,  5,   100),
        "Vibration_g":     np.clip(1.5 + 0.5*np.abs(load_factor)+rng.standard_normal(n_samples)*0.3*noise_level+strain_boost*2.5, 0.2, 8),
        "Tank_Level_pct":  np.clip(60  + 20*water_factor  + rng.standard_normal(n_samples)*5*noise_level  - strain_boost*15,  10,  100),
        "Pressure_kPa":    np.clip(250 + 80*water_factor  + rng.standard_normal(n_samples)*20*noise_level + strain_boost*60,  50,  500),
        "Turbidity_NTU":   np.clip(10  + 5*water_factor   + rng.standard_normal(n_samples)*2*noise_level  + strain_boost*10,  0.5, 50),
    })

    regime = np.where(anomaly_mask, "anomaly", np.where(strained_mask, "strained", "normal"))
    df["Regime"] = regime
    return df

# ─────────────────────────────────────────────
# PCA COMPUTATION
# ─────────────────────────────────────────────
SENSOR_COLS = [
    "Irradiance_W_m2", "PV_Voltage_V", "PV_Current_A",
    "Motor_RPM", "Flow_Rate_L_min", "Vibration_g",
    "Tank_Level_pct", "Pressure_kPa", "Turbidity_NTU"
]

SENSOR_LABELS_SHORT = [
    "Irradiance", "PV Voltage", "PV Current",
    "Motor RPM", "Flow Rate", "Vibration",
    "Tank Level", "Pressure", "Turbidity"
]

@st.cache_data
def run_pca(df: pd.DataFrame, n_components: int, standardize: bool):
    X = df[SENSOR_COLS].values
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    # Hotelling T²
    lambdas = pca.explained_variance_
    t2 = np.sum((scores ** 2) / lambdas, axis=1)

    score_df = pd.DataFrame(
        scores,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    score_df["T2_Stat"] = t2
    score_df["Regime"]  = df["Regime"].values
    score_df["Sample"]  = df["Sample"].values

    return pca, score_df, X_scaled

# ─────────────────────────────────────────────
# SECTION HEADER HELPER
# ─────────────────────────────────────────────
def section_header(tag: str, title: str, desc: str):
    st.markdown(f"""
    <div class="section-header">
      <div class="section-tag">{tag}</div>
      <div class="section-title">{title}</div>
      <div class="section-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
      <div style="font-size:2.5rem">☀️</div>
      <div style="font-size:1.1rem; font-weight:900; color:#1A1A17; letter-spacing:-0.02em">
        SolarPCA<span style="color:#FFB300">Monitor</span>
      </div>
      <div style="font-size:0.72rem; color:#9E9D96; margin-top:4px; letter-spacing:0.06em; text-transform:uppercase">
        Mining Site Operations
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Data Settings")
    n_samples = st.slider("Number of Samples", 50, 500, 200, 10,
                          help="Total sensor readings to simulate")
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.3, 0.05,
                             help="Gaussian noise added to sensor signals")
    seed = st.number_input("Random Seed", 1, 9999, 42, 1,
                            help="For reproducible data generation")

    st.markdown("---")
    st.markdown("### 🔬 PCA Settings")
    n_components = st.slider("Principal Components", 1, 9, 3, 1,
                              help="Number of PCs to retain")
    standardize = st.checkbox("Standardize Features (z-score)", value=True,
                               help="Recommended: scales sensors to unit variance")

    st.markdown("---")
    st.markdown("### 🎛️ Display")
    show_all_pcs = st.checkbox("Show all PC comparisons", value=False)
    table_rows   = st.select_slider("Table rows per page", [10, 20, 50, 100], 20)

    st.markdown("---")
    if st.button("🔄 Regenerate Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style="margin-top:2rem; padding:1rem; background:rgba(255,215,0,0.1); border-radius:12px; border:1px solid #FFF176;">
      <div style="font-size:0.72rem; color:#9E9D96; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px">Research Project</div>
      <div style="font-size:0.8rem; color:#3D3C38; line-height:1.5">PCA-Based Compression of Multivariate Solar-Pump Sensor Streams for Mining Site Operational Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GENERATE DATA & RUN PCA
# ─────────────────────────────────────────────
df = generate_sensor_data(n_samples, noise_level, int(seed))
pca_model, score_df, X_scaled = run_pca(df, n_components, standardize)

cum_var = np.cumsum(pca_model.explained_variance_ratio_)
compression_pct = int((9 - n_components) / 9 * 100)
n_normal   = (df["Regime"] == "normal").sum()
n_strained = (df["Regime"] == "strained").sum()
n_anomaly  = (df["Regime"] == "anomaly").sum()

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, #FFFDE7 0%, #FFFFFF 50%, #FFF9C4 100%);
  border-radius: 24px;
  padding: 2.5rem 3rem;
  margin-bottom: 1.5rem;
  border: 1px solid #FFF176;
  position: relative;
  overflow: hidden;
">
  <div style="position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,#FFD700,#FFF176,#FFD700)"></div>
  <div style="display:inline-block;background:#FFD700;color:#1A1A17;font-size:0.72rem;font-weight:700;
    letter-spacing:0.12em;text-transform:uppercase;padding:4px 14px;border-radius:100px;margin-bottom:0.8rem">
    ● Mining Site Operations · Live Monitoring
  </div>
  <h1 style="font-size:2.2rem;font-weight:900;color:#1A1A17;margin:0 0 0.5rem;line-height:1.1;letter-spacing:-0.03em">
    PCA-Based Sensor<br/>
    <span style="background:linear-gradient(135deg,#FFB300,#FFD700,#FF8F00);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
      Stream Compression
    </span>
  </h1>
  <p style="color:#9E9D96;font-size:1rem;max-width:600px;margin:0;line-height:1.6">
    Multivariate analysis of solar-output, pump-load & water-level sensor streams 
    using Principal Component Analysis for mining site operational intelligence.
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("📡 Sensor Channels",   "9",         "3 subsystems")
c2.metric("🔬 PCA Components",    n_components, f"of 9 original")
c3.metric("📊 Variance Retained", f"{cum_var[-1]*100:.1f}%", "cumulative")
c4.metric("🗜️ Compression",       f"{compression_pct}%",     "feature reduction")
c5.metric("⚠️ Strained Samples",  n_strained,   f"{n_strained/n_samples*100:.0f}% of total")
c6.metric("🚨 Anomaly Samples",   n_anomaly,    f"{n_anomaly/n_samples*100:.0f}% of total")

st.markdown("<br/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "☀️  Sensor Streams",
    "🔬  PCA Engine",
    "📊  Visualizations",
    "🚦  Op Regimes",
    "📋  Data Table",
])

# ═══════════════════════════════════════════
# TAB 1 — SENSOR STREAMS
# ═══════════════════════════════════════════
with tab1:
    section_header(
        "Objective 01",
        "Multivariate Sensor Streams",
        "Real-time collection from 9 sensor channels across 3 operational subsystems"
    )

    # Regime background shading helper
    def add_regime_shading(fig, df, row=1, col=1):
        strained_ranges, anomaly_ranges = [], []
        in_strained = in_anomaly = False
        s_start = a_start = None
        for i, r in enumerate(df["Regime"]):
            if r == "strained" and not in_strained:
                in_strained = True; s_start = i
            elif r != "strained" and in_strained:
                strained_ranges.append((s_start, i)); in_strained = False
            if r == "anomaly" and not in_anomaly:
                in_anomaly = True; a_start = i
            elif r != "anomaly" and in_anomaly:
                anomaly_ranges.append((a_start, i)); in_anomaly = False
        for s, e in strained_ranges:
            fig.add_vrect(x0=s, x1=e, fillcolor="rgba(251,191,36,0.12)",
                          line_width=0, row=row, col=col)
        for s, e in anomaly_ranges:
            fig.add_vrect(x0=s, x1=e, fillcolor="rgba(239,68,68,0.12)",
                          line_width=0, row=row, col=col)
        return fig

    # ─── Solar Subsystem ───
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem">
      <span style="font-size:1.5rem">☀️</span>
      <div>
        <div style="font-weight:700;font-size:1rem;color:#1A1A17">Solar Output Subsystem</div>
        <div style="font-size:0.78rem;color:#9E9D96">Photovoltaic Array Monitoring</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fig_solar = make_subplots(rows=1, cols=3, shared_xaxes=False,
                               subplot_titles=["Irradiance (W/m²)", "PV Voltage (V)", "PV Current (A)"])
    solar_sensors = ["Irradiance_W_m2", "PV_Voltage_V", "PV_Current_A"]
    solar_colors  = ["#FFD700", "#FFF176", "#FFE082"]

    for idx, (col_name, color) in enumerate(zip(solar_sensors, solar_colors), 1):
        fig_solar.add_trace(go.Scatter(
            x=df["Sample"], y=df[col_name], mode="lines",
            line=dict(color=color, width=2), name=col_name.replace("_", " "),
            fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba") if "rgb" in color else f"{color}18",
            hovertemplate=f"<b>{col_name.replace('_',' ')}</b>: %{{y:.1f}}<extra></extra>"
        ), row=1, col=idx)
        add_regime_shading(fig_solar, df, row=1, col=idx)

    fig_solar.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False)
    fig_solar.update_xaxes(showgrid=False, title_text="Sample")
    fig_solar.update_yaxes(gridcolor="#F2F1EE")
    st.plotly_chart(fig_solar, use_container_width=True)

    # ─── Pump Subsystem ───
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem">
      <span style="font-size:1.5rem">⚙️</span>
      <div>
        <div style="font-weight:700;font-size:1rem;color:#1A1A17">Pump Load Subsystem</div>
        <div style="font-size:0.78rem;color:#9E9D96">Motor & Hydraulic Monitoring</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fig_pump = make_subplots(rows=1, cols=3, shared_xaxes=False,
                              subplot_titles=["Motor RPM", "Flow Rate (L/min)", "Vibration (g)"])
    pump_sensors = ["Motor_RPM", "Flow_Rate_L_min", "Vibration_g"]
    pump_colors  = ["#94A3B8", "#64748B", "#475569"]

    for idx, (col_name, color) in enumerate(zip(pump_sensors, pump_colors), 1):
        fig_pump.add_trace(go.Scatter(
            x=df["Sample"], y=df[col_name], mode="lines",
            line=dict(color=color, width=2), name=col_name.replace("_", " "),
            hovertemplate=f"<b>{col_name.replace('_',' ')}</b>: %{{y:.1f}}<extra></extra>"
        ), row=1, col=idx)
        add_regime_shading(fig_pump, df, row=1, col=idx)

    fig_pump.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False)
    fig_pump.update_xaxes(showgrid=False, title_text="Sample")
    fig_pump.update_yaxes(gridcolor="#F2F1EE")
    st.plotly_chart(fig_pump, use_container_width=True)

    # ─── Water Subsystem ───
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem">
      <span style="font-size:1.5rem">💧</span>
      <div>
        <div style="font-weight:700;font-size:1rem;color:#1A1A17">Water Level Subsystem</div>
        <div style="font-size:0.78rem;color:#9E9D96">Reservoir & Pressure Monitoring</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fig_water = make_subplots(rows=1, cols=3, shared_xaxes=False,
                               subplot_titles=["Tank Level (%)", "Pressure (kPa)", "Turbidity (NTU)"])
    water_sensors = ["Tank_Level_pct", "Pressure_kPa", "Turbidity_NTU"]
    water_colors  = ["#60A5FA", "#3B82F6", "#93C5FD"]

    for idx, (col_name, color) in enumerate(zip(water_sensors, water_colors), 1):
        fig_water.add_trace(go.Scatter(
            x=df["Sample"], y=df[col_name], mode="lines",
            line=dict(color=color, width=2), name=col_name.replace("_", " "),
            fill="tozeroy", fillcolor=f"rgba(96,165,250,0.08)",
            hovertemplate=f"<b>{col_name.replace('_',' ')}</b>: %{{y:.1f}}<extra></extra>"
        ), row=1, col=idx)
        add_regime_shading(fig_water, df, row=1, col=idx)

    fig_water.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False)
    fig_water.update_xaxes(showgrid=False, title_text="Sample")
    fig_water.update_yaxes(gridcolor="#F2F1EE")
    st.plotly_chart(fig_water, use_container_width=True)

    st.markdown("*Shaded regions: 🟡 Strained · 🔴 Anomaly*", unsafe_allow_html=False)

    st.markdown("---")
    # ─── Correlation Matrix ───
    st.markdown("#### 🔗 Sensor Correlation Matrix")
    st.caption("Pearson correlation coefficients — higher values indicate PCA compression potential")

    corr = df[SENSOR_COLS].corr()
    corr.index = SENSOR_LABELS_SHORT
    corr.columns = SENSOR_LABELS_SHORT

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=SENSOR_LABELS_SHORT,
        y=SENSOR_LABELS_SHORT,
        colorscale=[[0,"#EFF6FF"],[0.5,"#FFFFFF"],[1,"#FFD700"]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, family="JetBrains Mono"),
        hoverongaps=False,
        colorbar=dict(title="r", tickfont=dict(family="JetBrains Mono"))
    ))
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=420,
                            xaxis=dict(tickfont=dict(size=11)),
                            yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig_corr, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — PCA ENGINE
# ═══════════════════════════════════════════
with tab2:
    section_header(
        "Objective 02",
        "PCA Compression Engine",
        "Principal Component Analysis compressing 9 sensor features while retaining maximum operational variance"
    )

    col_v, col_e, col_s = st.columns([1.2, 1, 1])

    # ─── Variance Explained ───
    with col_v:
        st.markdown("##### 📊 Variance Explained per Component")
        evr = pca_model.explained_variance_ratio_
        for i, ratio in enumerate(evr):
            pc_lbl = f"PC{i+1}"
            pct = ratio * 100
            st.markdown(f"""
            <div style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;font-size:0.82rem;font-weight:600;margin-bottom:4px">
                <span style="color:#3D3C38">{pc_lbl}</span>
                <span style="color:#1A1A17;font-family:'JetBrains Mono',monospace">{pct:.2f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(ratio))

        st.markdown(f"""
        <div style="margin-top:1rem;padding:12px 16px;background:#FFFDE7;border:1px solid #FFF176;
          border-radius:12px;display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:0.85rem;font-weight:600;color:#706F69">Cumulative Variance</span>
          <span style="font-size:1.3rem;font-weight:800;color:#FFB300;
            font-family:'JetBrains Mono',monospace">{cum_var[-1]*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ─── Eigenvalue Spectrum ───
    with col_e:
        st.markdown("##### 📐 Eigenvalue Spectrum")
        eigenvals = pca_model.explained_variance_
        fig_eigen = go.Figure(go.Bar(
            x=[f"PC{i+1}" for i in range(n_components)],
            y=eigenvals,
            marker_color=["#FFD700" if i == 0 else f"rgba(255,215,0,{0.9-i*0.2})" for i in range(n_components)],
            marker_line_color="#FFB300",
            marker_line_width=1,
            text=[f"{v:.2f}" for v in eigenvals],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig_eigen.update_layout(**PLOTLY_LAYOUT, height=280,
                                 xaxis_title="Component", yaxis_title="Eigenvalue",
                                 yaxis=dict(gridcolor="#F2F1EE"),
                                 xaxis=dict(showgrid=False))
        st.plotly_chart(fig_eigen, use_container_width=True)

    # ─── Scree Plot ───
    with col_s:
        st.markdown("##### 📉 Scree Plot (Elbow Analysis)")
        # Estimate all 9 eigenvalues
        pca_full = PCA(n_components=min(9, len(SENSOR_COLS)))
        pca_full.fit(X_scaled)
        all_evals = pca_full.explained_variance_
        all_cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
        pcs = [f"PC{i+1}" for i in range(len(all_evals))]

        fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
        fig_scree.add_trace(go.Scatter(
            x=pcs, y=all_evals, mode="lines+markers",
            line=dict(color="#FFD700", width=3),
            marker=dict(size=9, color="#FFD700", line=dict(color="white", width=2)),
            name="Eigenvalue", fill="tozeroy", fillcolor="rgba(255,215,0,0.08)"
        ), secondary_y=False)
        fig_scree.add_trace(go.Scatter(
            x=pcs, y=all_cumvar, mode="lines",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            name="Cum. Variance (%)"
        ), secondary_y=True)
        fig_scree.add_vline(x=n_components - 1, line_color="#FFD700",
                             line_dash="dot", line_width=2)
        fig_scree.update_layout(**PLOTLY_LAYOUT, height=280,
                                  legend=dict(font=dict(size=10)))
        fig_scree.update_yaxes(title_text="Eigenvalue", gridcolor="#F2F1EE", secondary_y=False)
        fig_scree.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True)
        st.plotly_chart(fig_scree, use_container_width=True)

    st.markdown("---")
    # ─── Component Loadings ───
    st.markdown("##### 🔢 PCA Component Loadings (Feature Weights)")
    st.caption("Each bar shows how strongly a sensor contributes to that principal component")

    loadings = pd.DataFrame(
        pca_model.components_,
        columns=SENSOR_LABELS_SHORT,
        index=[f"PC{i+1}" for i in range(n_components)]
    ).T

    fig_load = go.Figure()
    pc_colors = ["#FFD700", "#FFF176", "#FFE082", "#FFCA28", "#FFB300"]
    for i in range(n_components):
        pc = f"PC{i+1}"
        fig_load.add_trace(go.Bar(
            name=pc,
            x=SENSOR_LABELS_SHORT,
            y=loadings[pc].abs(),
            marker_color=pc_colors[i % len(pc_colors)],
            marker_line_color="#FFB300", marker_line_width=0.5,
        ))

    fig_load.update_layout(
        **PLOTLY_LAYOUT, height=320, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        yaxis_title="|Loading|",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#F2F1EE"),
    )
    st.plotly_chart(fig_load, use_container_width=True)

    # ─── Stats Banner ───
    st.markdown("---")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("📐 Dimensionality", f"9 → {n_components}", f"{compression_pct}% reduction")
    b2.metric("📊 Variance Kept", f"{cum_var[-1]*100:.1f}%", "of original info")
    b3.metric("🗜️ Compression", f"{compression_pct}%", f"{9-n_components} dims removed")
    recon_err = float(np.mean((X_scaled - pca_model.inverse_transform(
        pca_model.transform(X_scaled))) ** 2))
    b4.metric("📉 Reconstruction Error", f"{recon_err:.4f}", "MSE (standardized)")

# ═══════════════════════════════════════════
# TAB 3 — VISUALIZATIONS
# ═══════════════════════════════════════════
with tab3:
    section_header(
        "Objective 03",
        "Reduced PCA Representations",
        "PCA-projected sensor data plotted in principal component space — revealing hidden operational structure"
    )

    regime_color_map = {"normal": "#4ADE80", "strained": "#FBBF24", "anomaly": "#F87171"}

    # ─── PC1 vs PC2 scatter ───
    st.markdown("##### 🗺️ PC1 vs PC2 — Operating Regime Map")
    full_df = pd.concat([df.reset_index(drop=True), score_df[["PC1","PC2"] + ([f"PC{i+1}" for i in range(2, n_components)] if n_components > 2 else []) + ["T2_Stat"]].reset_index(drop=True)], axis=1)

    fig_12 = px.scatter(
        full_df, x="PC1", y="PC2", color="Regime",
        color_discrete_map=regime_color_map,
        hover_data={"Irradiance_W_m2": ":.1f", "Motor_RPM": ":.0f",
                    "Flow_Rate_L_min": ":.1f", "Vibration_g": ":.2f", "T2_Stat": ":.2f"},
        custom_data=["Sample"],
        labels={"PC1": "PC1 Score", "PC2": "PC2 Score", "Regime": "Regime"},
        height=460
    )
    fig_12.update_traces(marker=dict(size=8, opacity=0.78,
                                      line=dict(width=0.8, color="white")))
    fig_12.update_layout(**PLOTLY_LAYOUT,
                          legend=dict(title="Operating Regime",
                                      font=dict(size=12),
                                      itemsizing="constant"))
    fig_12.update_xaxes(gridcolor="#F2F1EE", zerolinecolor="#E5E4DF")
    fig_12.update_yaxes(gridcolor="#F2F1EE", zerolinecolor="#E5E4DF")
    st.plotly_chart(fig_12, use_container_width=True)

    if n_components >= 3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### PC1 vs PC3")
            fig_13 = px.scatter(full_df, x="PC1", y="PC3", color="Regime",
                                color_discrete_map=regime_color_map, height=320)
            fig_13.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.6, color="white")))
            fig_13.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            fig_13.update_xaxes(gridcolor="#F2F1EE"); fig_13.update_yaxes(gridcolor="#F2F1EE")
            st.plotly_chart(fig_13, use_container_width=True)
        with col_b:
            st.markdown("##### PC2 vs PC3")
            fig_23 = px.scatter(full_df, x="PC2", y="PC3", color="Regime",
                                color_discrete_map=regime_color_map, height=320)
            fig_23.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.6, color="white")))
            fig_23.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            fig_23.update_xaxes(gridcolor="#F2F1EE"); fig_23.update_yaxes(gridcolor="#F2F1EE")
            st.plotly_chart(fig_23, use_container_width=True)

    # ─── PC Timeseries ───
    st.markdown("---")
    st.markdown("##### 📈 Principal Component Score Timeseries")
    fig_ts = go.Figure()
    pc_ts_colors = ["#FFD700", "#FFF9C4", "#FFE082"]
    for i in range(min(n_components, 3)):
        pc = f"PC{i+1}"
        fig_ts.add_trace(go.Scatter(
            x=score_df["Sample"], y=score_df[pc], mode="lines",
            name=f"{pc} Score",
            line=dict(color=pc_ts_colors[i], width=2.5 if i == 0 else 2),
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(255,215,0,0.08)",
        ))
    # Regime shading on timeseries
    for s, e in [(score_df["Sample"][df["Regime"]=="strained"].min(), score_df["Sample"][df["Regime"]=="strained"].max())]:
        if not (isinstance(s, float) and math.isnan(s)):
            fig_ts.add_vrect(x0=s, x1=e, fillcolor="rgba(251,191,36,0.12)", line_width=0,
                             annotation_text="Strained", annotation_position="top left",
                             annotation_font_size=11)
    anomaly_samps = score_df["Sample"][df["Regime"] == "anomaly"]
    if len(anomaly_samps) > 0:
        fig_ts.add_vrect(x0=anomaly_samps.min(), x1=anomaly_samps.max(),
                         fillcolor="rgba(239,68,68,0.12)", line_width=0,
                         annotation_text="Anomaly", annotation_position="top left",
                         annotation_font_color="#EF4444", annotation_font_size=11)

    fig_ts.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis_title="Sample", yaxis_title="PC Score",
                          yaxis=dict(gridcolor="#F2F1EE"),
                          xaxis=dict(showgrid=False),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_ts, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 4 — OPERATING REGIMES
# ═══════════════════════════════════════════
with tab4:
    section_header(
        "Operational Intelligence",
        "Operating Regime Detection",
        "Automated classification of pump operating states using PCA score thresholds"
    )

    # Regime summary cards
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#F0FDF4,#DCFCE7);border:1px solid rgba(74,222,128,0.3);
          border-radius:20px;padding:1.4rem">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem">
            <div style="background:rgba(74,222,128,0.2);border-radius:12px;width:44px;height:44px;
              display:flex;align-items:center;justify-content:center;font-size:1.4rem">✅</div>
            <div>
              <div style="font-weight:700;color:#1A1A17">Normal Operation</div>
              <div style="font-size:0.75rem;color:#9E9D96">Stable equilibrium zone</div>
            </div>
            <div style="margin-left:auto;font-size:1.6rem;font-weight:800;color:#1A1A17;
              font-family:'JetBrains Mono',monospace">{n_normal/n_samples*100:.0f}%</div>
          </div>
          <div style="font-size:0.82rem;color:#706F69;line-height:1.6">
            System within designed parameters. Solar delivers rated output, 
            pump runs at optimal RPM, water levels stable.
          </div>
          <div style="margin-top:0.8rem;font-size:0.75rem;color:#16A34A;font-weight:600">
            {n_normal} samples · |PC1| &lt; 1.5σ · Vibration &lt; 2.5g
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#FFFBEB,#FEF3C7);border:1px solid rgba(251,191,36,0.4);
          border-radius:20px;padding:1.4rem">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem">
            <div style="background:rgba(251,191,36,0.2);border-radius:12px;width:44px;height:44px;
              display:flex;align-items:center;justify-content:center;font-size:1.4rem">⚠️</div>
            <div>
              <div style="font-weight:700;color:#1A1A17">Strained Operation</div>
              <div style="font-size:0.75rem;color:#9E9D96">Elevated stress zone</div>
            </div>
            <div style="margin-left:auto;font-size:1.6rem;font-weight:800;color:#1A1A17;
              font-family:'JetBrains Mono',monospace">{n_strained/n_samples*100:.0f}%</div>
          </div>
          <div style="font-size:0.82rem;color:#706F69;line-height:1.6">
            Solar output degraded. Pump compensates with increased load.
            Vibration rises — intervention recommended.
          </div>
          <div style="margin-top:0.8rem;font-size:0.75rem;color:#D97706;font-weight:600">
            {n_strained} samples · |PC1| 1.5–3.0σ · Vibration 2.5–5.0g
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#FFF5F5,#FEE2E2);border:1px solid rgba(239,68,68,0.3);
          border-radius:20px;padding:1.4rem">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem">
            <div style="background:rgba(239,68,68,0.2);border-radius:12px;width:44px;height:44px;
              display:flex;align-items:center;justify-content:center;font-size:1.4rem">🚨</div>
            <div>
              <div style="font-weight:700;color:#1A1A17">Anomaly Detected</div>
              <div style="font-size:0.75rem;color:#9E9D96">Critical out-of-distribution</div>
            </div>
            <div style="margin-left:auto;font-size:1.6rem;font-weight:800;color:#1A1A17;
              font-family:'JetBrains Mono',monospace">{n_anomaly/n_samples*100:.0f}%</div>
          </div>
          <div style="font-size:0.82rem;color:#706F69;line-height:1.6">
            Critical deviation from operational envelope. Potential equipment 
            failure or blockage. Immediate inspection required.
          </div>
          <div style="margin-top:0.8rem;font-size:0.75rem;color:#DC2626;font-weight:600">
            {n_anomaly} samples · |PC1| &gt; 3.0σ · T² &gt; 99th pct
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_d, col_t = st.columns(2)

    # ─── Regime Donut ───
    with col_d:
        st.markdown("##### 🍩 Regime Distribution")
        fig_donut = go.Figure(go.Pie(
            labels=["Normal", "Strained", "Anomaly"],
            values=[n_normal, n_strained, n_anomaly],
            hole=0.65,
            marker_colors=["#4ADE80", "#FBBF24", "#F87171"],
            marker_line=dict(color="white", width=3),
            textinfo="label+percent",
            textfont=dict(family="Outfit", size=12, color="#1A1A17"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(**PLOTLY_LAYOUT, height=320,
                                 annotations=[dict(text=f"{n_samples}<br><span style='font-size:0.7em'>samples</span>",
                                                   x=0.5, y=0.5, font_size=20, showarrow=False,
                                                   font=dict(family="JetBrains Mono", color="#1A1A17"))])
        st.plotly_chart(fig_donut, use_container_width=True)

    # ─── Hotelling T² ───
    with col_t:
        st.markdown("##### 📊 Hotelling T² Statistic")
        t2_vals = score_df["T2_Stat"].values
        threshold_95 = np.percentile(t2_vals, 95)
        threshold_99 = np.percentile(t2_vals, 99)

        bar_colors = [REGIME_COLORS.get(r, "#FFD700") for r in df["Regime"]]
        fig_t2 = go.Figure()
        fig_t2.add_trace(go.Bar(
            x=score_df["Sample"], y=t2_vals,
            marker_color=bar_colors, marker_line_width=0,
            name="T² Statistic",
            hovertemplate="Sample %{x}<br>T²: %{y:.2f}<extra></extra>",
        ))
        fig_t2.add_hline(y=threshold_95, line_color="#F59E0B", line_dash="dash", line_width=2,
                          annotation_text=f"95th pct: {threshold_95:.1f}", annotation_position="top right",
                          annotation_font=dict(color="#F59E0B", size=11))
        fig_t2.add_hline(y=threshold_99, line_color="#EF4444", line_dash="dash", line_width=2,
                          annotation_text=f"99th pct: {threshold_99:.1f}", annotation_position="top right",
                          annotation_font=dict(color="#EF4444", size=11))
        fig_t2.update_layout(**PLOTLY_LAYOUT, height=320,
                              xaxis_title="Sample", yaxis_title="T² Value",
                              yaxis=dict(gridcolor="#F2F1EE"),
                              xaxis=dict(showgrid=False))
        st.plotly_chart(fig_t2, use_container_width=True)

    # ─── Regime Timeline ───
    st.markdown("##### ⏱️ Regime Timeline — Operational State History")
    regime_num = df["Regime"].map({"normal": 0.33, "strained": 0.67, "anomaly": 1.0}).values
    color_seq  = df["Regime"].map(REGIME_COLORS).tolist()

    fig_timeline = go.Figure(go.Bar(
        x=df["Sample"], y=np.ones(len(df)),
        marker_color=color_seq, marker_line_width=0,
        hovertemplate="Sample %{x}<br>Regime: %{customdata}<extra></extra>",
        customdata=df["Regime"].values,
    ))
    fig_timeline.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
        height=100,
        xaxis_title="Sample",
        yaxis=dict(visible=False),
        xaxis=dict(showgrid=False),
        bargap=0,
        bargroupgap=0,
        margin=dict(l=20, r=20, t=10, b=30)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:1.5rem;font-size:0.8rem;font-weight:600">
      <span style="color:#4ADE80">● Normal</span>
      <span style="color:#FBBF24">● Strained</span>
      <span style="color:#F87171">● Anomaly</span>
    </div>
    """, unsafe_allow_html=True)

    # ─── Alert Log ───
    st.markdown("---")
    st.markdown("##### 🔔 Operational Alert Log")
    alerts = []
    prev_regime = None
    for i, row in df.iterrows():
        r = row["Regime"]
        if r == "anomaly" and prev_regime != "anomaly":
            alerts.append({"Level": "🚨 CRITICAL", "Sample": int(row["Sample"]),
                           "Message": f"Anomaly detected — T² exceeds 99th percentile. Immediate inspection required.",
                           "Sensor": f"Vibration: {row['Vibration_g']:.2f}g"})
        elif r == "strained" and prev_regime == "normal":
            alerts.append({"Level": "⚠️ WARNING", "Sample": int(row["Sample"]),
                           "Message": "Strained operation started — Solar irradiance drop, pump overcompensating.",
                           "Sensor": f"Irradiance: {row['Irradiance_W_m2']:.0f} W/m²"})
        prev_regime = r

    if alerts:
        alerts_df = pd.DataFrame(alerts)
        st.dataframe(alerts_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Level": st.column_config.TextColumn("Level", width="small"),
                         "Sample": st.column_config.NumberColumn("Sample #", format="%d"),
                         "Message": st.column_config.TextColumn("Alert", width="large"),
                         "Sensor": st.column_config.TextColumn("Key Sensor"),
                     })
    else:
        st.success("✅ No regime transitions detected in current dataset.")

# ═══════════════════════════════════════════
# TAB 5 — DATA TABLE
# ═══════════════════════════════════════════
with tab5:
    section_header(
        "Raw Dataset",
        "Sensor Data & PCA Scores",
        "Full multivariate readings with PCA-projected scores and regime labels"
    )

    # ─── Summary Pills ───
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("📋 Total Samples",   n_samples)
    p2.metric("✅ Normal",           n_normal,   f"{n_normal/n_samples*100:.0f}%")
    p3.metric("⚠️ Strained",         n_strained, f"{n_strained/n_samples*100:.0f}%")
    p4.metric("🚨 Anomaly",          n_anomaly,  f"{n_anomaly/n_samples*100:.0f}%")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ─── Filters ───
    fc1, fc2, fc3 = st.columns([2, 1.2, 1.2])
    with fc1:
        search_q = st.text_input("🔍 Search samples", placeholder="Type to filter...", label_visibility="collapsed")
    with fc2:
        regime_filter = st.selectbox("Regime Filter", ["All", "Normal", "Strained", "Anomaly"],
                                      label_visibility="collapsed")
    with fc3:
        view_mode = st.selectbox("View", ["Sensor Raw", "PCA Scores", "Combined"],
                                  label_visibility="collapsed")

    # Build combined dataframe
    combined_df = pd.concat([
        df.reset_index(drop=True),
        score_df[[f"PC{i+1}" for i in range(n_components)] + ["T2_Stat"]].reset_index(drop=True)
    ], axis=1)

    # Apply filters
    display_df = combined_df.copy()
    if regime_filter != "All":
        display_df = display_df[display_df["Regime"] == regime_filter.lower()]
    if search_q:
        mask = display_df.astype(str).apply(lambda row: search_q.lower() in " ".join(row).lower(), axis=1)
        display_df = display_df[mask]

    # Select columns by view
    if view_mode == "Sensor Raw":
        view_cols = ["Sample"] + SENSOR_COLS + ["Regime"]
    elif view_mode == "PCA Scores":
        view_cols = ["Sample"] + [f"PC{i+1}" for i in range(n_components)] + ["T2_Stat", "Regime"]
    else:
        view_cols = ["Sample", "Irradiance_W_m2", "Motor_RPM", "Flow_Rate_L_min", "Vibration_g",
                     "Tank_Level_pct"] + [f"PC{i+1}" for i in range(min(n_components,3))] + ["Regime"]
    display_df = display_df[[c for c in view_cols if c in display_df.columns]]

    # Rename for display
    rename_map = {
        "Irradiance_W_m2": "Irradiance (W/m²)", "PV_Voltage_V": "PV Voltage (V)",
        "PV_Current_A": "PV Current (A)", "Motor_RPM": "Motor RPM",
        "Flow_Rate_L_min": "Flow Rate (L/min)", "Vibration_g": "Vibration (g)",
        "Tank_Level_pct": "Tank Level (%)", "Pressure_kPa": "Pressure (kPa)",
        "Turbidity_NTU": "Turbidity (NTU)", "T2_Stat": "T² Statistic"
    }
    display_df = display_df.rename(columns=rename_map)

    st.caption(f"Showing **{len(display_df)}** of **{n_samples}** samples")

    # Column config
    col_cfg = {}
    for c in display_df.columns:
        if c == "Regime":
            col_cfg[c] = st.column_config.TextColumn("Regime", width="small")
        elif "PC" in c:
            col_cfg[c] = st.column_config.NumberColumn(c, format="%.3f", width="small")
        elif c == "T² Statistic":
            col_cfg[c] = st.column_config.NumberColumn(c, format="%.2f", width="small")
        elif c == "Sample":
            col_cfg[c] = st.column_config.NumberColumn(c, format="%d", width="small")
        else:
            col_cfg[c] = st.column_config.NumberColumn(c, format="%.2f")

    st.dataframe(display_df, use_container_width=True, hide_index=True,
                 height=min(600, 36 * min(len(display_df), table_rows) + 38),
                 column_config=col_cfg)

    # ─── Downloads ───
    st.markdown("##### ⬇️ Export Data")
    dl1, dl2, dl3 = st.columns(3)

    # Full CSV
    full_export = combined_df.copy()
    full_export = full_export.rename(columns=rename_map)
    csv_bytes = full_export.to_csv(index=False).encode("utf-8")
    with dl1:
        st.download_button(
            label="📥 Download CSV (All Data)",
            data=csv_bytes,
            file_name="solar_pump_pca_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # JSON
    json_data = {
        "metadata": {
            "total_samples": n_samples,
            "sensor_channels": 9,
            "pca_components": n_components,
            "cumulative_variance_retained": float(cum_var[-1]),
            "noise_level": noise_level,
            "seed": int(seed),
        },
        "data": combined_df.round(4).to_dict(orient="records")
    }
    json_bytes = json.dumps(json_data, indent=2).encode("utf-8")
    with dl2:
        st.download_button(
            label="📥 Download JSON",
            data=json_bytes,
            file_name="solar_pump_pca_data.json",
            mime="application/json",
            use_container_width=True,
        )

    # PCA Loadings
    loadings_df = pd.DataFrame(
        pca_model.components_,
        columns=SENSOR_LABELS_SHORT,
        index=[f"PC{i+1}" for i in range(n_components)]
    ).round(4)
    loadings_csv = loadings_df.to_csv().encode("utf-8")
    with dl3:
        st.download_button(
            label="📥 Download PCA Loadings",
            data=loadings_csv,
            file_name="pca_loadings.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    # ─── Descriptive Statistics ───
    st.markdown("##### 📐 Descriptive Statistics — All Sensor Channels")
    stats_df = df[SENSOR_COLS].describe().T.round(2)
    stats_df.index = SENSOR_LABELS_SHORT
    stats_df = stats_df[["min", "max", "mean", "std", "50%"]].rename(
        columns={"min": "Min", "max": "Max", "mean": "Mean",
                 "std": "Std Dev", "50%": "Median"})
    stats_df["Range"] = (stats_df["Max"] - stats_df["Min"]).round(2)
    st.dataframe(stats_df, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(c, format="%.2f") for c in stats_df.columns})
