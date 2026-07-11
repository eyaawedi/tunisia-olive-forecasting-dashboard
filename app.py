
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import folium
from streamlit_folium import st_folium

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="زيتون | Tunisia Olive Intelligence",
    page_icon="🫒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
C_RED       = "#5C8A3C"
C_DARKRED   = "#3A6325"
C_OLIVE     = "#2D5A1B"
C_OLIVE2    = "#6BAF45"
C_GOLD      = "#A8C96B"
C_GOLD2     = "#C8E08A"
C_EARTH     = "#4F7A28"
C_INK       = "#0D1F08"
C_CREAM     = "#F4F8EF"
C_PARCHMENT = "#E6F0D8"
C_MUTED     = "#6B8558"
C_PANEL     = "#FFFFFF"


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp {{
    font-family: 'DM Sans', sans-serif;
    background: {C_CREAM};
    color: {C_INK};
}}

section[data-testid="stSidebar"] {{
    background: {C_INK} !important;
    border-right: 1px solid rgba(201,162,39,0.25);
}}
section[data-testid="stSidebar"] > div {{ padding: 0 !important; }}
section[data-testid="stSidebar"] * {{ color: {C_CREAM} !important; }}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(201,162,39,0.3) !important;
    border-radius: 8px !important;
}}

.block-container {{ padding: 2rem 2.5rem 4rem 2.5rem !important; max-width: 1600px; }}

/* Hero */
.hero-wrap {{
    position: relative; background: {C_INK};
    border-radius: 28px; padding: 52px 56px 48px;
    margin-bottom: 24px; overflow: hidden;
}}
.hero-wrap::before {{
    content: ""; position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 110% 50%, rgba(201,162,39,0.18) 0%, transparent 65%),
        radial-gradient(ellipse 40% 60% at -10% 30%, rgba(200,16,46,0.22) 0%, transparent 55%);
    pointer-events: none;
}}
.hero-arabesque {{
    position: absolute; right: 52px; top: 50%;
    transform: translateY(-50%); font-size: 150px;
    line-height: 1; opacity: 0.07; color: {C_GOLD};
    pointer-events: none; user-select: none;
}}
.hero-arabic {{
    font-family: 'Playfair Display', serif; font-size: 12px;
    letter-spacing: 4px; color: {C_GOLD}; text-transform: uppercase;
    margin-bottom: 16px; display: block; font-style: italic;
}}
.hero-title {{
    font-family: 'Playfair Display', serif; font-size: 48px;
    font-weight: 900; color: white; line-height: 1.1; margin-bottom: 6px;
}}
.hero-title span {{ color: {C_GOLD}; font-style: italic; }}
.hero-sub {{
    color: rgba(255,255,255,0.68); font-size: 15px; max-width: 620px;
    line-height: 1.7; margin-top: 14px; font-weight: 300;
}}
.hero-tags {{ margin-top: 24px; display: flex; gap: 8px; flex-wrap: wrap; }}
.hero-tag {{
    padding: 5px 12px; border-radius: 999px;
    border: 1px solid rgba(201,162,39,0.45); color: {C_GOLD} !important;
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
    background: rgba(201,162,39,0.08); text-transform: uppercase;
}}

/* Sidebar */
.sb-logo {{
    padding: 28px 24px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 8px;
}}
.sb-logo-title {{
    font-family: 'Playfair Display', serif;
    font-size: 20px; font-weight: 700; color: white; margin-bottom: 2px;
}}
.sb-logo-arabic {{
    font-size: 10px; letter-spacing: 2.5px;
    color: {C_GOLD} !important; text-transform: uppercase; font-style: italic;
}}
.sb-section {{ padding: 14px 22px 8px; }}
.sb-section-label {{
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    color: rgba(255,255,255,0.35) !important; margin-bottom: 10px; font-weight: 600;
}}
.sb-formula {{
    background: rgba(201,162,39,0.1); border: 1px solid rgba(201,162,39,0.25);
    border-radius: 10px; padding: 12px 14px; margin: 8px 0;
}}
.sb-formula-main {{
    font-family: 'DM Mono', monospace; font-size: 12px;
    color: {C_GOLD} !important; font-weight: 500; line-height: 1.8;
}}
.sb-formula-sub {{
    font-size: 11px; color: rgba(255,255,255,0.4) !important;
    margin-top: 5px; line-height: 1.6;
}}

/* Smart filter pill bar */
.smart-filter-bar {{
    background: {C_INK};
    border-radius: 20px;
    padding: 18px 28px 20px;
    border: 1px solid rgba(201,162,39,0.2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}}
.smart-filter-bar::before {{
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 100% at 50% 120%, rgba(201,162,39,0.08) 0%, transparent 70%);
    pointer-events: none;
}}
.filter-title {{
    font-size: 10px; letter-spacing: 2.5px; text-transform: uppercase;
    color: rgba(201,162,39,0.8); font-weight: 700; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}}
.filter-label {{
    font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase;
    color: rgba(255,255,255,0.45); font-weight: 600; margin-bottom: 6px;
}}

/* KPI cards */
.kpi-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 18px; margin-bottom: 26px;
}}
.kpi-card {{
    background: {C_PANEL}; border-radius: 18px; padding: 24px 22px 20px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 18px rgba(0,0,0,0.05);
    position: relative; overflow: hidden;
}}
.kpi-card::after {{
    content: ""; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 18px 18px 0 0;
}}
.kpi-card.red::after   {{ background: linear-gradient(90deg, {C_RED}, {C_DARKRED}); }}
.kpi-card.olive::after {{ background: linear-gradient(90deg, {C_OLIVE}, {C_OLIVE2}); }}
.kpi-card.gold::after  {{ background: linear-gradient(90deg, {C_GOLD}, {C_GOLD2}); }}
.kpi-card.earth::after {{ background: linear-gradient(90deg, {C_EARTH}, #6B4423); }}
.kpi-icon {{ font-size: 24px; margin-bottom: 12px; display: block; }}
.kpi-label {{
    font-size: 10px; letter-spacing: 1.8px; text-transform: uppercase;
    color: {C_MUTED}; font-weight: 600; margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'Playfair Display', serif; font-size: 34px;
    font-weight: 900; color: {C_INK}; line-height: 1; margin-bottom: 6px;
}}
.kpi-note {{ font-size: 12px; color: {C_MUTED}; line-height: 1.4; }}
.kpi-badge {{
    display: inline-block; margin-top: 8px; padding: 3px 9px;
    border-radius: 999px; font-size: 11px; font-weight: 700;
}}
.kpi-badge.good {{ background: #E8F5E1; color: {C_OLIVE}; }}
.kpi-badge.warn {{ background: #FFF8E1; color: #8B6B00; }}
.kpi-badge.bad  {{ background: #FFE8EB; color: {C_RED}; }}

/* Section headers */
.sec-header {{ display: flex; align-items: baseline; gap: 12px; margin: 28px 0 4px; }}
.sec-title {{ font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 700; color: {C_INK}; }}
.sec-sub {{ font-size: 14px; color: {C_MUTED}; margin-bottom: 18px; font-weight: 300; }}

/* Callouts */
.callout {{
    border-radius: 14px; padding: 16px 20px; font-size: 14px;
    line-height: 1.65; margin: 16px 0;
    display: flex; gap: 12px; align-items: flex-start;
}}
.callout-icon {{ font-size: 18px; flex-shrink: 0; margin-top: 1px; }}
.callout.insight  {{ background: #F0F6EA; border: 1px solid rgba(74,103,65,0.2); color: #2A4025; }}
.callout.warning  {{ background: #FFF0F2; border: 1px solid rgba(200,16,46,0.2); color: #5A0B19; }}
.callout.gold-box {{ background: #FDF8EC; border: 1px solid rgba(201,162,39,0.3); color: #3C2E0A; }}

/* Story cards */
.story-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin: 18px 0; }}
.story-card {{
    background: white; border-radius: 18px; padding: 24px 22px;
    border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 2px 14px rgba(0,0,0,0.04);
}}
.story-card-icon {{
    width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; margin-bottom: 14px;
}}
.story-card-icon.red   {{ background: #FFF0F2; }}
.story-card-icon.olive {{ background: #EFF6E8; }}
.story-card-icon.gold  {{ background: #FDF8EC; }}
.story-card h3 {{
    font-family: 'Playfair Display', serif; font-size: 17px;
    font-weight: 700; color: {C_INK}; margin: 0 0 8px 0;
}}
.story-card p {{ color: {C_MUTED}; font-size: 13.5px; line-height: 1.65; margin: 0; }}

/* Formula hero */
.formula-hero {{
    background: {C_INK}; border-radius: 18px; padding: 28px 36px;
    text-align: center; position: relative; overflow: hidden; margin: 20px 0;
}}
.formula-hero::before {{
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 100% at 50% 50%, rgba(201,162,39,0.12) 0%, transparent 70%);
}}
.formula-label {{
    font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    color: rgba(255,255,255,0.4); margin-bottom: 10px;
}}
.formula-text {{
    font-family: 'DM Mono', monospace; font-size: 20px;
    color: {C_GOLD}; font-weight: 500; position: relative;
}}
.formula-text em {{ color: white; font-style: normal; }}

/* Map legend */
.map-legend {{ display: flex; gap: 18px; margin: 12px 0; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 7px; font-size: 12.5px; color: {C_MUTED}; font-weight: 500; }}
.legend-dot {{ width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }}

/* Economic impact cards */
.econ-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin: 18px 0; }}
.econ-card {{
    border-radius: 16px; padding: 20px; position: relative; overflow: hidden;
}}
.econ-card.over  {{ background: #FFF0F2; border: 1px solid rgba(200,16,46,0.18); }}
.econ-card.under {{ background: #EFF6E8; border: 1px solid rgba(74,103,65,0.18); }}
.econ-card.ok    {{ background: #FDF8EC; border: 1px solid rgba(201,162,39,0.25); }}
.econ-card-title {{ font-weight: 700; font-size: 14px; margin-bottom: 4px; }}
.econ-card-value {{ font-family: 'Playfair Display',serif; font-size: 26px; font-weight: 900; margin: 6px 0; }}
.econ-card-note  {{ font-size: 12px; line-height: 1.5; }}

/* Ornamental */
.ornamental-divider {{
    text-align: center; color: rgba(201,162,39,0.5);
    font-size: 18px; letter-spacing: 12px; margin: 24px 0;
}}

/* Metric comparison table */
.metric-table {{
    background: white; border-radius: 16px; overflow: hidden;
    border: 1px solid rgba(0,0,0,0.07); margin: 16px 0;
}}
.metric-row {{
    display: grid; grid-template-columns: 2fr 1fr 1fr 1fr;
    align-items: center; padding: 12px 20px;
    border-bottom: 1px solid rgba(0,0,0,0.05);
    font-size: 13px;
}}
.metric-row.header {{
    background: {C_INK}; color: rgba(255,255,255,0.6);
    font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600;
}}
.metric-row:last-child {{ border-bottom: none; }}

div[data-testid="stTabs"] button {{ font-family: 'DM Sans',sans-serif; font-weight: 600; font-size: 13.5px; }}
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}

/* Override streamlit selectbox / slider in dark filter bar */
.smart-filter-bar .stSelectbox div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(201,162,39,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fmt(x):
    return f"{x:,.0f}"

def fmt_k(x):
    return f"{x/1000:.1f}K"

def fig_base(fig, height=480, margin_t=80, margin_b=60, margin_l=60, margin_r=30):
    """Base layout with enough margin to prevent label clipping."""
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color=C_INK, size=12),
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant"
        ),
        title=dict(
            font=dict(family="Playfair Display, serif", size=15, color=C_INK),
            x=0.02, xanchor="left", pad=dict(b=12)
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False,
        tickfont=dict(size=11), title_font=dict(size=12)
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False,
        tickfont=dict(size=11), title_font=dict(size=12)
    )
    return fig

def badge(val, good_thresh=20, bad_thresh=50):
    if val < good_thresh:
        return f'<span class="kpi-badge good">✅ On Target</span>'
    elif val < bad_thresh:
        return f'<span class="kpi-badge warn">⚠️ Moderate</span>'
    else:
        return f'<span class="kpi-badge bad">🔴 High Error</span>'

def status_arrow(signed_err, ape):
    """Return emoji + arrow + text for error direction."""
    if ape < 10:
        return f"✅ Near-perfect ({ape:.1f}%)"
    elif signed_err > 0:
        return f"🔺 Over-forecast +{abs(signed_err):,.0f} t"
    else:
        return f"🔻 Under-forecast −{abs(signed_err):,.0f} t"


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = "final_master_dataset_fixed.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Check important columns early
    required_cols = ["governorate", "year", "production", "production_lag1"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in final_master_dataset_fixed.csv: {missing_cols}")

    # Clean governorate names
    df["governorate"] = (
        df["governorate"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # IMPORTANT FOR STREAMLIT CLOUD:
    # Force every non-categorical column to numeric because Cloud may read some columns as text.
    for col in df.columns:
        if col != "governorate":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "None": np.nan, "inf": np.nan, "-inf": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove infinity values created by ratios/growth features
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove impossible rows for training/evaluation
    df = df.dropna(subset=["year", "production", "production_lag1"]).copy()
    df["year"] = df["year"].astype(int)

    df = df.sort_values(["governorate", "year"]).reset_index(drop=True)
    return df

df = load_data()


# ─────────────────────────────────────────────
# METRICS — MAPE corrected (M = Mean)
# ─────────────────────────────────────────────
def mape(y_true, y_pred, epsilon=1.0):
    """Mean Absolute Percentage Error (MAPE)"""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def smape(y_true, y_pred, epsilon=1.0):
    """Symmetric Mean Absolute Percentage Error (sMAPE)"""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)) * 100

def evaluate(y_true, y_pred):
    return {
        "MAPE":  mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "MAE":   mean_absolute_error(y_true, y_pred),
        "RMSE":  np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":    r2_score(y_true, y_pred)
    }


# ─────────────────────────────────────────────
# FEATURES & MODEL
# ─────────────────────────────────────────────
feature_cols = [
    "year", "year_index", "governorate", "surface",
    "Temperature_C", "Precipitation_mm", "Soil_Moisture_Vol", "Precipitation_CHIRPS",
    "Winter", "Spring",
    "production_lag1", "production_lag2", "production_lag3",
    "prod_roll2_mean", "prod_roll3_mean", "prod_growth_lag1", "surface_change_ratio",
    "ndvi_peak", "ndvi_min", "ndvi_amplitude", "ndvi_winter_spring_diff", "ndvi_mean",
    "rain_temp_interaction", "chirps_temp_interaction", "moisture_temp_interaction",
    "ndvi_imputed_flag"
]
feature_cols = [c for c in feature_cols if c in df.columns]

@st.cache_resource
def load_saved_model():
    """
    Load the trained model artifact instead of training inside Streamlit.
    This makes deployment faster and prevents Streamlit Cloud startup crashes.
    """
    model_path = Path("olive_model.joblib")

    if not model_path.exists():
        st.error(
            "Missing model file: olive_model.joblib. "
            "Upload olive_model.joblib to the root of your GitHub repo, next to app.py."
        )
        st.stop()

    return joblib.load(model_path)

model = load_saved_model()


# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
valid_df = df[df["year"] >= 2021].copy()
X_valid = valid_df[feature_cols].copy()

# Keep governorate categorical and force all other columns numeric
for col in X_valid.columns:
    if col != "governorate":
        X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce")

X_valid = X_valid.replace([np.inf, -np.inf], np.nan)

baseline_pred = pd.to_numeric(X_valid["production_lag1"], errors="coerce").fillna(0).values

try:
    xgb_pred = np.maximum(np.expm1(model.predict(X_valid)), 0)
except Exception as e:
    st.error("Model prediction failed. Please make sure olive_model.joblib was trained using the same feature columns.")
    st.exception(e)
    st.stop()

final_pred = 0.98 * baseline_pred + 0.02 * xgb_pred

pred_df = valid_df[["year", "governorate", "production"]].copy()
pred_df["baseline_prediction"] = baseline_pred
pred_df["xgb_prediction"] = xgb_pred
pred_df["final_prediction"] = final_pred
pred_df["absolute_error"] = abs(pred_df["production"] - pred_df["final_prediction"])
pred_df["ape_percent"] = pred_df["absolute_error"] / np.maximum(abs(pred_df["production"]), 1) * 100
pred_df["signed_error"] = pred_df["final_prediction"] - pred_df["production"]
pred_df["direction"] = pred_df["signed_error"].apply(
    lambda x: "🔺 Over-forecast" if x > 0 else "🔻 Under-forecast"
)

metrics = evaluate(pred_df["production"], pred_df["final_prediction"])
baseline_metrics = evaluate(pred_df["production"], pred_df["baseline_prediction"])


# ─────────────────────────────────────────────
# NATIONAL AGGREGATION
# ─────────────────────────────────────────────
national_df = pred_df.groupby("year").agg(
    actual_national    = ("production", "sum"),
    baseline_national  = ("baseline_prediction", "sum"),
    predicted_national = ("final_prediction", "sum")
).reset_index()
national_df["national_ape"] = (
    abs(national_df["actual_national"] - national_df["predicted_national"])
    / national_df["actual_national"]
) * 100
national_mape = national_df["national_ape"].mean()


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
@st.cache_data
def get_feature_importance():
    try:
        # The saved pipeline may use either "pre" or "preprocessor" depending on the training notebook.
        pre_step_name = "preprocessor" if "preprocessor" in model.named_steps else "pre"
        preprocessor = model.named_steps[pre_step_name]
        xgb_model = model.named_steps["model"]

        names = preprocessor.get_feature_names_out()
        imp = xgb_model.feature_importances_

        return (
            pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(20)
        )
    except Exception:
        return pd.DataFrame()

feature_importance_df = get_feature_importance()


# ─────────────────────────────────────────────
# GEO COORDS
# ─────────────────────────────────────────────
gov_coords = {
    "ariana": (36.8665, 10.1647), "beja": (36.7256, 9.1817),
    "benarous": (36.7531, 10.2189), "bizerte": (37.2744, 9.8739),
    "gabes": (33.8815, 10.0982), "gafsa": (34.4311, 8.7757),
    "jendouba": (36.5011, 8.7802), "kairouan": (35.6781, 10.0963),
    "kasserine": (35.1676, 8.8365), "kef": (36.1822, 8.7148),
    "mahdia": (35.5047, 11.0622), "manouba": (36.8078, 10.1011),
    "medenine": (33.3549, 10.5055), "monastir": (35.7643, 10.8113),
    "nabeul": (36.4561, 10.7376), "sfax": (34.7398, 10.7600),
    "sidi bouzid": (35.0382, 9.4858), "siliana": (36.0849, 9.3708),
    "sousse": (35.8256, 10.6084), "tataouine": (32.9297, 10.4518),
    "zaghouan": (36.4029, 10.1429)
}
display_names = {"benarous": "Ben Arous", "sidi bouzid": "Sidi Bouzid"}
def nice_name(g):
    return display_names.get(g, g.title())


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-title">🫒 زيتون</div>
        <div class="sb-logo-arabic">Tunisia Olive Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-section-label">Hybrid Formula</div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 22px 6px;">
        <div class="sb-formula">
            <div class="sb-formula-main">ŷ = 0.98 × Baseline<br>+ 0.02 × XGBoost</div>
            <div class="sb-formula-sub">
                Baseline = production<sub>t−1</sub><br>
                XGBoost adds env. correction<br>
                Trained on log-transformed target
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:6px 22px 20px;">
        <div class="sb-section-label">Data Streams</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.5);line-height:2.1;">
            🌾&nbsp; GovAgri Records<br>
            🌡&nbsp; ERA5 Climate<br>
            🌧&nbsp; CHIRPS Rainfall<br>
            🛰&nbsp; NDVI Satellite<br>
            💧&nbsp; Soil Moisture
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 22px;border-top:1px solid rgba(255,255,255,0.07);padding-top:16px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section"><div class="sb-section-label">Live Performance</div></div>', unsafe_allow_html=True)

    gov_mape_val = metrics["MAPE"]
    nat_mape_val = national_mape
    mape_color   = C_OLIVE if gov_mape_val < 50 else C_RED
    nat_color    = C_OLIVE if nat_mape_val < 15 else C_RED

    st.markdown(f"""
    <div style="padding:0 22px 20px;display:flex;flex-direction:column;gap:10px;">
        <div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:12px 14px;">
            <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,0.35);margin-bottom:4px;">Gov. MAPE</div>
            <div style="font-family:'Playfair Display',serif;font-size:24px;font-weight:900;color:{mape_color};">{gov_mape_val:.1f}%</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:3px;">Mean Absolute Percentage Error</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:12px 14px;">
            <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,0.35);margin-bottom:4px;">National MAPE</div>
            <div style="font-family:'Playfair Display',serif;font-size:24px;font-weight:900;color:{nat_color};">{nat_mape_val:.1f}%</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:3px;">Aggregated forecast error</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:12px 14px;">
            <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,0.35);margin-bottom:4px;">R² Score</div>
            <div style="font-family:'Playfair Display',serif;font-size:24px;font-weight:900;color:{C_GOLD};">{metrics['R2']:.3f}</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:3px;">Variance explained</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-arabesque">🫒</div>
    <span class="hero-arabic">الجمهورية التونسية — République Tunisienne</span>
    <div class="hero-title">Tunisia <span>Olive</span> Production Intelligence</div>
    <div class="hero-sub">
        Regional decision-support platform combining satellite vegetation signals, ERA5 climate,
        CHIRPS rainfall, and soil moisture to forecast olive harvests across 21 Tunisian governorates.
    </div>
    <div class="hero-tags">
        <span class="hero-tag">🇹🇳 Tunisia-Focused</span>
        <span class="hero-tag">📍 21 Governorates</span>
        <span class="hero-tag">🛰 NDVI · ERA5 · CHIRPS</span>
        <span class="hero-tag">🤖 Hybrid ML</span>
        <span class="hero-tag">📅 Multi-Year Validation</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SMART FILTER BAR — dark, prominent, top of content
# ═══════════════════════════════════════════════
st.markdown(f"""
<div class="smart-filter-bar">
    <div class="filter-title">🎛️ &nbsp;Dashboard Controls — Filter your view</div>
</div>
""", unsafe_allow_html=True)

# Actual controls rendered via Streamlit (inside columns, visually inside the dark bar via CSS)
fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])

with fc1:
    st.markdown('<div class="filter-label" style="color:#7A8270;">📅 Validation Year</div>', unsafe_allow_html=True)
    year_options  = sorted(pred_df["year"].unique())
    selected_year = st.selectbox("year_sel", year_options, index=len(year_options)-1, label_visibility="collapsed")

with fc2:
    st.markdown('<div class="filter-label" style="color:#7A8270;">📍 Governorate Explorer</div>', unsafe_allow_html=True)
    selected_governorate = st.selectbox(
        "gov_sel", sorted(pred_df["governorate"].unique()),
        format_func=nice_name, label_visibility="collapsed"
    )

with fc3:
    st.markdown('<div class="filter-label" style="color:#7A8270;">⚠️ Show Govs with APE Above (%)</div>', unsafe_allow_html=True)
    error_filter = st.slider("err_filt", 0, 250, 0, step=5, label_visibility="collapsed")

with fc4:
    year_row = pred_df[pred_df["year"] == selected_year]
    n_gov    = len(year_row)
    n_bad    = len(year_row[year_row["ape_percent"] > 50])
    n_good   = len(year_row[year_row["ape_percent"] <= 25])
    st.markdown(f"""
    <div style="background:#FFF0F2;border-radius:12px;padding:10px 14px;text-align:center;border:1px solid rgba(200,16,46,0.15);margin-top:22px;">
        <div style="font-size:10px;color:{C_RED};letter-spacing:1.2px;text-transform:uppercase;font-weight:700;">
            🔴 High-Error
        </div>
        <div style="font-family:'Playfair Display',serif;font-size:26px;font-weight:900;color:{C_RED};line-height:1.1;">
            {n_bad}<span style="font-size:13px;color:{C_MUTED};">/{n_gov}</span>
        </div>
        <div style="font-size:10px;color:{C_OLIVE};font-weight:600;margin-top:3px;">✅ {n_good} on target</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════
mape_badge  = badge(metrics["MAPE"], 40, 80)
smape_badge = badge(metrics["SMAPE"], 40, 80)
r2_badge    = '<span class="kpi-badge good">✅ Strong Fit</span>' if metrics["R2"] > 0.7 else '<span class="kpi-badge warn">⚠️ Moderate</span>'
nat_badge   = '<span class="kpi-badge good">✅ Target Met</span>' if national_mape < 15 else '<span class="kpi-badge warn">⚠️ Above Target</span>'

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card red">
        <span class="kpi-icon">📍</span>
        <div class="kpi-label">Governorate MAPE</div>
        <div class="kpi-value">{metrics['MAPE']:.1f}%</div>
        <div class="kpi-note">Mean Absolute Percentage Error — regional primary metric</div>
        {mape_badge}
    </div>
    <div class="kpi-card olive">
        <span class="kpi-icon">⚖️</span>
        <div class="kpi-label">SMAPE</div>
        <div class="kpi-value">{metrics['SMAPE']:.1f}%</div>
        <div class="kpi-note">Symmetric MAPE — stable across high-low swings</div>
        {smape_badge}
    </div>
    <div class="kpi-card gold">
        <span class="kpi-icon">📈</span>
        <div class="kpi-label">R² Score</div>
        <div class="kpi-value">{metrics['R2']:.3f}</div>
        <div class="kpi-note">Explained production variance across governorates</div>
        {r2_badge}
    </div>
    <div class="kpi-card earth">
        <span class="kpi-icon">🇹🇳</span>
        <div class="kpi-label">National MAPE</div>
        <div class="kpi-value">{national_mape:.1f}%</div>
        <div class="kpi-note">Aggregated across all governorate forecasts</div>
        {nat_badge}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="callout insight">
    <span class="callout-icon">💡</span>
    <div><strong>Key insight:</strong> National MAPE is significantly lower than governorate-level MAPE —
    regional forecast errors partially cancel each other out through aggregation.
    This is expected and documented behavior for olive alternate-bearing panels.</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🇹🇳  Tunisia Context",
    "🗺️  Regional Map",
    "🫒  Governorate Explorer",
    "📈  National Forecast",
    "💹  Economic Impact",
    "📊  Model Evaluation",
    "🧩  Architecture"
])


# ══════════════════════════════════════════
# TAB 1 — CONTEXT
# ══════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="sec-header"><span class="sec-title">Why Olive Forecasting Matters in Tunisia</span></div>
    <div class="sec-sub">Tunisia is one of the world's top olive oil exporters. Reliable harvest forecasting is strategic infrastructure.</div>
    <div class="story-row">
        <div class="story-card">
            <div class="story-card-icon gold">🏛️</div>
            <h3>Strategic Sector</h3>
            <p>Olive oil accounts for a major share of Tunisia's agricultural exports. Accurate forecasting enables supply planning, pricing strategy, and export coordination months in advance.</p>
        </div>
        <div class="story-card">
            <div class="story-card-icon olive">📍</div>
            <h3>Regional Granularity</h3>
            <p>Each governorate has distinct soil, microclimate, and varietal mix. CRDA offices, cooperatives, and exporters need local-level intelligence — not just a national average.</p>
        </div>
        <div class="story-card">
            <div class="story-card-icon red">🌦️</div>
            <h3>Climate Vulnerability</h3>
            <p>Rainfall deficits, temperature stress, and soil moisture index feed the model months before harvest — enabling early-warning for at-risk governorates.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ornamental-divider">✦ · ✦ · ✦</div>', unsafe_allow_html=True)

    col_sun, col_bar = st.columns([1, 1])

    with col_sun:
        sources = pd.DataFrame({
            "Source":   ["Agricultural Records", "ERA5 Climate", "CHIRPS Rainfall", "NDVI Satellite", "Soil Moisture"],
            "Category": ["Production", "Climate", "Rainfall", "Remote Sensing", "Soil"],
        })
        fig_sun = px.sunburst(
            sources, path=["Category", "Source"], color="Category",
            color_discrete_map={
                "Production": C_RED, "Climate": C_OLIVE,
                "Rainfall": C_GOLD, "Remote Sensing": C_EARTH, "Soil": "#6B705C"
            },
            title="Five Data Streams Powering the Forecast"
        )
        fig_sun.update_traces(textfont_size=13, insidetextfont=dict(size=12))
        fig_base(fig_sun, 460, 60)
        st.plotly_chart(fig_sun, use_container_width=True)

    with col_bar:
        nat_hist = df.groupby("year")["production"].sum().reset_index()
        fig_hist_nat = go.Figure()
        fig_hist_nat.add_trace(go.Bar(
            x=nat_hist["year"], y=nat_hist["production"],
            marker=dict(
                color=nat_hist["production"],
                colorscale=[[0, "#C8E6C0"], [0.5, C_OLIVE], [1, C_DARKRED]],
                showscale=False
            ),
            text=[f"{v/1000:.0f}K t" for v in nat_hist["production"]],
            textposition="outside",
            textfont=dict(size=10, color=C_INK),
            cliponaxis=False
        ))
        fig_hist_nat.update_layout(
            title="Tunisia National Production — Historical (All Years)",
            uniformtext_minsize=8, uniformtext_mode="hide"
        )
        fig_base(fig_hist_nat, 460, 60, 80)
        fig_hist_nat.update_xaxes(tickmode="linear", dtick=2)
        fig_hist_nat.update_yaxes(range=[0, nat_hist["production"].max() * 1.18])
        st.plotly_chart(fig_hist_nat, use_container_width=True)

    st.markdown("""
    <div class="callout gold-box">
        <span class="callout-icon">🫒</span>
        <div><strong>Alternate-bearing effect:</strong> Olive trees naturally produce high yields one year and low yields the next.
        This biological cycle is the #1 source of forecast error at governorate level — and is visible clearly in the bars above.</div>
    </div>
    """, unsafe_allow_html=True)

    # Additional: Per-governorate production heatmap (year × governorate)
    st.markdown("""
    <div class="sec-header"><span class="sec-title" style="font-size:20px;">Production Heatmap — All Governorates × All Years</span></div>
    <div class="sec-sub">Alternate-bearing rhythm visible as alternating light/dark rows. Sfax dominates by volume.</div>
    """, unsafe_allow_html=True)

    hm_data = df.pivot_table(index="governorate", columns="year", values="production", aggfunc="sum")
    hm_data.index = [nice_name(g) for g in hm_data.index]
    fig_hm = go.Figure(go.Heatmap(
        z=hm_data.values,
        x=[str(c) for c in hm_data.columns],
        y=hm_data.index.tolist(),
        colorscale=[[0, "#EFF6E8"], [0.4, C_GOLD], [0.7, C_EARTH], [1, C_DARKRED]],
        hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Production: %{z:,.0f} t<extra></extra>",
        text=[[f"{v/1000:.0f}K" if not np.isnan(v) else "" for v in row] for row in hm_data.values],
        texttemplate="%{text}",
        textfont=dict(size=9, color=C_INK),
        colorbar=dict(title="Tonnes", thickness=14, len=0.8)
    ))
    fig_hm.update_layout(title="Production (tonnes) — Governorate × Year Matrix")
    fig_base(fig_hm, 560, 60, 80, 120)
    fig_hm.update_xaxes(tickangle=0, tickfont=dict(size=11))
    fig_hm.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════
# TAB 2 — REGIONAL MAP
# ══════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div class="sec-header"><span class="sec-title">Regional Forecast Map — {selected_year}</span></div>
    <div class="sec-sub">Circle size reflects production volume · Color reflects forecast accuracy (APE)</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="map-legend">
        <div class="legend-item"><div class="legend-dot" style="background:#2E8B57"></div> APE &lt; 20% — ✅ Accurate</div>
        <div class="legend-item"><div class="legend-dot" style="background:#E8A020"></div> APE 20–50% — ⚠️ Moderate</div>
        <div class="legend-item"><div class="legend-dot" style="background:#C8102E"></div> APE &gt; 50% — 🔴 Difficult</div>
    </div>
    """, unsafe_allow_html=True)

    year_map_df = pred_df[pred_df["year"] == selected_year].copy()
    year_map_df["lat"] = year_map_df["governorate"].map(lambda x: gov_coords.get(x, (None, None))[0])
    year_map_df["lon"] = year_map_df["governorate"].map(lambda x: gov_coords.get(x, (None, None))[1])
    year_map_df = year_map_df.dropna(subset=["lat", "lon"])

    m = folium.Map(location=[34.8, 9.6], zoom_start=6, tiles=None)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB", name="CartoDB Light", control=False
    ).add_to(m)

    for _, row in year_map_df.iterrows():
        ape   = row["ape_percent"]
        color = "#2E8B57" if ape < 20 else ("#E8A020" if ape < 50 else "#C8102E")
        icon  = "✅" if ape < 20 else ("⚠️" if ape < 50 else "🔴")
        direction = "🔺 Over" if row["signed_error"] > 0 else "🔻 Under"
        err_txt   = status_arrow(row["signed_error"], ape)

        popup_html = f"""
        <div style="font-family:'DM Sans',sans-serif;padding:12px;min-width:220px;border-top:3px solid {color}">
            <div style="font-size:15px;font-weight:700;color:#0F1A0D;margin-bottom:8px;">
                {icon} {nice_name(row['governorate'])}
            </div>
            <div style="color:#555;font-size:12.5px;line-height:2.1;">
                <b>Year:</b> {int(row['year'])}<br>
                <b>Actual:</b> {row['production']:,.0f} t<br>
                <b>Predicted:</b> {row['final_prediction']:,.0f} t<br>
                <b>Gap:</b> <span style="color:{color};font-weight:700">{err_txt}</span><br>
                <b>APE:</b> <span style="color:{color};font-weight:700">{ape:.1f}%</span>
            </div>
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7 + min(row["production"] / 50000, 12),
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{nice_name(row['governorate'])}  |  APE: {ape:.1f}%  {icon}  {direction}",
            color=color, fill=True, fill_color=color, fill_opacity=0.72, weight=1.5
        ).add_to(m)

    st_folium(m, width=None, height=560)

    # Companion bar chart below map
    year_map_sorted = year_map_df.sort_values("ape_percent", ascending=False)
    year_map_sorted["gov_name"] = year_map_sorted["governorate"].map(nice_name)
    bar_colors_map = [
        C_RED if v > 50 else (C_GOLD if v > 20 else C_OLIVE)
        for v in year_map_sorted["ape_percent"]
    ]
    icon_labels = [
        f"🔴 {v:.0f}%" if v > 50 else (f"⚠️ {v:.0f}%" if v > 20 else f"✅ {v:.0f}%")
        for v in year_map_sorted["ape_percent"]
    ]
    fig_map_bar = go.Figure(go.Bar(
        x=year_map_sorted["gov_name"],
        y=year_map_sorted["ape_percent"],
        marker_color=bar_colors_map,
        text=icon_labels,
        textposition="outside",
        textfont=dict(size=10, color=C_INK),
        cliponaxis=False
    ))
    fig_map_bar.add_hline(y=50, line_color=C_RED, line_dash="dash", line_width=1.5,
                          annotation_text="🔴 High-error threshold (50%)",
                          annotation_font=dict(size=11, color=C_RED),
                          annotation_position="top right")
    fig_map_bar.add_hline(y=20, line_color=C_OLIVE, line_dash="dot", line_width=1.5,
                          annotation_text="✅ Good threshold (20%)",
                          annotation_font=dict(size=11, color=C_OLIVE),
                          annotation_position="top left")
    fig_map_bar.update_layout(
        title=f"APE by Governorate — {selected_year} (sorted worst → best)",
        yaxis_title="APE (%)"
    )
    fig_base(fig_map_bar, 380, 60, 80)
    fig_map_bar.update_xaxes(tickangle=-40, tickfont=dict(size=10))
    fig_map_bar.update_yaxes(range=[0, year_map_sorted["ape_percent"].max() * 1.2])
    st.plotly_chart(fig_map_bar, use_container_width=True)

    st.markdown("""
    <div class="callout gold-box">
        <span class="callout-icon">🗺️</span>
        <div>Click any circle for detailed statistics. Use the Year filter above to compare different
        harvest seasons. Red clusters correspond to alternate-bearing crash years or localised climate shocks.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 3 — GOVERNORATE EXPLORER
# ══════════════════════════════════════════
with tab3:
    gov_display_name = nice_name(selected_governorate)
    st.markdown(f"""
    <div class="sec-header"><span class="sec-title">{gov_display_name} — Forecast Detail</span></div>
    <div class="sec-sub">Year {selected_year} · Hybrid model breakdown · Full historical trend</div>
    """, unsafe_allow_html=True)

    one = pred_df[
        (pred_df["year"] == selected_year) &
        (pred_df["governorate"] == selected_governorate)
    ]

    if not one.empty:
        actual   = float(one["production"].iloc[0])
        baseline = float(one["baseline_prediction"].iloc[0])
        final    = float(one["final_prediction"].iloc[0])
        ape      = float(one["ape_percent"].iloc[0])
        signed   = float(one["signed_error"].iloc[0])
        direction_txt = f"🔺 Over-forecast by {abs(signed):,.0f} t" if signed > 0 else f"🔻 Under-forecast by {abs(signed):,.0f} t"
        dir_color     = C_RED if signed > 0 else C_OLIVE
        ape_class     = "red" if ape > 50 else ("gold" if ape > 25 else "olive")
        ape_icon      = "🔴" if ape > 50 else ("⚠️" if ape > 25 else "✅")

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card olive">
                <span class="kpi-icon">✅</span>
                <div class="kpi-label">Actual Production</div>
                <div class="kpi-value">{fmt(actual)}</div>
                <div class="kpi-note">Observed tonnes · {selected_year}</div>
            </div>
            <div class="kpi-card gold">
                <span class="kpi-icon">🔮</span>
                <div class="kpi-label">Final Prediction</div>
                <div class="kpi-value">{fmt(final)}</div>
                <div class="kpi-note">Hybrid forecast (0.98 / 0.02 blend)</div>
                <span style="font-size:12px;font-weight:700;color:{dir_color};">{direction_txt}</span>
            </div>
            <div class="kpi-card earth">
                <span class="kpi-icon">📌</span>
                <div class="kpi-label">Baseline Only</div>
                <div class="kpi-value">{fmt(baseline)}</div>
                <div class="kpi-note">Lag-1 persistence (previous year)</div>
            </div>
            <div class="kpi-card {ape_class}">
                <span class="kpi-icon">{ape_icon}</span>
                <div class="kpi-label">APE This Case</div>
                <div class="kpi-value">{ape:.1f}%</div>
                <div class="kpi-note">{'High-variance year — alternate-bearing or climate shock' if ape > 50 else 'Within acceptable forecast range'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_bar_g, col_needle = st.columns([3, 2])

        with col_bar_g:
            # Side-by-side actual vs predicted vs baseline — clean, no overlap
            categories = ["✅ Actual", "🔮 Predicted", "📌 Baseline"]
            values     = [actual, final, baseline]
            bar_colors_g = [C_OLIVE, C_RED if ape > 30 else C_GOLD, C_EARTH]

            fig_gov_bar = go.Figure()
            for cat, val, col in zip(categories, values, bar_colors_g):
                fig_gov_bar.add_trace(go.Bar(
                    name=cat, x=[cat], y=[val],
                    marker_color=col,
                    text=[f"<b>{val:,.0f} t</b>"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=13, color="white", family="DM Sans"),
                    width=0.5
                ))

            # Arrow annotation above bars showing gap
            arrow_color  = C_RED if signed > 0 else C_OLIVE
            arrow_symbol = "🔺" if signed > 0 else "🔻"
            gap_label    = f"{arrow_symbol} {abs(signed):,.0f} t gap · APE = {ape:.1f}%"

            fig_gov_bar.add_annotation(
                x=0.5, xref="paper",
                y=max(actual, final, baseline) * 1.13,
                yref="y",
                text=gap_label,
                showarrow=False,
                font=dict(size=13, color=arrow_color, family="DM Sans"),
                bgcolor="rgba(255,255,255,0.9)",
                borderpad=7, bordercolor=arrow_color, borderwidth=2
            )
            # Draw a bracket/line between actual and predicted
            fig_gov_bar.add_shape(
                type="line",
                x0=0, x1=1, xref="paper",
                y0=actual, y1=actual, yref="y",
                line=dict(color=C_OLIVE, width=1.5, dash="dot")
            )

            fig_gov_bar.update_layout(
                title=f"{gov_display_name} — Actual vs Predicted vs Baseline ({selected_year})",
                showlegend=True, bargap=0.35,
                legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)")
            )
            fig_base(fig_gov_bar, 420, 80, 80)
            fig_gov_bar.update_yaxes(range=[0, max(actual, final, baseline) * 1.25])
            st.plotly_chart(fig_gov_bar, use_container_width=True)

        with col_needle:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=ape,
                number={"suffix": "%", "font": {"family": "Playfair Display", "size": 38, "color": C_INK}},
                title={"text": f"APE — {gov_display_name}", "font": {"size": 12, "family": "DM Sans", "color": C_MUTED}},
                delta={"reference": 30, "suffix": "%", "increasing": {"color": C_RED}, "decreasing": {"color": C_OLIVE}},
                gauge={
                    "axis": {"range": [0, 150], "ticksuffix": "%", "tickfont": {"size": 10}},
                    "bar": {"color": C_RED if ape > 50 else (C_GOLD if ape > 25 else C_OLIVE), "thickness": 0.22},
                    "steps": [
                        {"range": [0, 25],   "color": "#E8F5E1"},
                        {"range": [25, 50],  "color": "#FFF8E1"},
                        {"range": [50, 150], "color": "#FFE8EB"}
                    ],
                    "threshold": {"line": {"color": C_INK, "width": 2}, "thickness": 0.75, "value": 50}
                }
            ))
            fig_g.update_layout(
                height=260, paper_bgcolor="white",
                font=dict(family="DM Sans", color=C_INK),
                margin=dict(t=40, b=10, l=20, r=20)
            )
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown(f"""
            <div style="background:white;border-radius:14px;border:1px solid rgba(0,0,0,0.07);padding:16px 18px;">
                <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:{C_MUTED};margin-bottom:10px;font-weight:600;">Error Summary</div>
                <div style="display:flex;flex-direction:column;gap:8px;font-size:13px;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:{C_MUTED}">Absolute Error</span>
                        <span style="font-weight:700;color:{C_INK}">{abs(signed):,.0f} t</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:{C_MUTED}">Direction</span>
                        <span style="font-weight:700;color:{dir_color}">{'🔺 Over' if signed>0 else '🔻 Under'}-forecast</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:{C_MUTED}">MAPE</span>
                        <span style="font-weight:700;color:{C_RED if ape>50 else C_OLIVE}">{ape:.1f}%</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:{C_MUTED}">Signal</span>
                        <span style="font-weight:700;">{'🔴 High error' if ape>50 else ('⚠️ Moderate' if ape>25 else '✅ Good')}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Full historical trend
    gov_history = df[df["governorate"] == selected_governorate].copy()
    gov_pred    = pred_df[pred_df["governorate"] == selected_governorate].copy()

    fig_trend = go.Figure()

    if not gov_pred.empty:
        fig_trend.add_vrect(
            x0=gov_pred["year"].min() - 0.4,
            x1=gov_pred["year"].max() + 0.4,
            fillcolor="rgba(200,16,46,0.04)",
            layer="below", line_width=0,
            annotation_text="📅 Validation period",
            annotation_position="top left",
            annotation_font=dict(size=11, color=C_RED)
        )

    fig_trend.add_trace(go.Scatter(
        x=gov_history["year"], y=gov_history["production"],
        mode="lines+markers", name="✅ Historical Actual",
        line=dict(color=C_OLIVE, width=3),
        marker=dict(size=8, color=C_OLIVE, symbol="circle", line=dict(color="white", width=2))
    ))
    fig_trend.add_trace(go.Scatter(
        x=gov_pred["year"], y=gov_pred["final_prediction"],
        mode="lines+markers", name="🔮 Model Prediction",
        line=dict(color=C_RED, width=3, dash="dash"),
        marker=dict(size=10, color=C_RED, symbol="diamond", line=dict(color="white", width=2))
    ))
    fig_trend.add_trace(go.Scatter(
        x=gov_pred["year"], y=gov_pred["baseline_prediction"],
        mode="lines", name="📌 Baseline (Lag-1)",
        line=dict(color=C_GOLD, width=2, dash="dot"), opacity=0.75
    ))

    if not one.empty:
        fig_trend.add_annotation(
            x=selected_year, y=actual,
            text=f"◀ {selected_year}<br>Actual: {fmt(actual)}<br>APE: {ape:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor=C_GOLD, arrowsize=1.4,
            ax=55, ay=-55,
            font=dict(size=10, color=C_INK, family="DM Sans"),
            bgcolor="rgba(255,255,255,0.92)", borderpad=6,
            bordercolor=C_GOLD, borderwidth=1.5
        )

    fig_trend.update_layout(
        title=f"{gov_display_name} — Full Production History & Validation Forecasts",
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        yaxis_title="Production (tonnes)"
    )
    fig_base(fig_trend, 480, 80, 80)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Filtered table
    st.markdown(f"""
    <div class="sec-title" style="font-size:18px;margin-top:20px;">
        All Governorates — {selected_year}
        <span style="font-size:13px;color:{C_MUTED};font-family:'DM Sans';font-weight:400;margin-left:10px;">
            showing MAPE ≥ {error_filter}%
        </span>
    </div>
    """, unsafe_allow_html=True)

    year_table = pred_df[
        (pred_df["year"] == selected_year) &
        (pred_df["ape_percent"] >= error_filter)
    ].sort_values("ape_percent", ascending=False).copy()
    year_table["governorate"] = year_table["governorate"].map(nice_name)
    year_table["status"]      = year_table["ape_percent"].apply(
        lambda v: "🔴 High" if v > 50 else ("⚠️ Moderate" if v > 25 else "✅ Good")
    )
    year_table["direction"]   = year_table["signed_error"].apply(
        lambda x: "🔺 Over-forecast" if x > 0 else "🔻 Under-forecast"
    )
    display_cols = {
        "governorate": "Governorate", "production": "Actual (t)",
        "final_prediction": "Predicted (t)", "ape_percent": "APE (%)",
        "absolute_error": "Abs Error (t)", "direction": "Direction", "status": "Status"
    }
    year_table = year_table.rename(columns=display_cols)
    st.dataframe(year_table[list(display_cols.values())].round(1), use_container_width=True)


# ══════════════════════════════════════════
# TAB 4 — NATIONAL FORECAST
# ══════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="sec-header"><span class="sec-title">National Production Forecast</span></div>
    <div class="sec-sub">Sum of all governorate predictions vs. observed national totals — aggregation smooths regional errors.</div>
    """, unsafe_allow_html=True)

    col_chart, col_right = st.columns([3, 2])

    with col_chart:
        fig_nat = go.Figure()
        fig_nat.add_trace(go.Scatter(
            x=national_df["year"], y=national_df["actual_national"],
            mode="lines+markers", name="✅ Actual",
            line=dict(color=C_OLIVE, width=4),
            marker=dict(size=12, color=C_OLIVE, symbol="circle", line=dict(color="white", width=2)),
            fill="tozeroy", fillcolor="rgba(74,103,65,0.07)"
        ))
        fig_nat.add_trace(go.Scatter(
            x=national_df["year"], y=national_df["predicted_national"],
            mode="lines+markers", name="🔮 Predicted",
            line=dict(color=C_RED, width=4, dash="dash"),
            marker=dict(size=12, color=C_RED, symbol="diamond", line=dict(color="white", width=2))
        ))
        fig_nat.add_trace(go.Scatter(
            x=national_df["year"], y=national_df["baseline_national"],
            mode="lines", name="📌 Baseline",
            line=dict(color=C_GOLD, width=2, dash="dot"), opacity=0.7
        ))

        # Per-year MAPE labels — placed clearly above the max line
        for _, row in national_df.iterrows():
            icon      = "✅" if row["national_ape"] < 15 else "⚠️"
            label_col = C_OLIVE if row["national_ape"] < 15 else C_RED
            fig_nat.add_annotation(
                x=row["year"],
                y=max(row["actual_national"], row["predicted_national"]),
                text=f"{icon} {row['national_ape']:.1f}%",
                showarrow=True, arrowhead=2, arrowsize=0.8,
                arrowcolor=label_col, ax=0, ay=-32,
                font=dict(size=10, color=label_col, family="DM Sans"),
                bgcolor="rgba(255,255,255,0.88)", borderpad=4,
                bordercolor=label_col, borderwidth=1
            )
        fig_nat.update_layout(
            title="National Production: Actual vs Predicted — per-year MAPE annotated",
            legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
            yaxis_title="Production (tonnes)"
        )
        fig_base(fig_nat, 480, 90, 90)
        st.plotly_chart(fig_nat, use_container_width=True)

        # Waterfall chart — year-on-year error evolution
        st.markdown("""
        <div class="sec-sub" style="margin-top:12px;">Signed error waterfall — did we over or under-forecast each year?</div>
        """, unsafe_allow_html=True)

        national_df["signed_nat_err"] = national_df["predicted_national"] - national_df["actual_national"]
        wf_colors = [C_RED if v > 0 else C_OLIVE for v in national_df["signed_nat_err"]]
        wf_icons  = ["🔺" if v > 0 else "🔻" for v in national_df["signed_nat_err"]]
        wf_labels = [f"{ic} {v/1000:+.1f}K t" for ic, v in zip(wf_icons, national_df["signed_nat_err"])]

        fig_wf = go.Figure(go.Bar(
            x=[str(y) for y in national_df["year"]],
            y=national_df["signed_nat_err"],
            marker_color=wf_colors,
            text=wf_labels,
            textposition="outside",
            textfont=dict(size=11, color=C_INK),
            cliponaxis=False
        ))
        fig_wf.add_hline(y=0, line_color=C_INK, line_width=2)
        fig_wf.add_annotation(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text="🔺 Red = Over-forecast &nbsp;&nbsp; 🔻 Green = Under-forecast",
            showarrow=False, font=dict(size=11, color=C_MUTED, family="DM Sans"),
            bgcolor="rgba(255,255,255,0.85)", borderpad=6
        )
        fig_wf.update_layout(
            title="National Signed Error by Year (tonnes) — Over vs Under",
            yaxis_title="Signed Error (tonnes)"
        )
        fig_base(fig_wf, 340, 60, 80)
        max_abs = national_df["signed_nat_err"].abs().max()
        fig_wf.update_yaxes(range=[-max_abs * 1.4, max_abs * 1.4])
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_right:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=national_mape,
            number={"suffix": "%", "font": {"family": "Playfair Display", "size": 42}},
            title={"text": "National MAPE", "font": {"size": 13, "family": "DM Sans", "color": C_MUTED}},
            delta={"reference": 15, "suffix": "%", "increasing": {"color": C_RED}, "decreasing": {"color": C_OLIVE}},
            gauge={
                "axis": {"range": [0, 30], "ticksuffix": "%", "tickfont": {"size": 10}},
                "bar": {"color": C_OLIVE if national_mape <= 15 else C_RED, "thickness": 0.22},
                "steps": [
                    {"range": [0, 10],  "color": "#E8F5E1"},
                    {"range": [10, 20], "color": "#FFF8E1"},
                    {"range": [20, 30], "color": "#FFE8EB"}
                ],
                "threshold": {"line": {"color": C_GOLD, "width": 3}, "thickness": 0.8, "value": 15}
            }
        ))
        fig_gauge.update_layout(
            height=300, paper_bgcolor="white",
            font=dict(family="DM Sans", color=C_INK),
            margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="background:white;border-radius:14px;border:1px solid rgba(0,0,0,0.07);overflow:hidden;">
            <div style="padding:12px 18px;background:{C_INK};display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,0.5);font-weight:600;">
                <span>Year</span><span>Actual</span><span>Predicted</span><span>MAPE</span>
            </div>
        """, unsafe_allow_html=True)

        for _, row in national_df.iterrows():
            acc   = "✅" if row["national_ape"] < 15 else ("⚠️" if row["national_ape"] < 30 else "🔴")
            col   = C_OLIVE if row["national_ape"] < 15 else C_RED
            st.markdown(
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;padding:10px 18px;'
                f'border-bottom:1px solid rgba(0,0,0,0.05);font-size:12.5px;align-items:center;">'
                f'<span style="font-weight:700;">{int(row["year"])}</span>'
                f'<span>{row["actual_national"]/1000:.0f}K</span>'
                f'<span>{row["predicted_national"]/1000:.0f}K</span>'
                f'<span style="color:{col};font-weight:700;">{acc} {row["national_ape"]:.1f}%</span></div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="callout insight">
        <span class="callout-icon">📈</span>
        <div><strong>Aggregation benefit:</strong> National MAPE is substantially lower than governorate MAPE.
        Regional over-forecasts and under-forecasts partially offset each other, demonstrating the statistical
        benefit of running a portfolio of regional forecasts rather than a single national model.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 5 — ECONOMIC IMPACT
# ══════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="sec-header"><span class="sec-title">Economic Impact Analysis</span></div>
    <div class="sec-sub">What does a forecast error mean in practice? Translating tonnage gaps into economic context.</div>
    """, unsafe_allow_html=True)

    OLIVE_OIL_YIELD  = 0.18
    PRICE_PER_TON_OO = 5500

    pred_df["oil_actual"]    = pred_df["production"] * OLIVE_OIL_YIELD
    pred_df["oil_predicted"] = pred_df["final_prediction"] * OLIVE_OIL_YIELD
    pred_df["oil_gap"]       = pred_df["oil_predicted"] - pred_df["oil_actual"]
    pred_df["value_gap_usd"] = pred_df["oil_gap"] * PRICE_PER_TON_OO

    total_over_usd  = pred_df[pred_df["value_gap_usd"] > 0]["value_gap_usd"].sum()
    total_under_usd = pred_df[pred_df["value_gap_usd"] < 0]["value_gap_usd"].sum()
    total_abs_usd   = pred_df["value_gap_usd"].abs().sum()
    n_over          = (pred_df["value_gap_usd"] > 0).sum()
    n_under         = (pred_df["value_gap_usd"] < 0).sum()

    st.markdown(f"""
    <div style="background:{C_PARCHMENT};border-radius:16px;padding:14px 22px;margin-bottom:18px;font-size:13px;color:{C_MUTED};">
        💡 <strong>Assumptions:</strong> 18% oil extraction yield · Export price ~$5,500 USD/ton olive oil · For directional illustration only
    </div>
    <div class="econ-grid">
        <div class="econ-card over">
            <div class="econ-card-title" style="color:{C_RED};">🔺 Over-forecast Cases</div>
            <div class="econ-card-value" style="color:{C_RED};">+${total_over_usd/1e6:.1f}M</div>
            <div class="econ-card-note" style="color:{C_MUTED};">
                {n_over} governorate-years where we predicted <em>more</em> than actual.<br>
                Risk: over-procurement, excess logistics capacity booked.
            </div>
        </div>
        <div class="econ-card under">
            <div class="econ-card-title" style="color:{C_OLIVE};">🔻 Under-forecast Cases</div>
            <div class="econ-card-value" style="color:{C_OLIVE};">${total_under_usd/1e6:.1f}M</div>
            <div class="econ-card-note" style="color:{C_MUTED};">
                {n_under} governorate-years where we predicted <em>less</em> than actual.<br>
                Risk: missed export opportunities, storage shortfalls.
            </div>
        </div>
        <div class="econ-card ok">
            <div class="econ-card-title" style="color:{C_GOLD};">⚡ Total Absolute Gap</div>
            <div class="econ-card-value" style="color:{C_GOLD};">${total_abs_usd/1e6:.1f}M</div>
            <div class="econ-card-note" style="color:{C_MUTED};">
                Combined directional exposure across all governorate-years in the validation set.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Actual vs Predicted side-by-side economic bar chart ──
    st.markdown(f"""
    <div class="sec-header"><span class="sec-title" style="font-size:20px;">Actual vs Predicted — Economic Side-by-Side</span></div>
    <div class="sec-sub">Top governorates by volume for {selected_year} · bars are scale-comparable · gap = economic exposure</div>
    """, unsafe_allow_html=True)

    year_econ = pred_df[pred_df["year"] == selected_year].copy()
    year_econ = year_econ.sort_values("production", ascending=False).head(12)
    year_econ["gov_name"] = year_econ["governorate"].map(nice_name)

    fig_side = go.Figure()
    fig_side.add_trace(go.Bar(
        name="✅ Actual Production",
        x=year_econ["gov_name"],
        y=year_econ["production"],
        marker_color=C_OLIVE, opacity=0.92,
        text=[f"<b>{v/1000:.1f}K</b>" for v in year_econ["production"]],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=10, color="white")
    ))
    fig_side.add_trace(go.Bar(
        name="🔮 Predicted Production",
        x=year_econ["gov_name"],
        y=year_econ["final_prediction"],
        marker_color=C_RED, opacity=0.75,
        text=[f"<b>{v/1000:.1f}K</b>" for v in year_econ["final_prediction"]],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=10, color="white")
    ))

    # Annotate worst-gap govs
    for _, r in year_econ.iterrows():
        gap     = r["final_prediction"] - r["production"]
        ape_v   = r["ape_percent"]
        if ape_v > 30:
            icon = "🔺" if gap > 0 else "🔻"
            fig_side.add_annotation(
                x=r["gov_name"],
                y=max(r["production"], r["final_prediction"]) * 1.04,
                text=f"{icon} {ape_v:.0f}% MAPE",
                showarrow=False,
                font=dict(size=9, color=C_RED if gap > 0 else C_OLIVE, family="DM Sans"),
                bgcolor="rgba(255,255,255,0.85)", borderpad=3,
                bordercolor=C_RED if gap > 0 else C_OLIVE, borderwidth=1
            )

    fig_side.update_layout(
        barmode="group",
        title=f"✅ Actual vs 🔮 Predicted — Top Governorates by Volume ({selected_year})",
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        bargap=0.22, bargroupgap=0.05,
        yaxis_title="Production (tonnes)"
    )
    fig_base(fig_side, 460, 90, 90)
    fig_side.update_xaxes(tickangle=-30, tickfont=dict(size=10))
    fig_side.update_yaxes(range=[0, max(year_econ["production"].max(), year_econ["final_prediction"].max()) * 1.28])
    st.plotly_chart(fig_side, use_container_width=True)

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        # Signed error lollipop
        year_err = pred_df[pred_df["year"] == selected_year].copy().sort_values("signed_error")
        year_err["gov_name"] = year_err["governorate"].map(nice_name)

        fig_lollipop = go.Figure()
        for _, r in year_err.iterrows():
            lc = C_RED if r["signed_error"] > 0 else C_OLIVE
            fig_lollipop.add_shape(
                type="line",
                x0=0, x1=r["signed_error"],
                y0=r["gov_name"], y1=r["gov_name"],
                line=dict(color=lc, width=2)
            )
        fig_lollipop.add_trace(go.Scatter(
            x=year_err["signed_error"],
            y=year_err["gov_name"],
            mode="markers",
            text=[f"{v:+,.0f} t" for v in year_err["signed_error"]],
            textposition="middle right",
            textfont=dict(size=9, color=C_INK),
            marker=dict(
                size=10,
                color=[C_RED if v > 0 else C_OLIVE for v in year_err["signed_error"]],
                line=dict(color="white", width=1.5)
            ),
            showlegend=False
        ))
        fig_lollipop.add_vline(x=0, line_color=C_INK, line_width=1.5)
        fig_lollipop.add_annotation(
            x=year_err["signed_error"].max() * 0.55, y=0.97, xref="x", yref="paper",
            text="🔺 Over-forecast", showarrow=False,
            font=dict(size=11, color=C_RED, family="DM Sans"),
            bgcolor="rgba(255,240,242,0.85)", borderpad=4
        )
        fig_lollipop.add_annotation(
            x=year_err["signed_error"].min() * 0.55, y=0.97, xref="x", yref="paper",
            text="🔻 Under-forecast", showarrow=False,
            font=dict(size=11, color=C_OLIVE, family="DM Sans"),
            bgcolor="rgba(240,246,234,0.85)", borderpad=4
        )
        fig_lollipop.update_layout(
            title=f"Forecast Direction by Governorate ({selected_year})",
            xaxis_title="Signed Error (tonnes)",
        )
        fig_base(fig_lollipop, 460, 60, 80, 110)
        fig_lollipop.update_xaxes(tickformat=",")
        st.plotly_chart(fig_lollipop, use_container_width=True)

    with col_e2:
        # USD value gap bar
        year_econ2 = pred_df[pred_df["year"] == selected_year].copy()
        year_econ2["gov_name"] = year_econ2["governorate"].map(nice_name)
        year_econ2 = year_econ2.sort_values("value_gap_usd")
        bar_colors_e = [C_RED if v > 0 else C_OLIVE for v in year_econ2["value_gap_usd"]]
        icons_e = ["🔺" if v > 0 else "🔻" for v in year_econ2["value_gap_usd"]]
        labels_e = [f"{ic} ${abs(v)/1e6:.2f}M" for ic, v in zip(icons_e, year_econ2["value_gap_usd"])]

        fig_val = go.Figure(go.Bar(
            x=year_econ2["gov_name"],
            y=year_econ2["value_gap_usd"] / 1e6,
            marker_color=bar_colors_e, opacity=0.87,
            text=labels_e,
            textposition="outside",
            textfont=dict(size=9, color=C_INK),
            cliponaxis=False
        ))
        fig_val.add_hline(y=0, line_color=C_INK, line_width=1.5)
        fig_val.update_layout(
            title=f"Estimated Economic Gap by Governorate ({selected_year}) · USD Millions",
            yaxis_title="USD Millions (gap)",
        )
        fig_base(fig_val, 460, 60, 90)
        max_v = year_econ2["value_gap_usd"].abs().max() / 1e6
        fig_val.update_xaxes(tickangle=-35, tickfont=dict(size=10))
        fig_val.update_yaxes(range=[-max_v * 1.4, max_v * 1.4])
        st.plotly_chart(fig_val, use_container_width=True)

    # Economic summary across all years
    st.markdown(f"""
    <div class="sec-header"><span class="sec-title" style="font-size:20px;">Multi-Year Economic Exposure</span></div>
    <div class="sec-sub">Total USD gap per validation year — direction matters more than magnitude</div>
    """, unsafe_allow_html=True)

    yr_econ_agg = pred_df.groupby("year").agg(
        total_gap_usd=("value_gap_usd", "sum"),
        abs_gap_usd=("value_gap_usd", lambda x: x.abs().sum()),
        n_over=("value_gap_usd", lambda x: (x > 0).sum()),
        n_under=("value_gap_usd", lambda x: (x < 0).sum())
    ).reset_index()

    fig_yr_econ = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total Signed Gap by Year (USD M)", "Over vs Under Count by Year"),
        horizontal_spacing=0.12
    )
    yr_colors = [C_RED if v > 0 else C_OLIVE for v in yr_econ_agg["total_gap_usd"]]
    fig_yr_econ.add_trace(go.Bar(
        x=[str(y) for y in yr_econ_agg["year"]],
        y=yr_econ_agg["total_gap_usd"] / 1e6,
        marker_color=yr_colors,
        text=[f"${v/1e6:+.1f}M" for v in yr_econ_agg["total_gap_usd"]],
        textposition="outside", textfont=dict(size=11, color=C_INK),
        cliponaxis=False, name="Signed Gap"
    ), row=1, col=1)
    fig_yr_econ.add_trace(go.Bar(
        x=[str(y) for y in yr_econ_agg["year"]],
        y=yr_econ_agg["n_over"],
        name="🔺 Over-forecast govs",
        marker_color=C_RED, opacity=0.85,
        text=yr_econ_agg["n_over"], textposition="inside",
        textfont=dict(size=11, color="white")
    ), row=1, col=2)
    fig_yr_econ.add_trace(go.Bar(
        x=[str(y) for y in yr_econ_agg["year"]],
        y=yr_econ_agg["n_under"],
        name="🔻 Under-forecast govs",
        marker_color=C_OLIVE, opacity=0.85,
        text=yr_econ_agg["n_under"], textposition="inside",
        textfont=dict(size=11, color="white")
    ), row=1, col=2)
    fig_yr_econ.update_layout(
        barmode="group", height=360,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", size=12, color=C_INK),
        margin=dict(l=60, r=30, t=80, b=80),
        legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)")
    )
    fig_yr_econ.update_xaxes(tickfont=dict(size=11))
    fig_yr_econ.update_yaxes(tickfont=dict(size=11))
    max_g = yr_econ_agg["total_gap_usd"].abs().max() / 1e6
    fig_yr_econ.update_yaxes(range=[-max_g * 1.35, max_g * 1.35], row=1, col=1)
    st.plotly_chart(fig_yr_econ, use_container_width=True)

    st.markdown("""
    <div class="callout warning">
        <span class="callout-icon">📌</span>
        <div><strong>Interpretation note:</strong> Economic figures are illustrative estimates based on simplified
        assumptions. Real export value depends on contract prices, oil quality grades, and logistics costs.
        The directional signal (🔺 over vs. 🔻 under) is more meaningful than the absolute dollar amount.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 6 — MODEL EVALUATION
# ══════════════════════════════════════════
with tab6:
    st.markdown("""
    <div class="sec-header"><span class="sec-title">Model Evaluation & Error Diagnostics</span></div>
    <div class="sec-sub">MAPE · SMAPE · R² · MAE · RMSE — baseline vs hybrid comparison · per-governorate diagnostics</div>
    """, unsafe_allow_html=True)

    comparison = pd.DataFrame([
        {"Model": "📌 Baseline (Lag-1)", "MAPE": baseline_metrics["MAPE"],
         "SMAPE": baseline_metrics["SMAPE"], "MAE": baseline_metrics["MAE"],
         "RMSE": baseline_metrics["RMSE"], "R²": baseline_metrics["R2"]},
        {"Model": "🔮 Hybrid (0.98/0.02)", "MAPE": metrics["MAPE"],
         "SMAPE": metrics["SMAPE"], "MAE": metrics["MAE"],
         "RMSE": metrics["RMSE"], "R²": metrics["R2"]}
    ])

    # Metric table with arrows
    mape_delta  = baseline_metrics["MAPE"] - metrics["MAPE"]
    smape_delta = baseline_metrics["SMAPE"] - metrics["SMAPE"]
    mae_delta   = baseline_metrics["MAE"]   - metrics["MAE"]
    rmse_delta  = baseline_metrics["RMSE"]  - metrics["RMSE"]
    r2_delta    = metrics["R2"] - baseline_metrics["R2"]

    def delta_badge(d, positive_good=True):
        better = d > 0 if positive_good else d < 0
        icon   = "✅" if better else "🔴"
        sign   = "+" if d > 0 else ""
        return f'{icon} <b style="color:{"#4A6741" if better else "#C8102E"}">{sign}{d:.2f}</b>'

    st.markdown(f"""
    <div class="metric-table">
        <div class="metric-row header">
            <span>Metric</span><span>Baseline</span><span>Hybrid</span><span>Δ Change</span>
        </div>
        <div class="metric-row">
            <span><b>MAPE</b> <span style="font-size:11px;color:{C_MUTED};">(lower=better)</span></span>
            <span style="color:{C_EARTH};font-weight:700;">{baseline_metrics['MAPE']:.1f}%</span>
            <span style="color:{C_OLIVE};font-weight:700;">{metrics['MAPE']:.1f}%</span>
            <span>{delta_badge(mape_delta, True)}</span>
        </div>
        <div class="metric-row">
            <span><b>SMAPE</b> <span style="font-size:11px;color:{C_MUTED};">(lower=better)</span></span>
            <span style="color:{C_EARTH};font-weight:700;">{baseline_metrics['SMAPE']:.1f}%</span>
            <span style="color:{C_OLIVE};font-weight:700;">{metrics['SMAPE']:.1f}%</span>
            <span>{delta_badge(smape_delta, True)}</span>
        </div>
        <div class="metric-row">
            <span><b>MAE</b> <span style="font-size:11px;color:{C_MUTED};">(lower=better)</span></span>
            <span style="color:{C_EARTH};font-weight:700;">{baseline_metrics['MAE']:,.0f} t</span>
            <span style="color:{C_OLIVE};font-weight:700;">{metrics['MAE']:,.0f} t</span>
            <span>{delta_badge(mae_delta, True)}</span>
        </div>
        <div class="metric-row">
            <span><b>RMSE</b> <span style="font-size:11px;color:{C_MUTED};">(lower=better)</span></span>
            <span style="color:{C_EARTH};font-weight:700;">{baseline_metrics['RMSE']:,.0f} t</span>
            <span style="color:{C_OLIVE};font-weight:700;">{metrics['RMSE']:,.0f} t</span>
            <span>{delta_badge(rmse_delta, True)}</span>
        </div>
        <div class="metric-row">
            <span><b>R²</b> <span style="font-size:11px;color:{C_MUTED};">(higher=better)</span></span>
            <span style="color:{C_EARTH};font-weight:700;">{baseline_metrics['R2']:.4f}</span>
            <span style="color:{C_OLIVE};font-weight:700;">{metrics['R2']:.4f}</span>
            <span>{delta_badge(r2_delta, True)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        # MAPE + SMAPE grouped bar — with enough margin
        fig_cmp = go.Figure()
        for metric_name, color in [("MAPE", C_RED), ("SMAPE", C_OLIVE)]:
            fig_cmp.add_trace(go.Bar(
                name=metric_name,
                x=comparison["Model"],
                y=comparison[metric_name],
                marker_color=color,
                text=[f"<b>{v:.1f}%</b>" for v in comparison[metric_name]],
                textposition="outside",
                textfont=dict(size=12, color=C_INK),
                cliponaxis=False
            ))
        # Improvement arrow
        arrow_txt = f"{'▼ Improved' if mape_delta>0 else '▲ Worse'} {abs(mape_delta):.1f} pp"
        arrow_col = C_OLIVE if mape_delta > 0 else C_RED
        fig_cmp.add_annotation(
            x=1, y=metrics["MAPE"] + 12,
            text=arrow_txt,
            showarrow=True, arrowhead=2, arrowcolor=arrow_col,
            ax=0, ay=-35, font=dict(size=11, color=arrow_col, family="DM Sans"),
            bgcolor="rgba(255,255,255,0.9)", borderpad=5,
            bordercolor=arrow_col, borderwidth=1.5
        )
        fig_cmp.update_layout(
            barmode="group",
            title="MAPE & SMAPE — Baseline vs Hybrid (lower is better)",
            legend=dict(orientation="h", y=1.08, x=0),
            bargap=0.3
        )
        fig_base(fig_cmp, 420, 80, 90)
        fig_cmp.update_yaxes(range=[0, max(comparison["MAPE"].max(), comparison["SMAPE"].max()) * 1.3])
        st.plotly_chart(fig_cmp, use_container_width=True)

    with col_m2:
        # Radar chart
        cats = ["MAPE ↓", "SMAPE ↓", "MAE ↓", "RMSE ↓", "R² ↑"]
        def norm(v, metric):
            if metric == "R²":
                return max(0, min(1, v))
            return max(0, 1 - v / 200)

        base_vals   = [
            norm(baseline_metrics["MAPE"], "MAPE"), norm(baseline_metrics["SMAPE"], "SMAPE"),
            norm(baseline_metrics["MAE"] / 1000, "MAE"), norm(baseline_metrics["RMSE"] / 1000, "RMSE"),
            baseline_metrics["R2"]
        ]
        hybrid_vals = [
            norm(metrics["MAPE"], "MAPE"), norm(metrics["SMAPE"], "SMAPE"),
            norm(metrics["MAE"] / 1000, "MAE"), norm(metrics["RMSE"] / 1000, "RMSE"),
            metrics["R2"]
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=base_vals + [base_vals[0]], theta=cats + [cats[0]],
            fill="toself", name="📌 Baseline",
            line_color=C_EARTH, fillcolor="rgba(139,94,52,0.15)"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=hybrid_vals + [hybrid_vals[0]], theta=cats + [cats[0]],
            fill="toself", name="🔮 Hybrid",
            line_color=C_OLIVE, fillcolor="rgba(74,103,65,0.15)"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9))),
            showlegend=True,
            title="Model Performance Radar (normalized · higher = better)",
            legend=dict(orientation="h", y=-0.12, x=0.2)
        )
        fig_base(fig_radar, 420, 60)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Actual vs Predicted scatter
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        fig_scatter_ap = go.Figure()
        max_prod = pred_df["production"].max()
        fig_scatter_ap.add_trace(go.Scatter(
            x=[0, max_prod], y=[0, max_prod],
            mode="lines", name="✅ Perfect Forecast",
            line=dict(color=C_INK, width=1.5, dash="dash"), opacity=0.35
        ))
        fig_scatter_ap.add_trace(go.Scatter(
            x=pred_df["production"],
            y=pred_df["final_prediction"],
            mode="markers",
            marker=dict(
                size=8, color=pred_df["ape_percent"],
                colorscale=[[0, C_OLIVE], [0.5, C_GOLD], [1, C_RED]],
                showscale=True, opacity=0.8,
                colorbar=dict(title="MAPE %", thickness=12, len=0.7, x=1.02),
                line=dict(color="white", width=0.8)
            ),
            hovertext=[
                f"<b>{nice_name(g)}</b> · {y}<br>Actual: {a:,.0f} t<br>Predicted: {p:,.0f} t<br>MAPE: {m:.1f}%"
                f"<br>{'🔺 Over' if s>0 else '🔻 Under'}-forecast"
                for g, y, a, p, m, s in zip(
                    pred_df["governorate"], pred_df["year"],
                    pred_df["production"], pred_df["final_prediction"],
                    pred_df["ape_percent"], pred_df["signed_error"]
                )
            ],
            hoverinfo="text",
            name="Governorate-Year"
        ))
        # Annotate quadrants
        fig_scatter_ap.add_annotation(
            x=max_prod * 0.1, y=max_prod * 0.85,
            text="🔺 Over-forecast region",
            showarrow=False, font=dict(size=10, color=C_RED, family="DM Sans"),
            bgcolor="rgba(255,240,242,0.8)", borderpad=4
        )
        fig_scatter_ap.add_annotation(
            x=max_prod * 0.7, y=max_prod * 0.15,
            text="🔻 Under-forecast region",
            showarrow=False, font=dict(size=10, color=C_OLIVE, family="DM Sans"),
            bgcolor="rgba(240,246,234,0.8)", borderpad=4
        )
        fig_scatter_ap.update_layout(
            title="Actual vs Predicted — All Governorate-Years<br><sup>Dashed = perfect forecast · Color = MAPE · Hover for details</sup>",
            xaxis_title="Actual Production (t)",
            yaxis_title="Predicted Production (t)"
        )
        fig_base(fig_scatter_ap, 440, 90, 80, 70)
        st.plotly_chart(fig_scatter_ap, use_container_width=True)

    with col_s2:
        # MAPE histogram with clear zones
        fig_hist = px.histogram(
            pred_df, x="ape_percent", nbins=28,
            color_discrete_sequence=[C_OLIVE],
            title="MAPE Distribution — All Governorate-Years<br><sup>Left tail = accurate cases · Long right tail = alternate-bearing outliers</sup>"
        )
        med_val = pred_df["ape_percent"].median()
        mea_val = pred_df["ape_percent"].mean()
        fig_hist.add_vrect(x0=0, x1=25, fillcolor="rgba(74,103,65,0.07)", line_width=0,
                           annotation_text="✅ Good zone<br>(MAPE < 25%)",
                           annotation_font=dict(size=10, color=C_OLIVE),
                           annotation_position="top left")
        fig_hist.add_vrect(
            x0=50, x1=pred_df["ape_percent"].max() + 5,
            fillcolor="rgba(200,16,46,0.07)", line_width=0,
            annotation_text="🔴 Difficult zone<br>(MAPE > 50%)",
            annotation_font=dict(size=10, color=C_RED),
            annotation_position="top right"
        )
        fig_hist.add_vline(
            x=med_val, line_color=C_GOLD, line_width=2.5, line_dash="dash",
            annotation_text=f"⬛ Median {med_val:.1f}%",
            annotation_font=dict(size=11, color=C_GOLD),
            annotation_position="top left"
        )
        fig_hist.add_vline(
            x=mea_val, line_color=C_RED, line_width=2.5, line_dash="dot",
            annotation_text=f"🔴 Mean {mea_val:.1f}%",
            annotation_font=dict(size=11, color=C_RED),
            annotation_position="top right"
        )
        fig_base(fig_hist, 440, 90, 80)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Governorate MAPE ranking
    gov_summary = pred_df.groupby("governorate").agg(
        mean_actual     = ("production", "mean"),
        mean_prediction = ("final_prediction", "mean"),
        MAPE            = ("ape_percent", "mean"),
        MAE             = ("absolute_error", "mean"),
        n_years         = ("year", "count")
    ).reset_index().sort_values("MAPE", ascending=False)

    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        top10 = gov_summary.head(10).copy()
        top10["gov_name"] = top10["governorate"].map(nice_name)
        top10["label"] = top10.apply(
            lambda r: f"🔴 {r['MAPE']:.1f}%" if r["MAPE"] > 50 else (f"⚠️ {r['MAPE']:.1f}%" if r["MAPE"] > 25 else f"✅ {r['MAPE']:.1f}%"),
            axis=1
        )
        fig_govrank = go.Figure(go.Bar(
            x=top10["MAPE"],
            y=top10["gov_name"],
            orientation="h",
            marker=dict(
                color=top10["MAPE"],
                colorscale=[[0, C_GOLD], [0.5, C_EARTH], [1, C_RED]],
                showscale=False
            ),
            text=top10["label"],
            textposition="outside",
            textfont=dict(size=11, color=C_INK),
            cliponaxis=False
        ))
        fig_govrank.update_layout(
            title="10 Most Challenging Governorates by MAPE",
            xaxis_title="Mean MAPE (%)",
            yaxis={"categoryorder": "total ascending"}
        )
        fig_base(fig_govrank, 440, 60, 80, 130)
        fig_govrank.update_xaxes(range=[0, top10["MAPE"].max() * 1.3])
        st.plotly_chart(fig_govrank, use_container_width=True)

    with col_g2:
        # Production volume vs MAPE scatter with quadrant annotation
        fig_quad = px.scatter(
            pred_df,
            x="production", y="ape_percent",
            size="absolute_error", color="year",
            hover_name="governorate",
            color_continuous_scale=[[0, C_OLIVE], [0.5, C_GOLD], [1, C_RED]],
            title="Production Volume vs. MAPE<br><sup>Small volume → inflated MAPE · Bubble size = absolute error</sup>",
            labels={"production": "Actual Production (t)", "ape_percent": "APE (%)"}
        )
        fig_quad.add_hline(y=50, line_color=C_RED, line_width=1.5, line_dash="dash",
                           annotation_text="⚠️ High-error threshold (50%)",
                           annotation_font=dict(size=10, color=C_RED),
                           annotation_position="top right")
        fig_quad.add_annotation(
            x=pred_df["production"].quantile(0.12),
            y=pred_df["ape_percent"].quantile(0.78),
            text="🔴 Low-volume cases<br>inflate MAPE artificially",
            showarrow=True, arrowhead=2, arrowcolor=C_RED,
            ax=40, ay=-30,
            font=dict(size=10, color=C_RED, family="DM Sans"),
            bgcolor="rgba(255,240,242,0.88)", borderpad=5
        )
        fig_quad.update_traces(marker=dict(opacity=0.75, line=dict(color="white", width=0.8)))
        fig_base(fig_quad, 440, 90, 80)
        st.plotly_chart(fig_quad, use_container_width=True)

    # Year-over-year MAPE trend per governorate (box plot)
    st.markdown("""
    <div class="sec-header"><span class="sec-title" style="font-size:20px;">MAPE Distribution by Year — Box Plot</span></div>
    <div class="sec-sub">Spread shows inter-governorate variance · Outliers = alternate-bearing crash cases</div>
    """, unsafe_allow_html=True)

    fig_box = go.Figure()
    for yr in sorted(pred_df["year"].unique()):
        yr_data = pred_df[pred_df["year"] == yr]["ape_percent"]
        yr_med  = yr_data.median()
        box_col = C_OLIVE if yr_med < 40 else C_RED
        fig_box.add_trace(go.Box(
            y=yr_data, name=str(yr),
            marker_color=box_col, line_color=box_col,
            fillcolor=f"rgba{'(74,103,65,0.3)' if yr_med < 40 else '(200,16,46,0.3)'}",
            boxpoints="outliers",
            hovertemplate="Year: " + str(yr) + "<br>MAPE: %{y:.1f}%<extra></extra>"
        ))
    fig_box.add_hline(y=50, line_color=C_RED, line_dash="dash", line_width=1.5,
                      annotation_text="🔴 50% threshold",
                      annotation_font=dict(size=10, color=C_RED))
    fig_box.update_layout(
        title="MAPE Box Plot by Validation Year — All Governorates",
        yaxis_title="APE (%)", showlegend=False
    )
    fig_base(fig_box, 380, 60, 80)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
    <div class="callout warning">
        <span class="callout-icon">⚠️</span>
        <div><strong>Main limitation:</strong> Governorate MAPE is structurally inflated by near-zero production years
        (alternate-bearing crash years). A governorate producing 500 t when we predict 1,500 t generates 200% MAPE —
        even though the absolute error is small in national context. SMAPE and MAE give a more robust picture.</div>
    </div>
    """, unsafe_allow_html=True)

    gov_display2 = gov_summary.copy()
    gov_display2["governorate"] = gov_display2["governorate"].map(nice_name)
    gov_display2["status"] = gov_display2["MAPE"].apply(
        lambda v: "🔴 Difficult" if v > 50 else ("⚠️ Moderate" if v > 25 else "✅ Good")
    )
    gov_display2.columns = ["Governorate", "Avg Actual (t)", "Avg Predicted (t)", "APE (%)", "MAE (t)", "Years", "Status"]
    st.dataframe(gov_display2.round(1), use_container_width=True)


# ══════════════════════════════════════════
# TAB 7 — ARCHITECTURE
# ══════════════════════════════════════════
with tab7:
    st.markdown("""
    <div class="sec-header"><span class="sec-title">System Architecture & Design</span></div>
    <div class="sec-sub">CRISP-DM methodology · Three-layer design · Hybrid formula · Feature signals</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-hero">
        <div class="formula-label">Final Prediction Formula</div>
        <div class="formula-text">
            <em>ŷ</em> = 0.98 × <em>Baseline<sub>t−1</sub></em> + 0.02 × <em>XGBoost(X<sub>env</sub>)</em>
        </div>
        <div style="color:rgba(255,255,255,0.38);font-size:12px;margin-top:10px;font-family:'DM Sans';">
            Baseline = previous year production &nbsp;·&nbsp; XGBoost trained on log₁p(production)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="story-row">
        <div class="story-card">
            <div class="story-card-icon olive">📥</div>
            <h3>1 · Data Layer</h3>
            <p>Agricultural records, ERA5 climate, CHIRPS rainfall, NDVI vegetation indices (peak, min, amplitude, seasonal diff),
            and soil moisture merged by governorate × year. Lag features, rolling means, growth rates, and interaction terms engineered.</p>
        </div>
        <div class="story-card">
            <div class="story-card-icon gold">🤖</div>
            <h3>2 · Modeling Layer</h3>
            <p>Persistence baseline (lag-1) anchors stability for this small panel dataset.
            XGBoost adds nonlinear environmental correction. 98/2 blend deliberately conservative
            — prevents overfitting in low-data regime.</p>
        </div>
        <div class="story-card">
            <div class="story-card-icon red">📤</div>
            <h3>3 · Decision Layer</h3>
            <p>Governorate forecasts aggregate to national totals. Outputs feed interactive visualization,
            error diagnostics, economic impact analysis, and future deployment for CRDA offices and exporters.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout gold-box">
        <span class="callout-icon">🔁</span>
        <div><strong>CRISP-DM:</strong> Business understanding → Data understanding →
        Data preparation → Modeling → Evaluation → Deployment preparation.
        Each architecture layer maps directly to a CRISP-DM phase.</div>
    </div>
    """, unsafe_allow_html=True)

    # CRISP-DM pipeline diagram (HTML)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0;margin:24px 0;position:relative;">
        {"".join([
            f'<div style="background:{"#4A6741" if i%2==0 else "#C9A227"};color:white;padding:16px 10px;text-align:center;font-size:12px;font-weight:600;position:relative;">'
            f'<div style="font-size:20px;margin-bottom:6px;">{icon}</div>{label}'
            f'{"<div style=position:absolute;right:-12px;top:50%;transform:translateY(-50%);font-size:18px;color:#0F1A0D;z-index:10;>→</div>" if i < 5 else ""}'
            f'</div>'
            for i, (icon, label) in enumerate([
                ("🎯","Business<br>Understanding"),("🔍","Data<br>Understanding"),
                ("🧹","Data<br>Preparation"),("🤖","Modeling"),
                ("📊","Evaluation"),("🚀","Deployment")
            ])
        ])}
    </div>
    """, unsafe_allow_html=True)

    if not feature_importance_df.empty:
        fi_sorted = feature_importance_df.sort_values("importance", ascending=True).copy()
        fi_sorted["category"] = fi_sorted["feature"].apply(
            lambda f: "🌿 NDVI" if "ndvi" in f.lower() else
                      ("🌡 Climate" if any(x in f.lower() for x in ["temp", "prec", "moist", "rain", "chirps", "winter", "spring"]) else
                       ("📅 Time" if any(x in f.lower() for x in ["year", "index"]) else
                        ("🫒 Production" if any(x in f.lower() for x in ["lag", "roll", "growth", "surface"]) else "🔧 Other")))
        )
        cat_colors = {
            "🌿 NDVI": C_OLIVE, "🌡 Climate": C_RED,
            "📅 Time": C_EARTH, "🫒 Production": C_GOLD, "🔧 Other": C_MUTED
        }

        fig_fi = go.Figure()
        for cat in fi_sorted["category"].unique():
            sub = fi_sorted[fi_sorted["category"] == cat]
            fig_fi.add_trace(go.Bar(
                x=sub["importance"],
                y=sub["feature"],
                orientation="h",
                name=cat,
                marker_color=cat_colors.get(cat, C_MUTED),
                text=[f"{v:.4f}" for v in sub["importance"]],
                textposition="outside",
                textfont=dict(size=9, color=C_INK),
                cliponaxis=False
            ))

        fig_fi.update_layout(
            barmode="stack",
            title="XGBoost Feature Importances — Top 20 Predictors (colored by category)",
            xaxis_title="Importance Score",
            legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)")
        )
        fig_base(fig_fi, 640, 80, 80, 200)
        fig_fi.update_xaxes(range=[0, fi_sorted["importance"].max() * 1.3])
        st.plotly_chart(fig_fi, use_container_width=True)


# ── Footer ───────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:36px 0 16px;color:{C_MUTED};font-size:12.5px;">
    <div style="font-family:'Playfair Display',serif;font-size:22px;color:{C_GOLD};margin-bottom:6px;">زيتون</div>
    Tunisia Olive Production Intelligence · XGBoost · ERA5 · CHIRPS · NDVI · Soil Moisture
</div>
""", unsafe_allow_html=True)
