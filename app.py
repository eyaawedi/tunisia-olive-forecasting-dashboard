
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import folium
    from streamlit_folium import st_folium
    HAS_MAP = True
except Exception:
    HAS_MAP = False


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Tunisia Olive Intelligence",
    page_icon="🫒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# COLORS
# =====================================================
OLIVE = "#5F7F2A"
DARK = "#24351F"
GOLD = "#C9A227"
RED = "#C8102E"
CREAM = "#FFFDF4"
MUTED = "#6B705C"
LIGHT_GREEN = "#EEF5E5"


# =====================================================
# CSS
# =====================================================
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(180deg, {CREAM} 0%, #ffffff 50%, #F3F7EC 100%);
}}
.block-container {{
    padding-top: 1.6rem;
    max-width: 1500px;
}}
.hero {{
    background: linear-gradient(135deg, #24351F 0%, #5F7F2A 70%, #C9A227 100%);
    border-radius: 28px;
    padding: 34px 38px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 18px 40px rgba(36,53,31,0.22);
}}
.hero h1 {{
    font-size: 44px;
    margin: 0 0 8px 0;
    font-weight: 900;
}}
.hero p {{
    font-size: 17px;
    max-width: 920px;
    opacity: 0.95;
}}
.badge {{
    display:inline-block;
    margin-right:8px;
    margin-top:12px;
    padding:7px 13px;
    border-radius:999px;
    border:1px solid rgba(255,255,255,0.35);
    background:rgba(255,255,255,0.15);
    font-weight:700;
    font-size:13px;
}}
.card {{
    background: white;
    border-radius: 20px;
    padding: 20px 22px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 10px 24px rgba(0,0,0,0.06);
    min-height: 128px;
}}
.card-label {{
    color: {MUTED};
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: .6px;
    font-weight: 800;
}}
.card-value {{
    color: {DARK};
    font-size: 34px;
    font-weight: 900;
    margin-top: 6px;
}}
.card-note {{
    color: #777;
    font-size: 13px;
    margin-top: 5px;
}}
.info {{
    background: {LIGHT_GREEN};
    border-left: 6px solid {OLIVE};
    padding: 16px 18px;
    border-radius: 14px;
    margin: 16px 0;
    color: {DARK};
}}
.warning {{
    background: #FFF3E0;
    border-left: 6px solid {GOLD};
    padding: 16px 18px;
    border-radius: 14px;
    margin: 16px 0;
    color: #4A3510;
}}
section[data-testid="stSidebar"] {{
    background: {DARK};
}}
section[data-testid="stSidebar"] * {{
    color: white !important;
}}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HELPERS
# =====================================================
def clean_gov(x):
    return str(x).strip().lower()

display_names = {
    "benarous": "Ben Arous",
    "sidi bouzid": "Sidi Bouzid"
}
def nice_name(g):
    return display_names.get(g, str(g).title())

def mape(y_true, y_pred, epsilon=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def smape(y_true, y_pred, epsilon=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)) * 100

def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="card">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        <div class="card-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def fmt(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


# =====================================================
# LOAD DATA
# =====================================================
DATA_PATH = "final_master_dataset_fixed.csv"
PRED_PATH = "streamlit_predictions.csv"

@st.cache_data
def load_all():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["governorate"] = df["governorate"].apply(clean_gov)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "production" in df.columns:
        df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df = df.sort_values(["governorate", "year"]).reset_index(drop=True)

    pred = pd.read_csv(PRED_PATH)
    pred.columns = [c.strip() for c in pred.columns]
    pred["governorate"] = pred["governorate"].apply(clean_gov)

    numeric_cols = [
        "year", "production", "baseline_prediction", "xgb_prediction",
        "final_prediction", "absolute_error", "ape_percent", "signed_error"
    ]
    for col in numeric_cols:
        if col in pred.columns:
            pred[col] = pd.to_numeric(pred[col], errors="coerce")

    pred["year"] = pred["year"].astype(int)
    pred["direction"] = np.where(pred["signed_error"] >= 0, "Over-forecast", "Under-forecast")
    return df, pred

try:
    df, pred_df = load_all()
except FileNotFoundError as e:
    st.error(
        "Missing file. Make sure GitHub root contains: "
        "`app.py`, `requirements.txt`, `final_master_dataset_fixed.csv`, and `streamlit_predictions.csv`."
    )
    st.exception(e)
    st.stop()


# =====================================================
# METRICS
# =====================================================
gov_mape = mape(pred_df["production"], pred_df["final_prediction"])
gov_smape = smape(pred_df["production"], pred_df["final_prediction"])
gov_mae = np.mean(np.abs(pred_df["production"] - pred_df["final_prediction"]))
gov_rmse = np.sqrt(np.mean((pred_df["production"] - pred_df["final_prediction"]) ** 2))

national_df = pred_df.groupby("year").agg(
    actual_national=("production", "sum"),
    predicted_national=("final_prediction", "sum"),
    baseline_national=("baseline_prediction", "sum")
).reset_index()

national_df["national_ape"] = (
    np.abs(national_df["actual_national"] - national_df["predicted_national"])
    / np.maximum(np.abs(national_df["actual_national"]), 1)
) * 100

national_mape = national_df["national_ape"].mean()


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("## 🫒 Tunisia Olive Intelligence")
st.sidebar.markdown("Forecasting dashboard based on precomputed model outputs.")
st.sidebar.markdown("---")

years = sorted(pred_df["year"].dropna().unique())
selected_year = st.sidebar.selectbox("Validation year", years, index=len(years)-1)

governorates = sorted(pred_df["governorate"].unique())
selected_gov = st.sidebar.selectbox("Governorate", governorates, format_func=nice_name)

error_filter = st.sidebar.slider("Show APE above (%)", 0, 250, 0, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Final formula")
st.sidebar.code("ŷ = 0.98 × Baseline + 0.02 × XGBoost")
st.sidebar.caption("Baseline = previous-year production")


# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🇹🇳 Tunisia Olive Production Intelligence</h1>
    <p>
    A deployed decision-support dashboard for monitoring Tunisian olive production forecasts
    at governorate and national levels.
    </p>
    <span class="badge">21 Governorates</span>
    <span class="badge">Hybrid Forecast</span>
    <span class="badge">National Aggregation</span>
    <span class="badge">No Online Retraining</span>
</div>
""", unsafe_allow_html=True)


# =====================================================
# KPI CARDS
# =====================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Governorate MAPE", f"{gov_mape:.1f}%", "Primary regional metric")
with c2:
    metric_card("SMAPE", f"{gov_smape:.1f}%", "Stable percentage error")
with c3:
    metric_card("RMSE", fmt(gov_rmse), "Penalizes large errors")
with c4:
    metric_card("National MAPE", f"{national_mape:.1f}%", "Secondary target ≤ 15%")

st.markdown("""
<div class="info">
<b>Key insight:</b> the national error is much lower because governorate errors partially cancel after aggregation.
Governorate-level results are mainly useful for regional monitoring and risk detection.
</div>
""", unsafe_allow_html=True)


# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🇹🇳 Overview",
    "🗺️ Map",
    "🫒 Governorate",
    "📈 National",
    "📊 Evaluation"
])


# =====================================================
# TAB 1
# =====================================================
with tab1:
    st.subheader("Project Overview")

    a, b, c = st.columns(3)
    with a:
        st.markdown("""
        <div class="card">
            <div class="card-label">Data Sources</div>
            <div class="card-note">
            Agriculture records, ERA5 climate, CHIRPS rainfall, MODIS NDVI, and soil moisture.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="card">
            <div class="card-label">Model Logic</div>
            <div class="card-note">
            A strong persistence baseline is corrected slightly by XGBoost environmental signals.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c:
        st.markdown("""
        <div class="card">
            <div class="card-label">Decision Support</div>
            <div class="card-note">
            National planning plus governorate-level risk and uncertainty monitoring.
            </div>
        </div>
        """, unsafe_allow_html=True)

    hist = df.groupby("year")["production"].sum().reset_index()
    fig = px.line(
        hist,
        x="year",
        y="production",
        markers=True,
        title="Historical National Olive Production"
    )
    fig.update_traces(line=dict(color=OLIVE, width=4), marker=dict(size=9))
    fig.update_layout(height=470, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 2
# =====================================================
with tab2:
    st.subheader(f"Regional Forecast Map — {selected_year}")

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

    map_df = pred_df[pred_df["year"] == selected_year].copy()
    map_df["lat"] = map_df["governorate"].map(lambda x: gov_coords.get(x, (np.nan, np.nan))[0])
    map_df["lon"] = map_df["governorate"].map(lambda x: gov_coords.get(x, (np.nan, np.nan))[1])
    map_df = map_df.dropna(subset=["lat", "lon"])

    if HAS_MAP:
        m = folium.Map(location=[34.8, 9.6], zoom_start=6, tiles="CartoDB positron")
        for _, row in map_df.iterrows():
            ape = row["ape_percent"]
            color = "green" if ape < 20 else ("orange" if ape < 50 else "red")
            popup = f"""
            <b>{nice_name(row['governorate'])}</b><br>
            Year: {int(row['year'])}<br>
            Actual: {row['production']:,.0f}<br>
            Predicted: {row['final_prediction']:,.0f}<br>
            APE: {row['ape_percent']:.1f}%
            """
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6 + min(row["production"] / 50000, 12),
                tooltip=f"{nice_name(row['governorate'])}: {ape:.1f}% APE",
                popup=folium.Popup(popup, max_width=260),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75
            ).add_to(m)
        st_folium(m, width=None, height=540)
    else:
        st.warning("Map libraries are unavailable. Showing chart instead.")

    sorted_year = map_df.sort_values("ape_percent", ascending=False)
    fig = px.bar(
        sorted_year,
        x="ape_percent",
        y=sorted_year["governorate"].map(nice_name),
        orientation="h",
        color="ape_percent",
        color_continuous_scale=["#5F7F2A", "#C9A227", "#C8102E"],
        title=f"APE by Governorate — {selected_year}"
    )
    fig.update_layout(height=620, yaxis_title="", xaxis_title="APE (%)", plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# TAB 3
# =====================================================
with tab3:
    st.subheader(f"{nice_name(selected_gov)} Forecast Detail")

    row = pred_df[(pred_df["year"] == selected_year) & (pred_df["governorate"] == selected_gov)]

    if not row.empty:
        row = row.iloc[0]
        a, b, c, d = st.columns(4)
        with a:
            metric_card("Actual", fmt(row["production"]), "Observed production")
        with b:
            metric_card("Predicted", fmt(row["final_prediction"]), "Hybrid model forecast")
        with c:
            metric_card("Baseline", fmt(row["baseline_prediction"]), "Previous-year production")
        with d:
            metric_card("APE", f"{row['ape_percent']:.1f}%", row["direction"])

        comp = pd.DataFrame({
            "Type": ["Actual", "Baseline", "Final Prediction"],
            "Production": [row["production"], row["baseline_prediction"], row["final_prediction"]]
        })
        fig = px.bar(
            comp,
            x="Type",
            y="Production",
            color="Type",
            text="Production",
            color_discrete_map={
                "Actual": OLIVE,
                "Baseline": GOLD,
                "Final Prediction": RED
            },
            title=f"{nice_name(selected_gov)} — Actual vs Forecast ({selected_year})"
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(height=460, showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    trend = df[df["governorate"] == selected_gov].copy()
    pred_trend = pred_df[pred_df["governorate"] == selected_gov].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["year"], y=trend["production"],
        mode="lines+markers", name="Historical actual",
        line=dict(color=OLIVE, width=4)
    ))
    fig.add_trace(go.Scatter(
        x=pred_trend["year"], y=pred_trend["final_prediction"],
        mode="lines+markers", name="Validation prediction",
        line=dict(color=RED, width=4, dash="dash")
    ))
    fig.update_layout(
        title=f"{nice_name(selected_gov)} — Historical Production and Validation Forecast",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    year_table = pred_df[(pred_df["year"] == selected_year) & (pred_df["ape_percent"] >= error_filter)].copy()
    year_table["governorate"] = year_table["governorate"].map(nice_name)
    st.dataframe(year_table.sort_values("ape_percent", ascending=False), use_container_width=True)


# =====================================================
# TAB 4
# =====================================================
with tab4:
    st.subheader("National Aggregation")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=national_df["year"],
        y=national_df["actual_national"],
        mode="lines+markers",
        name="Actual national production",
        line=dict(color=OLIVE, width=4)
    ))
    fig.add_trace(go.Scatter(
        x=national_df["year"],
        y=national_df["predicted_national"],
        mode="lines+markers",
        name="Predicted national production",
        line=dict(color=RED, width=4, dash="dash")
    ))
    fig.update_layout(
        title="Actual vs Predicted National Production",
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning">
    <b>Interpretation:</b> National MAPE is the secondary planning objective.
    It is lower because errors across governorates partly cancel out.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(national_df, use_container_width=True)


# =====================================================
# TAB 5
# =====================================================
with tab5:
    st.subheader("Model Evaluation")

    results = pd.DataFrame([
        {"Level": "Governorate", "Metric": "MAPE", "Value": gov_mape},
        {"Level": "Governorate", "Metric": "SMAPE", "Value": gov_smape},
        {"Level": "Governorate", "Metric": "RMSE", "Value": gov_rmse},
        {"Level": "National", "Metric": "MAPE", "Value": national_mape},
    ])

    fig = px.bar(
        results,
        x="Metric",
        y="Value",
        color="Level",
        barmode="group",
        color_discrete_map={"Governorate": OLIVE, "National": RED},
        title="Evaluation Metrics"
    )
    fig.update_layout(height=460, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    gov_summary = pred_df.groupby("governorate").agg(
        mean_actual=("production", "mean"),
        mean_prediction=("final_prediction", "mean"),
        MAPE=("ape_percent", "mean"),
        MAE=("absolute_error", "mean")
    ).reset_index().sort_values("MAPE", ascending=False)

    gov_summary["governorate"] = gov_summary["governorate"].map(nice_name)

    fig = px.bar(
        gov_summary.head(10),
        x="MAPE",
        y="governorate",
        orientation="h",
        color="MAPE",
        color_continuous_scale=["#C9A227", "#C8102E"],
        title="Most Difficult Governorates by MAPE"
    )
    fig.update_layout(height=520, yaxis_title="", plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(gov_summary, use_container_width=True)
