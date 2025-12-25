import io
import base64
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page Configuration
st.set_page_config(page_title="TalhaBashir | COVID-19 Intelligence", layout="wide", page_icon="🛡️")

# --- CUSTOM COLOR PALETTE ---
PRIMARY_COLOR = "#008080"  # Teal
SECONDARY_COLOR = "#DAA520" # Goldenrod
ACCENT_COLOR = "#FF7F50"    # Coral
BG_GRADIENT = "linear-gradient(135deg, #002b36 0%, #004d4d 100%)"

# --- ADVANCED CSS INJECTION ---
st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background: {BG_GRADIENT};
        color: #e0f2f1;
    }}

    /* Global Header Redesign */
    .main-header {{
        background: rgba(255, 255, 255, 0.05);
        padding: 2.5rem;
        border-radius: 15px;
        border-left: 10px solid {SECONDARY_COLOR};
        margin-bottom: 2.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }}

    /* Metric Card Styling */
    [data-testid="stMetric"] {{
        background: rgba(0, 128, 128, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid {SECONDARY_COLOR};
        transition: transform 0.3s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: scale(1.02);
        background: rgba(0, 128, 128, 0.3);
    }}

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {{
        background-color: #001a1a !important;
        border-right: 1px solid {SECONDARY_COLOR};
    }}

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: rgba(218, 165, 32, 0.1);
        border-radius: 5px 5px 0 0;
        color: white;
        padding: 0 20px;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(218, 165, 32, 0.3);
    }}

    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {PRIMARY_COLOR};
        border-bottom: 3px solid {SECONDARY_COLOR};
    }}

    /* Buttons */
    .stButton > button {{
        width: 100%;
        border-radius: 25px;
        border: 2px solid {SECONDARY_COLOR};
        background: transparent;
        color: white;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background: {SECONDARY_COLOR};
        color: #002b36;
    }}

    .info-box {{
        background: rgba(255, 127, 80, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-right: 4px solid {ACCENT_COLOR};
    }}
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown(f"""
<div class="main-header">
    <h1 style='margin:0; color: {SECONDARY_COLOR}; font-family: "Trebuchet MS";'>🛡️ COVID-19 Strategic Analysis Portal</h1>
    <h3 style='margin:0.2rem 0; font-weight: 300;'>Predictive Insights & Healthcare Surveillance</h3>
    <hr style="border-color: {SECONDARY_COLOR}; opacity: 0.3;">
    <p style='margin:0; font-size: 1rem; opacity: 0.8;'>
        Principal Investigator: <strong>TalhaBashir</strong> | ID: <strong>2430-0162</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data(show_spinner="Accessing Global Databases...")
def load_data():
    try:
        url = "https://raw.githubusercontent.com/COVID19Tracking/covid-tracking-data/master/data/states_daily_4pm_et.csv"
        df = pd.read_csv(url)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values(['state', 'date'])
        
        cols_to_fix = ['positive', 'death', 'hospitalizedCurrently', 'positiveIncrease']
        for col in cols_to_fix:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Database Connection Failed: {e}")
        return pd.DataFrame()

df = load_data()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913447.png", width=80)
    st.markdown("## ⚙️ SYSTEM CONTROLS")
    
    selected_states = st.multiselect(
        "Target Jurisdictions",
        options=sorted(df['state'].unique()),
        default=['NY', 'CA', 'WA']
    )
    
    date_range = st.date_input(
        "Temporal Window",
        value=(df['date'].min(), df['date'].max())
    )
    
    st.markdown("---")
    st.markdown("### 👤 ANALYST PROFILE")
    st.info(f"**Name:** TalhaBashir\n\n**Reg ID:** 2430-0162")
    
    if st.button("♻️ Reset System"):
        st.rerun()

# Data Filtering
mask = (df['state'].isin(selected_states)) & (df['date'] >= pd.to_datetime(date_range[0]))
if len(date_range) > 1:
    mask &= (df['date'] <= pd.to_datetime(date_range[1]))
filtered_df = df[mask]

# --- MAIN LAYOUT TABS ---
t1, t2, t3, t4 = st.tabs(["📊 Executive Summary", "📈 Propagation Trends", "🤖 Predictive ML", "📂 Data Vault"])

# TAB 1: EXECUTIVE SUMMARY
with t1:
    st.subheader("Current Surveillance Status")
    latest = df[df['date'] == df['date'].max()]
    
    # Redesigned Layout: Metrics at top in a clean row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confirmed Cases", f"{latest['positive'].sum():,.0f}", "Global Total")
    m2.metric("Mortality", f"{latest['death'].sum():,.0f}", "Cumulative")
    m3.metric("Clinical Load", f"{latest['hospitalizedCurrently'].sum():,.0f}", "Current")
    m4.metric("Positivity Rate", f"{(latest['positiveIncrease'].sum()/1000):.2f}%", "Daily Avg")

    st.markdown("### System-Wide Distribution")
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # State Comparison Bar Chart
        fig_bar = px.bar(
            latest.nlargest(10, 'positive'),
            x='state', y='positive',
            color='positive',
            color_continuous_scale='Tealgrn',
            title="Top 10 States by Volume"
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            <h4 style='color:{SECONDARY_COLOR}'>Analyst Insight</h4>
            The current visualization focuses on the highest-impact zones within the selected temporal window. 
            <b>TalhaBashir</b> suggests monitoring the hospitalization-to-ICU ratio for system stress indicators.
        </div>
        """, unsafe_allow_html=True)

# TAB 2: PROPAGATION TRENDS
with t2:
    st.subheader("Temporal Progression Analysis")
    
    metric_choice = st.selectbox("Select Telemetry Stream", ["positiveIncrease", "deathIncrease", "hospitalizedCurrently"])
    
    fig_trend = px.line(
        filtered_df, 
        x='date', y=metric_choice, 
        color='state',
        line_shape='spline',
        render_mode='svg',
        title=f"Time-Series: {metric_choice.replace('Increase', ' Daily')}"
    )
    fig_trend.update_layout(
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# TAB 3: PREDICTIVE ML
with t3:
    st.subheader("🤖 Machine Learning Projection")
    st.write("Model: Random Forest Regressor | Target: Daily New Cases")
    
    if st.button("⚡ Initialize ML Training Sequence"):
        # Quick data prep
        ml_data = filtered_df.dropna(subset=['positive', 'death', 'positiveIncrease'])
        X = ml_data[['positive', 'death', 'hospitalizedCurrently']].fillna(0)
        y = ml_data['positiveIncrease']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("Model Training Complete")
            st.metric("Model Reliability (R²)", f"{r2_score(y_test, preds):.4f}")
        with c2:
            st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, preds):,.2f}")
            
        # Prediction Chart
        res_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds}).reset_index()
        fig_res = px.scatter(res_df, x='Actual', y='Predicted', 
                           marginal_x="histogram", marginal_y="rug",
                           color_discrete_sequence=[ACCENT_COLOR])
        fig_res.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                        line=dict(color=SECONDARY_COLOR, dash="dash"))
        st.plotly_chart(fig_res, use_container_width=True)

# TAB 4: DATA VAULT
with t4:
    st.subheader("Binary Data Export")
    st.dataframe(filtered_df.style.background_gradient(cmap='Blues'), height=400)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Secure Dataset",
        csv,
        "TalhaBashir_COVID_Data.csv",
        "text/csv",
        use_container_width=True
    )

# --- FOOTER ---
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; opacity: 0.6; padding: 20px;'>
    <small>Data Processing Unit: TalhaBashir (2430-0162) | Source: COVID Tracking Project | Built with Streamlit & Python</small>
</div>
""", unsafe_allow_html=True)