import numpy as np
import pandas as pd
from datetime import datetime

import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Laptop Price Analytics",
    layout="wide",
    page_icon="💻"
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Laptop Price Analytics",
    layout="wide",
    page_icon="💻"
)

# =====================================================
# PROFESSIONAL CORPORATE THEME (FIXED)
# =====================================================
PRIMARY_COLOR = "#2563EB"     # Professional Blue (Less dark, more vibrant)
BG_COLOR = "#F8FAFC"          # Very Light Gray-Blue
CARD_BG = "#FFFFFF"           # White
TEXT_COLOR = "#1E293B"        # Slate 800 (High contrast dark gray)
SUBTEXT_COLOR = "#64748B"     # Slate 500

st.markdown(f"""
<style>

/* -------- General App -------- */
body, .stApp {{
  background-color: {BG_COLOR};
}}

/* -------- Typography -------- */
h1, h2, h3, h4, h5, h6 {{
  color: {TEXT_COLOR} !important;
  font-family: 'Inter', sans-serif; /* Clean modern font */
}}

p, div, label, span {{
  color: {TEXT_COLOR};
}}

/* -------- Header Card (FIXED) -------- */
.header-card {{
  background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
  padding: 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}}

/* Force Header Text to be White */
.header-card h1 {{
  color: #FFFFFF !important;
  font-weight: 700;
  margin-bottom: 0.5rem;
}}

.header-card p {{
  color: #E2E8F0 !important; /* Light gray for subtitles */
  font-size: 1rem;
  margin: 0;
}}

/* -------- Tabs (Clean Underline Style) -------- */
.stTabs [data-baseweb="tab-list"] {{
  background-color: transparent;
  gap: 2rem;
  border-bottom: 1px solid #E2E8F0;
}}

.stTabs [data-baseweb="tab"] {{
  background-color: transparent;
  color: {SUBTEXT_COLOR};
  font-weight: 600;
  padding-bottom: 1rem;
  border: none;
}}

.stTabs [aria-selected="true"] {{
  color: {PRIMARY_COLOR} !important;
  border-bottom: 3px solid {PRIMARY_COLOR};
}}

/* -------- Metric Cards -------- */
[data-testid="stMetric"] {{
  background-color: {CARD_BG};
  border: 1px solid #E2E8F0;
  border-radius: 10px;
  padding: 1.5rem;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}}

[data-testid="stMetricLabel"] {{
  color: {SUBTEXT_COLOR} !important;
  font-size: 0.9rem;
}}

[data-testid="stMetricValue"] {{
  color: {TEXT_COLOR} !important;
  font-weight: 700;
}}

/* -------- Buttons -------- */
.stButton > button {{
  background-color: {PRIMARY_COLOR};
  color: white;
  border-radius: 8px;
  border: none;
  padding: 0.6rem 1.2rem;
  font-weight: 500;
  transition: all 0.2s;
}}

.stButton > button:hover {{
  background-color: #1E40AF; /* Darker blue on hover */
  transform: translateY(-1px);
}}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header-card">
  <h1>💻 Laptop Price Analytics Dashboard</h1>
  <p>Market Analysis & Machine Learning Price Prediction</p>
  <p><b>Talha Bashir , Abdullah</b> | Roll No: 2430-0162 , 2430-0067 | PAI Course Project</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("laptopData.csv")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df["Ram_GB"] = df["Ram"].str.extract(r"(\d+)").astype(float)
    df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")
    df["Weight_kg"] = df["Weight"].str.extract(r"(\d+\.?\d*)").astype(float)

    df["Price_Category"] = pd.cut(
        df["Price"],
        bins=[0, 30000, 60000, 100000, np.inf],
        labels=["Budget", "Mid-Range", "Premium", "Luxury"]
    )

    df.dropna(subset=["Company", "TypeName"], inplace=True)
    return df


df = load_data()

if df.empty:
    st.error("Dataset not found or empty.")
    st.stop()

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🎛️ Filters")

companies = sorted(df["Company"].dropna().unique())
types = sorted(df["TypeName"].dropna().unique())

selected_companies = st.sidebar.multiselect(
    "Company", companies, companies[:5] if len(companies) >= 5 else companies
)
selected_types = st.sidebar.multiselect("Laptop Type", types, types)

price_min, price_max = float(df["Price"].min()), float(df["Price"].max())
price_range = st.sidebar.slider(
    "Price Range (₹)", price_min, price_max, (price_min, price_max)
)

filtered_df = df[
    (df["Company"].isin(selected_companies)) &
    (df["TypeName"].isin(selected_types)) &
    (df["Price"].between(price_range[0], price_range[1]))
]

if filtered_df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "🏠 Overview", "📊 Market", "💻 Specs",
    "📈 Trends", "🤖 ML Model", "💾 Export"
])

# =====================================================
# OVERVIEW
# =====================================================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Price", f"₹{filtered_df['Price'].mean():,.0f}")
    c2.metric("Total Laptops", len(filtered_df))
    c3.metric("Avg RAM", f"{filtered_df['Ram_GB'].mean():.1f} GB")
    c4.metric("Companies", filtered_df["Company"].nunique())

    fig = px.bar(
        filtered_df.groupby("Company")["Price"].mean()
        .sort_values(ascending=False).head(10),
        orientation="h",
        title="Top 10 Companies by Average Price",
        color_discrete_sequence=[PRIMARY_COLOR]
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# MARKET
# =====================================================
with tabs[1]:
    # 1. Create a proper DataFrame with names and counts
    share_df = filtered_df["Company"].value_counts().head(8).reset_index()
    share_df.columns = ["Company", "Count"]

    # 2. Explicitly tell Plotly what to use for names and values
    fig = px.pie(
        share_df,
        names="Company", 
        values="Count",
        hole=0.4,
        title="Market Share by Company"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# SPECS (NA-SAFE SCATTER)
# =====================================================
with tabs[2]:
    fig = px.box(
        filtered_df, x="Ram", y="Price",
        title="RAM vs Price"
    )
    st.plotly_chart(fig, use_container_width=True)

    scatter_df = filtered_df.dropna(
        subset=["Inches", "Price", "Weight_kg", "Company"]
    )

    if not scatter_df.empty:
        fig = px.scatter(
            scatter_df,
            x="Inches",
            y="Price",
            size="Weight_kg",
            color="Company",
            size_max=25,
            title="Screen Size vs Price (Bubble = Weight)"
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TRENDS
# =====================================================
with tabs[3]:
    fig = px.box(
        filtered_df,
        x="Price_Category",
        y="Price",
        title="Price Distribution by Category"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# MACHINE LEARNING
# =====================================================
with tabs[4]:
    st.subheader("🤖 Random Forest Price Prediction")

    n_estimators = st.slider("Trees", 50, 300, 150, 25)
    max_depth = st.slider("Max Depth", 5, 40, 20, 5)
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100

    if st.button("🚀 Train Model"):
        model_df = filtered_df.dropna(subset=[
            "Price", "Ram_GB", "Inches",
            "Weight_kg", "Company", "TypeName", "OpSys"
        ])

        le_c, le_t, le_o = LabelEncoder(), LabelEncoder(), LabelEncoder()
        model_df["Company_enc"] = le_c.fit_transform(model_df["Company"])
        model_df["Type_enc"] = le_t.fit_transform(model_df["TypeName"])
        model_df["OS_enc"] = le_o.fit_transform(model_df["OpSys"])

        X = model_df[[
            "Ram_GB", "Inches", "Weight_kg",
            "Company_enc", "Type_enc", "OS_enc"
        ]]
        y = model_df["Price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # --- NEW CODE FROM IMAGE STARTS HERE ---
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"₹{rmse:,.0f}")
        c2.metric("MAE", f"₹{mae:,.0f}")
        c3.metric("R²", f"{r2:.4f}")
        # --- NEW CODE ENDS HERE ---

# =====================================================
# EXPORT
# =====================================================
with tabs[5]:
    st.dataframe(filtered_df.head(20), use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered CSV",
        csv,
        f"laptops_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv"
    )