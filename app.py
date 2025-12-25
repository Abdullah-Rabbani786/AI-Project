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
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Laptop Price Analytics",
    layout="wide",
    page_icon="💻"
)

# =====================================================
# CLEAN & MINIMAL THEME
# =====================================================
# A soft, airy palette with no heavy dark blocks
PRIMARY_COLOR = "#2563EB"      # A clean, standard blue for buttons only
BG_COLOR = "#FAFAFA"           # Almost white (Very clean)
CARD_BG = "#FFFFFF"            # Pure White
TEXT_COLOR = "#334155"         # Slate 700 (Softer than black)
HEADER_COLOR = "#0F172A"       # Slate 900 (Dark & crisp for titles)

st.markdown(f"""
<style>

/* -------- App Background -------- */
.stApp {{
  background-color: {BG_COLOR};
}}

/* -------- Typography -------- */
h1, h2, h3 {{
  color: {HEADER_COLOR} !important;
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  letter-spacing: -0.5px;
}}

/* -------- Header Card (Clean White) -------- */
.header-card {{
  background-color: {CARD_BG};
  padding: 2rem;
  border-radius: 12px;
  border: 1px solid #E2E8F0; /* Subtle border */
  margin-bottom: 2rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Very soft shadow */
}}

.header-card h1 {{
  color: {HEADER_COLOR} !important;
  margin-bottom: 0.2rem;
}}

.header-card p {{
  color: #64748B !important; /* Muted slate for subtitle */
  font-size: 1rem;
}}

/* -------- Metric Cards (Minimalist) -------- */
[data-testid="stMetric"] {{
  background-color: {CARD_BG};
  border: 1px solid #F1F5F9;
  border-radius: 10px;
  padding: 1.2rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}}

[data-testid="stMetricLabel"] {{
  color: #94A3B8 !important; /* Light gray label */
  font-size: 0.85rem;
  font-weight: 500;
}}

[data-testid="stMetricValue"] {{
  color: {HEADER_COLOR} !important;
  font-size: 1.8rem;
}}

/* -------- Tabs (Simple & Elegant) -------- */
.stTabs [data-baseweb="tab-list"] {{
  background-color: transparent;
  border-bottom: 1px solid #E2E8F0;
  gap: 24px;
}}

.stTabs [data-baseweb="tab"] {{
  background-color: transparent;
  color: #64748B;
  font-weight: 500;
  border: none;
  padding-bottom: 12px;
}}

.stTabs [aria-selected="true"] {{
  color: {PRIMARY_COLOR} !important;
  border-bottom: 2px solid {PRIMARY_COLOR};
}}

/* -------- Buttons (Subtle) -------- */
.stButton > button {{
  background-color: {PRIMARY_COLOR};
  color: white;
  border-radius: 8px;
  font-weight: 500;
  border: none;
  padding: 0.5rem 1.2rem;
  box-shadow: none;
}}

.stButton > button:hover {{
  background-color: #1D4ED8;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
}}

/* Sidebar tweaks to match */
section[data-testid="stSidebar"] {{
  background-color: #FFFFFF;
  border-right: 1px solid #F1F5F9;
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