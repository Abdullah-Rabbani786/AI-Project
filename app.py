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
# LIGHT THEME (EYE FRIENDLY)
# =====================================================
PRIMARY_COLOR = "#2563EB"
ACCENT_COLOR = "#38BDF8"
BG_COLOR = "#F8FAFC"
CARD_BG = "#FFFFFF"
TEXT_COLOR = "#0F172A"
SUBTEXT_COLOR = "#475569"

st.markdown(f"""
<style>
body {{
  background-color: {BG_COLOR};
}}

.block-container {{
  padding-top: 1.5rem;
}}

h1, h2, h3, h4, h5, h6 {{
  color: {TEXT_COLOR} !important;
  font-weight: 700;
}}

p, label, span, div {{
  color: {SUBTEXT_COLOR} !important;
}}

.header-card {{
  background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
  padding: 2.2rem;
  border-radius: 18px;
  margin-bottom: 2rem;
  color: white;
}}

section[data-testid="stSidebar"] > div {{
  background-color: #FFFFFF;
  border-right: 1px solid #E2E8F0;
}}

.stTabs [data-baseweb="tab-list"] {{
  background-color: #E5E7EB;
  border-radius: 12px;
  padding: 0.4rem;
}}

.stTabs [data-baseweb="tab"] {{
  background-color: transparent;
  color: {TEXT_COLOR};
  font-weight: 600;
  border-radius: 10px;
  padding: 0.6rem 1.4rem;
}}

.stTabs [aria-selected="true"] {{
  background-color: {PRIMARY_COLOR};
  color: white !important;
}}

[data-testid="stMetric"] {{
  background-color: {CARD_BG};
  border-radius: 14px;
  padding: 1.2rem;
  border: 1px solid #E2E8F0;
  box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}}

.stButton > button {{
  background-color: {PRIMARY_COLOR};
  color: white;
  border-radius: 10px;
  font-weight: 600;
  border: none;
  padding: 0.6rem 1.6rem;
}}

.stButton > button:hover {{
  background-color: #1D4ED8;
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
  <p><b>Talha Bashir</b> | Roll No: 2430-0162 | PAI Course Project</p>
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
    fig = px.pie(
        filtered_df["Company"].value_counts().head(8),
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
        preds = model.predict(X_test)

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"₹{mean_squared_error(y_test, preds, squared=False):,.0f}")
        c2.metric("MAE", f"₹{mean_absolute_error(y_test, preds):,.0f}")
        c3.metric("R²", f"{r2_score(y_test, preds):.4f}")

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
