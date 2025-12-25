import numpy as np
import pandas as pd
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Laptop Price Analytics",
    layout="wide",
    page_icon="💻"
)

# =====================================================
# STYLING
# =====================================================
PRIMARY_COLOR = "#00D9FF"
ACCENT_COLOR = "#4ECDC4"

st.markdown(f"""
<style>
body {{
  background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}}
h1, h2, h3, h4, h5, h6, p, label, span, div {{
  color: #EAEAEA !important;
}}
.stButton>button {{
  background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
  color: white;
  font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown(f"""
<div style="background:linear-gradient(135deg,{PRIMARY_COLOR},{ACCENT_COLOR});
padding:2rem;border-radius:20px">
<h1>💻 Laptop Price Analytics Dashboard</h1>
<p>Market Analysis & ML Price Prediction</p>
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

    df["Price"] = pd.to_numeric(df.get("Price"), errors="coerce")

    if "Ram" in df.columns:
        df["Ram_GB"] = df["Ram"].str.extract(r"(\d+)").astype(float)

    if "Inches" in df.columns:
        df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")

    if "Weight" in df.columns:
        df["Weight_kg"] = df["Weight"].str.extract(r"(\d+\.?\d*)").astype(float)

    for col in ["Ram_GB", "Inches", "Weight_kg"]:
        if col not in df.columns:
            df[col] = np.nan

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

companies = sorted(df["Company"].dropna().astype(str).unique())
types = sorted(df["TypeName"].dropna().astype(str).unique())

selected_companies = st.sidebar.multiselect(
    "Company", companies, companies[:5] if len(companies) >= 5 else companies
)
selected_types = st.sidebar.multiselect("Laptop Type", types, types)

price_min = float(df["Price"].min())
price_max = float(df["Price"].max())

price_range = st.sidebar.slider(
    "Price Range (₹)",
    price_min,
    price_max,
    (price_min, price_max)
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
    "🏠 Overview",
    "📊 Market",
    "💻 Specs",
    "📈 Trends",
    "🤖 ML Model",
    "💾 Export"
])

# =====================================================
# TAB 1 — OVERVIEW
# =====================================================
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Price", f"₹{filtered_df['Price'].mean():,.0f}")
    col2.metric("Total Laptops", len(filtered_df))
    col3.metric("Avg RAM", f"{filtered_df['Ram_GB'].mean():.1f} GB")
    col4.metric("Companies", filtered_df["Company"].nunique())

    fig = px.bar(
        filtered_df.groupby("Company")["Price"].mean()
        .sort_values(ascending=False)
        .head(10),
        orientation="h",
        title="Top 10 Companies by Average Price"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2 — MARKET
# =====================================================
with tabs[1]:
    fig = px.pie(
        filtered_df["Company"].value_counts().head(8),
        hole=0.4,
        title="Market Share by Company"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 3 — SPECS (FIXED SCATTER ERROR HERE)
# =====================================================
with tabs[2]:
    fig = px.box(
        filtered_df,
        x="Ram",
        y="Price",
        title="RAM vs Price"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 🔴 FIX: Drop NaNs before scatter with size
    scatter_df = filtered_df.dropna(
        subset=["Inches", "Price", "Weight_kg", "Company"]
    )

    if scatter_df.empty:
        st.info("Not enough data to display Screen Size vs Price.")
    else:
        fig = px.scatter(
            scatter_df,
            x="Inches",
            y="Price",
            color="Company",
            size="Weight_kg",
            size_max=25,
            hover_data=["Company", "TypeName"],
            title="Screen Size vs Price (Bubble = Weight)"
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 4 — TRENDS
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
# TAB 5 — MACHINE LEARNING
# =====================================================
with tabs[4]:
    st.subheader("🤖 Random Forest Price Prediction")

    n_estimators = st.slider("Trees", 50, 300, 150, 25)
    max_depth = st.slider("Max Depth", 5, 40, 20, 5)
    test_size = st.slider("Test Size %", 10, 40, 20, 5) / 100

    if st.button("🚀 Train Model"):
        model_df = filtered_df.dropna(subset=[
            "Price", "Ram_GB", "Inches", "Weight_kg",
            "Company", "TypeName", "OpSys"
        ])

        le_c = LabelEncoder()
        le_t = LabelEncoder()
        le_o = LabelEncoder()

        model_df["Company_enc"] = le_c.fit_transform(model_df["Company"])
        model_df["Type_enc"] = le_t.fit_transform(model_df["TypeName"])
        model_df["OS_enc"] = le_o.fit_transform(model_df["OpSys"])

        X = model_df[
            ["Ram_GB", "Inches", "Weight_kg",
             "Company_enc", "Type_enc", "OS_enc"]
        ]
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

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"₹{mean_squared_error(y_test, preds, squared=False):,.0f}")
        col2.metric("MAE", f"₹{mean_absolute_error(y_test, preds):,.0f}")
        col3.metric("R²", f"{r2_score(y_test, preds):.4f}")

        fig = px.scatter(
            x=y_test,
            y=preds,
            labels={"x": "Actual Price", "y": "Predicted Price"},
            title="Actual vs Predicted Prices"
        )
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=y_test.max(), y1=y_test.max()
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 6 — EXPORT
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
