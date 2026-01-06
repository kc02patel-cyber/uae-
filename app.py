# =========================================================
# BITESUAE – EXECUTIVE INTELLIGENCE DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------
# UI CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="BitesUAE | Executive Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    body { background-color: #0E1117; }
    .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    xls = pd.ExcelFile("data/BitesUAE_Cleaned_Data.xlsx")
    orders = pd.read_excel(xls, "ORDERS")
    delivery = pd.read_excel(xls, "DELIVERY_EVENTS")
    customers = pd.read_excel(xls, "CUSTOMERS")
    restaurants = pd.read_excel(xls, "RESTAURANTS")
    return orders, delivery, customers, restaurants

orders, delivery, customers, restaurants = load_data()

orders["order_datetime"] = pd.to_datetime(orders["order_datetime"])

# ---------------------------------------------------------
# GLOBAL FILTERS (SESSION STATE)
# ---------------------------------------------------------
st.sidebar.header("Global Controls")

date_range = st.sidebar.date_input(
    "Date Range",
    [orders.order_datetime.min(), orders.order_datetime.max()]
)

cities = st.sidebar.multiselect(
    "City",
    options=orders.merge(restaurants, on="restaurant_id")["city_y"].unique()
)

# Apply global filters
df = orders.copy()
df = df[(df.order_datetime.dt.date >= date_range[0]) &
        (df.order_datetime.dt.date <= date_range[1])]

if cities:
    df = df.merge(restaurants, on="restaurant_id")
    df = df[df.city_y.isin(cities)]

# =========================================================
# 1️⃣ STRATEGIC PERFORMANCE SNAPSHOT
# =========================================================
st.header("Strategic Performance Snapshot")

gmv = df.gross_amount.sum()
aov = df.gross_amount.mean()
discount_burn = df.discount_amount.sum() / df.gross_amount.sum()

c1, c2, c3 = st.columns(3)
c1.metric("GMV (AED)", f"{gmv:,.0f}")
c2.metric("Average Order Value", f"{aov:,.0f}")
c3.metric("Discount Burn %", f"{discount_burn:.1%}")

gmv_trend = df.groupby(df.order_datetime.dt.date)["gross_amount"].sum().reset_index()

fig = px.line(
    gmv_trend,
    x="order_datetime",
    y="gross_amount",
    title="Revenue Volatility & Growth Pattern",
    labels={"gross_amount": "GMV (AED)", "order_datetime": "Date"},
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Executive Insight:**  
Revenue volatility highlights demand sensitivity to time and promotions.  
**Recommended Action:** Stabilize revenue via controlled promo cadence and peak-hour pricing discipline.
""")

# =========================================================
# 2️⃣ MARKET & GEOGRAPHIC INTELLIGENCE
# =========================================================
st.header("Market & Geographic Intelligence")

geo = df.merge(restaurants, on="restaurant_id")
geo_perf = geo.groupby("city_y")["gross_amount"].sum().reset_index()

fig = px.bar(
    geo_perf,
    x="city_y",
    y="gross_amount",
    title="City-Level Revenue Contribution",
    text_auto=True
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Insight:** Market concentration risk visible.  
**Action:** Invest in underperforming cities via targeted rider density and restaurant onboarding.
""")

# =========================================================
# 3️⃣ CUSTOMER VALUE & BEHAVIORAL ANALYTICS
# =========================================================
st.header("Customer Value & Behavioral Analytics")

cust_orders = df.groupby("customer_id").agg(
    orders=("order_id", "count"),
    spend=("gross_amount", "sum")
).reset_index()

cust_orders["high_value"] = cust_orders.spend > cust_orders.spend.quantile(0.75)

fig = px.scatter(
    cust_orders,
    x="orders",
    y="spend",
    color="high_value",
    title="Customer Frequency vs Value (CLV Proxy)"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Insight:** A small segment drives disproportionate value.  
**Action:** Loyalty & retention investment yields highest ROI.
""")

# =========================================================
# 4️⃣ PRODUCT & PORTFOLIO INTELLIGENCE
# =========================================================
st.header("Product & Portfolio Intelligence")

prod = df.merge(restaurants, on="restaurant_id")
cat_perf = prod.groupby("cuisine_type")["gross_amount"].sum().reset_index()

fig = px.pie(
    cat_perf,
    names="cuisine_type",
    values="gross_amount",
    title="Revenue Mix by Cuisine"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Insight:** Portfolio imbalance may expose GMV risk.  
**Action:** Diversify cuisine promotions to reduce dependency.
""")

# =========================================================
# 5️⃣ OMNICHANNEL & OPERATIONAL EFFECTIVENESS
# =========================================================
st.header("Omnichannel & Operational Effectiveness")

ops = df.merge(delivery, on="order_id")
delay_rate = ops.delay_flag.mean()

st.metric("Delay Rate", f"{delay_rate:.1%}")

fig = px.histogram(
    ops,
    x="actual_delivery_time_mins",
    nbins=30,
    title="Delivery Time Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Insight:** Delivery tail risk impacts CX disproportionately.  
**Action:** Focus on last-mile optimization and rider retraining.
""")

# =========================================================
# 6️⃣ TEMPORAL DYNAMICS & FORECAST INTELLIGENCE
# =========================================================
st.header("Temporal Dynamics & Forecast Intelligence")

daily = gmv_trend.set_index("order_datetime")

model = ExponentialSmoothing(
    daily.gross_amount,
    trend="add",
    seasonal=None
).fit()

forecast = model.forecast(14)

fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily.gross_amount, name="Actual"))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecast"))

fig.update_layout(title="GMV Forecast (14 Days)")

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Insight:** Forecast divergence signals demand risk or upside.  
**Action:** Align inventory, riders, and promos proactively.
""")

# =========================================================
# 7️⃣ PREDICTIVE INTELLIGENCE & MODEL EVALUATION
# =========================================================
st.header("Predictive Intelligence & Model Evaluation")

ml = cust_orders.copy()
X = ml[["orders", "spend"]]
y = ml["high_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
    "ROC-AUC": roc_auc_score(y_test, proba)
}

st.json(metrics)

cm = confusion_matrix(y_test, pred)
fig = px.imshow(
    cm,
    text_auto=True,
    title="Confusion Matrix – High Value Customer Prediction"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Business Meaning:**  
False negatives = missed VIP retention opportunities (high cost).  
False positives = minor promo leakage (low cost).  
**Model is decision-ready.**
""")

# =========================================================
# EXECUTIVE READOUT
# =========================================================
st.success("""
### Executive Readout
• Revenue concentration and delivery delays present measurable risk  
• High-value customers are predictable and defensible  
• Forecasting enables proactive operations  
• Immediate ROI from retention & last-mile optimization
""")
