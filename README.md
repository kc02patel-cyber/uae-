📊 BitesUAE – Executive Intelligence Dashboard

A production-grade analytics and predictive intelligence dashboard built using Python + Streamlit, designed to support CXO-level strategic and operational decision-making for a large-scale food delivery platform operating in the UAE.

This project goes beyond descriptive analytics to include forecasting, machine learning, behavioral intelligence, and executive recommendations.

🚀 Business Context

BitesUAE is a multi-city food delivery platform operating across Dubai, Abu Dhabi, Sharjah, and Ajman, serving thousands of customers via a distributed restaurant and rider network.

Leadership challenges addressed:

Revenue volatility and margin risk

Customer lifetime value uncertainty

Delivery delays and operational inefficiencies

Promotion effectiveness and portfolio imbalance

Predictive identification of high-value customers

This dashboard provides decision-grade clarity, not just visual reporting.

🎯 Key Objectives

Deliver executive-ready KPIs tied to business decisions

Enable root-cause analysis of operational delays

Forecast future demand and revenue trends

Identify and predict high-value customers

Provide clear strategic and operational recommendations

🧱 Dashboard Architecture
Global Design Principles

Consulting-grade UI/UX (executive friendly)

Interactive, drill-down visualizations only

Clear separation between strategic and operational insights

Session-safe filtering architecture

🧭 Executive Dashboard Sections
1️⃣ Strategic Performance Snapshot

Purpose: Monitor financial health and risk

Includes

GMV, AOV, Discount Burn

Revenue volatility trends

Growth signals and margin pressure

Decisions Enabled

Promo governance

Revenue stabilization strategy

2️⃣ Market & Geographic Intelligence

Purpose: Identify concentration risk and growth markets

Includes

City-level GMV contribution

Market performance comparison

Decisions Enabled

Market expansion

Rider & restaurant density planning

3️⃣ Customer Value & Behavioral Analytics

Purpose: Understand customer quality, not just volume

Includes

Frequency vs spend analysis

CLV proxy segmentation

Loyalty signal identification

Decisions Enabled

Retention strategy

VIP program investment

4️⃣ Product & Portfolio Intelligence

Purpose: Optimize cuisine and category mix

Includes

Revenue by cuisine

Portfolio concentration analysis

Decisions Enabled

Category-specific promotions

Menu diversification

5️⃣ Omnichannel & Operational Effectiveness

Purpose: Improve customer experience and cost efficiency

Includes

Delivery delay distribution

Operational bottleneck visibility

Decisions Enabled

Rider retraining

Last-mile optimization

6️⃣ Temporal Dynamics & Forecast Intelligence

Purpose: Anticipate demand, not react to it

Includes

Time-series trend analysis

14-day GMV forecast (Holt-Winters)

Decisions Enabled

Capacity planning

Inventory & staffing optimization

7️⃣ Predictive Intelligence & Model Evaluation

Purpose: Move from hindsight to foresight

Machine Learning Model

High-Value Customer Classification (Random Forest)

Metrics Displayed

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix (interactive)

Business Meaning

False Negatives → Lost retention opportunity (high cost)

False Positives → Minor promo leakage (low cost)

Model is decision-ready, not experimental.

🧠 Machine Learning Summary
Component	Description
Model	Random Forest Classifier
Target	High-Value Customer
Features	Order frequency, total spend
Use Case	Retention & loyalty prioritization
📈 Forecasting Methodology

Holt-Winters Exponential Smoothing

Trend detection

Short-term demand projection (14 days)

Actual vs predicted comparison

Business Value:
Reduces reactive decision-making and operational surprises.

🎛️ Filtering Logic (Advanced Streamlit State)
Global Filters

Date range

City

Section-Level Filters

Each section applies independent filters

Filter changes in one section do not impact others

Why this matters:
Mimics enterprise BI tools (Power BI / Tableau), not academic dashboards.
