import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📊 Marketing Mix Modeling (MMM) MVP")

# Sidebar uploader
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Identify spend columns automatically
    spend_cols = [c for c in df.columns if c.endswith("_spend")]
    if "revenue" not in df.columns:
        st.error("Your dataset must contain a 'revenue' column.")
    else:
        # ---- Case A ----
        st.subheader("Case A: Past Performance Analysis")
        if st.button("Run Case A"):
            channel_totals = df[spend_cols].sum()
            revenue_total = df["revenue"].sum()
            roi = revenue_total / channel_totals

            st.write("### ROI by Channel")
            st.table(roi.to_frame("ROI"))

            st.write("### Spend vs Revenue Trend")
            fig = px.line(df, x="date_start", y=spend_cols + ["revenue"], title="Spends & Revenue Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # ---- Case B ----
        st.subheader("Case B: Budget Optimization for Future")
        budget = st.number_input("Enter upcoming budget", min_value=0, step=1000)
        if st.button("Run Case B"):
            X = df[spend_cols]
            y = df["revenue"]
            model = LinearRegression().fit(X, y)
            coefs = pd.Series(model.coef_, index=spend_cols)

            # Normalize coefficients into weights
            weights = coefs / coefs.sum()
            optimized_plan = weights * budget

            st.write("### Optimized Budget Allocation")
            st.table(optimized_plan.to_frame("Optimized Spend"))

        # ---- Case C ----
        st.subheader("Case C: Media Plan from Sales Target")
        target = st.number_input("Enter sales target", min_value=0, step=1000)
        if st.button("Run Case C"):
            X = df[spend_cols]
            y = df["revenue"]
            model = LinearRegression().fit(X, y)

            coefs = pd.Series(model.coef_, index=spend_cols)
            base_pred = model.intercept_
            shortfall = target - base_pred

            if shortfall <= 0:
                st.warning("Target is too low compared to baseline model. Increase target.")
            else:
                weights = coefs / coefs.sum()
                recommended_spends = weights * shortfall
                st.write("### Recommended Media Plan")
                st.table(recommended_spends.to_frame("Recommended Spend"))
