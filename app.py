import streamlit as st
import pandas as pd

st.title("📊 Marketing Mix Modeling (MMM) MVP")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Case A
    st.subheader("Case A: Past Performance Analysis")
    if st.button("Run Case A"):
        st.success("✅ Placeholder: Analyze past spends & revenue trends")

    # Case B
    st.subheader("Case B: Budget Optimization for Future")
    budget = st.number_input("Enter upcoming budget", min_value=0)
    if st.button("Run Case B"):
        st.success(f"✅ Placeholder: Optimize allocation for budget {budget}")

    # Case C
    st.subheader("Case C: Media Plan from Sales Target")
    target = st.number_input("Enter sales target", min_value=0)
    if st.button("Run Case C"):
        st.success(f"✅ Placeholder: Recommend media mix to reach sales target {target}")
