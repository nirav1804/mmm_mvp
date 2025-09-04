# app.py
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Try to import meridien; if not available we'll fallback.
MERIDIEN_AVAILABLE = True
try:
    from meridien import MMM as MeridienMMM
except Exception:
    MERIDIEN_AVAILABLE = False

from sklearn.linear_model import LinearRegression

# -------------------------
# Fallback SimpleMMM (deterministic linear fallback)
# -------------------------
class SimpleMMM:
    def __init__(self, date_var, dep_var, indep_vars, freq="W"):
        self.date_var = date_var
        self.dep_var = dep_var
        self.indep_vars = indep_vars
        self.freq = freq
        self.model = LinearRegression()
        self.coef_ = None
        self.intercept_ = 0.0
        self.mean_spend_ = None

    def fit(self, df: pd.DataFrame):
        X = df[self.indep_vars].fillna(0.0).values
        y = df[self.dep_var].fillna(0.0).values
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = float(self.model.intercept_)
        self.mean_spend_ = df[self.indep_vars].mean().values
        return self

    def get_roi(self) -> pd.DataFrame:
        # ROI approximated as coefficient (revenue per unit spend)
        roi = pd.DataFrame({"variable": self.indep_vars, "roi": self.coef_.astype(float)})
        return roi.sort_values("roi", ascending=False).reset_index(drop=True)

    def get_decomp(self) -> pd.DataFrame:
        channel_effect = self.coef_ * self.mean_spend_
        channel_effect = np.clip(channel_effect, a_min=0, a_max=None)
        total = channel_effect.sum()
        if total <= 0:
            contrib = np.zeros_like(channel_effect)
        else:
            contrib = channel_effect / total * 100.0
        return pd.DataFrame({"variable": self.indep_vars, "contribution": contrib}).sort_values("contribution", ascending=False).reset_index(drop=True)

    def optimize_budget(self, budget: float) -> pd.DataFrame:
        roi_df = self.get_roi()
        roi_pos = roi_df[roi_df["roi"] > 0].copy()
        if roi_pos.empty or budget <= 0:
            return pd.DataFrame({"variable": self.indep_vars, "allocation": [0.0]*len(self.indep_vars)})
        roi_sum = roi_pos["roi"].sum()
        roi_pos["allocation"] = budget * roi_pos["roi"] / roi_sum
        out = pd.DataFrame({"variable": self.indep_vars}).merge(roi_pos[["variable", "allocation"]], on="variable", how="left").fillna(0.0)
        exp_rev = self.intercept_ + (out.set_index("variable")["allocation"].reindex(self.indep_vars).values * self.coef_).sum()
        out["expected_revenue"] = exp_rev
        return out.sort_values("allocation", ascending=False).reset_index(drop=True)

    def optimize_for_target(self, target: float) -> pd.DataFrame:
        roi_df = self.get_roi()
        roi_pos = roi_df[roi_df["roi"] > 0].copy()
        sum_roi = roi_pos["roi"].sum()
        if sum_roi <= 0:
            raise ValueError("Cannot reach target with non-positive ROIs.")
        baseline = self.intercept_
        gap = max(0.0, target - baseline)
        if gap == 0:
            out = pd.DataFrame({"variable": self.indep_vars, "allocation": [0.0]*len(self.indep_vars)})
            out["required_budget"] = 0.0
            out["expected_revenue"] = target
            return out
        # Conservative budget approx using sum of squares
        sum_roi_sq = (roi_pos["roi"] ** 2).sum()
        if sum_roi_sq > 0:
            budget = gap * (sum_roi / sum_roi_sq)
        else:
            avg_roi = sum_roi / len(roi_pos)
            budget = gap / max(avg_roi, 1e-9)
        roi_pos["allocation"] = budget * roi_pos["roi"] / sum_roi
        out = pd.DataFrame({"variable": self.indep_vars}).merge(roi_pos[["variable", "allocation"]], on="variable", how="left").fillna(0.0)
        out["required_budget"] = out["allocation"].sum()
        out["expected_revenue"] = baseline + (out.set_index("variable")["allocation"].reindex(self.indep_vars).values * self.coef_).sum()
        return out.sort_values("allocation", ascending=False).reset_index(drop=True)

# -------------------------
# Helpers
# -------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def multi_sheet_excel_bytes(**named_dfs) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in named_dfs.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.read()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="MMM MVP (Meridien preferred)", layout="wide")
st.title("📊 MMM MVP — Meridien (Bayesian) preferred")

st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload Excel/CSV (weekly data)", type=["xlsx", "csv"])
brand = st.sidebar.text_input("Brand (optional)", "My Brand")
st.sidebar.markdown("**Expect columns:** `date_start` (date), `revenue` (target), `*_spend` (media channels)")

if not uploaded:
    st.info("Upload your dataset (Book1.xlsx) with `date_start`, `revenue`, and `*_spend` columns.")
    st.stop()

# Read file
if uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# Basic validation
if "date_start" not in df.columns or "revenue" not in df.columns:
    st.error("Dataset must contain `date_start` and `revenue` columns.")
    st.stop()

# parse dates and find media cols
df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
df = df.dropna(subset=["date_start"]).copy()
media_cols = [c for c in df.columns if c.endswith("_spend")]
if not media_cols:
    st.error("No media spend columns found. Ensure channel columns end with `_spend`.")
    st.stop()

# date filter
min_d, max_d = df["date_start"].min(), df["date_start"].max()
date_range = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
mask = (df["date_start"] >= pd.to_datetime(date_range[0])) & (df["date_start"] <= pd.to_datetime(date_range[1]))
df_f = df.loc[mask].reset_index(drop=True)

st.sidebar.write(f"Rows used: {len(df_f)} | Channels: {len(media_cols)}")
with st.expander("Preview data (first rows)"):
    st.dataframe(df_f[["date_start", "revenue", *media_cols]].head(20))

# Train model (Meridien preferred)
st.header("Run MMM Model")
use_meridien = MERIDIEN_AVAILABLE and st.sidebar.checkbox("Prefer Meridien (Bayesian)", value=MERIDIEN_AVAILABLE)

meridien_ok = False
model = None
contribs = None
roi = None

if use_meridien:
    st.caption("Attempting to run Meridien (Bayesian MMM).")
    try:
        model = MeridienMMM(date_var="date_start", dep_var="revenue", indep_vars=media_cols, freq="W")
        model.fit(df_f)                # fit Bayesian model
        # Meridien returns decomposition & roi via these methods (if available)
        contribs = model.get_decomp()
        roi = model.get_roi()
        meridien_ok = True
        st.success("Meridien model fitted successfully (Bayesian).")
    except Exception as e:
        st.warning(f"Meridien run failed or not fully compatible in this environment: {e}")
        st.info("Switching to fallback SimpleMMM.")
        meridien_ok = False

if not meridien_ok:
    st.caption("Using fallback linear MMM (deterministic).")
    model = SimpleMMM(date_var="date_start", dep_var="revenue", indep_vars=media_cols, freq="W")
    model.fit(df_f)
    contribs = model.get_decomp()
    roi = model.get_roi()
    st.success("Fallback MMM fitted.")

# -------------------------
# Case A: Historical analysis
# -------------------------
st.header("Case A — Historical Analysis (Contribution & ROI)")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Contribution (%)")
    st.dataframe(contribs)
    fig_pie = px.pie(contribs, names="variable", values="contribution", title="Media contribution (%)")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.download_button("⬇ Download Contribution CSV", df_to_csv_bytes(contribs), file_name="contribution.csv")

with c2:
    st.subheader("ROI (revenue per ₹ spend)")
    st.dataframe(roi)
    fig_roi = px.bar(roi, x="variable", y="roi", title="Channel ROI")
    st.plotly_chart(fig_roi, use_container_width=True)
    st.download_button("⬇ Download ROI CSV", df_to_csv_bytes(roi), file_name="roi.csv")

# -------------------------
# Case B: Budget optimization
# -------------------------
st.header("Case B — Budget Optimization (Given Budget → Allocation)")
future_budget = st.number_input("Enter future budget (₹)", min_value=0.0, step=100000.0, value=10_000_000.0)

if st.button("Optimize Budget"):
    try:
        if meridien_ok and hasattr(model, "optimize_budget"):
            opt_plan = model.optimize_budget(budget=future_budget)
        else:
            opt_plan = model.optimize_budget(budget=future_budget)
    except Exception as e:
        st.warning(f"Model-based optimizer failed: {e}\nUsing fallback proportional ROI allocation.")
        opt_plan = SimpleMMM(date_var="date_start", dep_var="revenue", indep_vars=media_cols).fit(df_f).optimize_budget(future_budget)

    st.subheader("Optimized Allocation")
    st.dataframe(opt_plan)
    st.plotly_chart(px.bar(opt_plan, x="variable", y="allocation", title=f"Optimized allocation for ₹{int(future_budget):,}"), use_container_width=True)
    st.download_button("⬇ Download Optimized plan (CSV)", df_to_csv_bytes(opt_plan), file_name="optimized_plan.csv")

# -------------------------
# Case C: Target-driven planning
# -------------------------
st.header("Case C — Target-driven Planning (Target revenue → Required budget & allocation)")
target_rev = st.number_input("Enter target revenue (₹)", min_value=0.0, step=100000.0, value=float(df_f["revenue"].mean() * len(df_f)))

if st.button("Plan for Target"):
    try:
        if meridien_ok and hasattr(model, "optimize_for_target"):
            target_plan = model.optimize_for_target(target=target_rev)
        else:
            target_plan = model.optimize_for_target(target=target_rev)
    except Exception as e:
        st.warning(f"Model-based target planner failed: {e}\nUsing fallback target planner.")
        target_plan = SimpleMMM(date_var="date_start", dep_var="revenue", indep_vars=media_cols).fit(df_f).optimize_for_target(target_rev)

    st.subheader("Allocation to achieve target")
    st.dataframe(target_plan)
    st.plotly_chart(px.bar(target_plan, x="variable", y="allocation", title=f"Allocation to achieve ₹{int(target_rev):,}"), use_container_width=True)
    st.download_button("⬇ Download Target plan (CSV)", df_to_csv_bytes(target_plan), file_name="target_plan.csv")

# -------------------------
# Full report export
# -------------------------
if st.button("Export Full Excel Report (A+B+C)"):
    out_bytes = multi_sheet_excel_bytes(
        Meta=pd.DataFrame({
            "Brand":[brand],
            "Used_Meridien":[bool(meridien_ok)],
            "Rows_used":[len(df_f)],
            "From":[df_f["date_start"].min()],
            "To":[df_f["date_start"].max()]
        }),
        Contribution=contribs,
        ROI=roi,
        OptimizedPlan=(opt_plan if 'opt_plan' in locals() else pd.DataFrame()),
        TargetPlan=(target_plan if 'target_plan' in locals() else pd.DataFrame())
    )
    st.download_button("📘 Download Full Report (Excel)", data=out_bytes, file_name="mmm_full_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
