import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# --- 1. Dashboard Layout & Config ---
st.set_page_config(
    page_title="Employee Attrition Risk Portal",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Employee Attrition & Salary Insights Portal")
st.markdown(
    "Evaluate attrition drivers and predict real-time resignation risk for individual employees."
)


# --- 2. Load & Cache Data ---
@st.cache_data
def load_data():
    return pd.read_csv("employee_attrition.csv")


df = load_data()


# --- 3. Sidebar Filters ---
st.sidebar.header("Filter Department Data")
dept_filter = st.sidebar.multiselect(
    "Select Department:",
    options=df["Department"].unique(),
    default=df["Department"].unique(),
)

filtered_df = df[df["Department"].isin(dept_filter)]

# --- 4. High-Level Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", len(filtered_df))

attrition_count = (filtered_df["Attrition"] == "Yes").sum()
attrition_rate = (attrition_count / len(filtered_df)) * 100
col2.metric("Attrition Rate", f"{attrition_rate:.1f}%")

avg_salary = filtered_df["MonthlySalary"].mean()
col3.metric("Avg Monthly Salary", f"${avg_salary:,.2f}")

avg_overtime = filtered_df["OvertimeHours"].mean()
col4.metric("Avg Monthly Overtime", f"{avg_overtime:.1f} hrs")

st.divider()

# --- 5. Interactive Cross-Tabulation & Charting ---
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Cross-Tabulation: Attrition by Job Satisfaction")
    ct = pd.crosstab(
        filtered_df["JobSatisfaction"],
        filtered_df["Attrition"],
        normalize="index",
    )
    st.dataframe(ct.style.format("{:.1%}"), use_container_width=True)

with right_col:
    st.subheader("Overtime vs. Attrition Risk")
    fig = px.box(
        filtered_df,
        x="Attrition",
        y="OvertimeHours",
        color="Attrition",
        title="Distribution of Overtime Hours by Resignation Status",
        color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# --- 6. Live Attrition Risk Predictor ---
st.subheader("🤖 Predict Individual Employee Attrition Risk")


# Train baseline model behind the scenes
@st.cache_resource
def train_model(data):
    encoded = pd.get_dummies(
        data[["Department", "JobSatisfaction", "Attrition"]],
        drop_first=True,
        dtype=int,
    )
    X_full = pd.concat(
        [data[["YearsAtCompany", "MonthlySalary", "OvertimeHours"]], encoded],
        axis=1,
    )
    X = X_full.drop(columns=["Attrition_Yes"])
    y = X_full["Attrition_Yes"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf, X.columns


model, feature_names = train_model(df)

# Prediction Inputs
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    in_years = st.number_input("Years at Company", 1, 30, 3)
with p_col2:
    in_salary = st.number_input("Monthly Salary ($)", 2000, 20000, 5000)
with p_col3:
    in_overtime = st.number_input("Overtime Hours/Month", 0, 60, 25)
with p_col4:
    in_satisfaction = st.selectbox(
        "Job Satisfaction", ["Low", "Medium", "High", "Very High"]
    )

if st.button("Predict Flight Risk"):
    # Build a 1-row DataFrame matching trained features
    input_dict = {
        "YearsAtCompany": in_years,
        "MonthlySalary": in_salary,
        "OvertimeHours": in_overtime,
        "Department_R&D": 0,
        "Department_Sales": 1,
        "JobSatisfaction_Low": 1 if in_satisfaction == "Low" else 0,
        "JobSatisfaction_Medium": 1 if in_satisfaction == "Medium" else 0,
        "JobSatisfaction_Very High": 1 if in_satisfaction == "Very High" else 0,
    }

    input_df = pd.DataFrame([input_dict])[feature_names]
    risk_prob = model.predict_proba(input_df)[0][1] * 100

    if risk_prob > 50:
        st.error(
            f"🚨 **High Flight Risk Detected!** Estimated Attrition Probability: **{risk_prob:.1f}%**"
        )
    else:
        st.success(
            f"✅ **Low Flight Risk.** Estimated Attrition Probability: **{risk_prob:.1f}%**"
        )