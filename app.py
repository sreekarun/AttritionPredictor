import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and features used in training
model = joblib.load("hr_attrition_model_selected_features_.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("selected_features.pkl")  # Exactly 20 features used in training

# Page config and title
st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("👩‍💼 Employee Attrition Predictor")
st.markdown("Predict whether an employee is at risk of attrition based on selected features.")

# ---- UI Inputs ----
st.subheader("📋 Employee Details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    years_with_curr_manager = st.slider("Years with Current Manager", 0, 20, 3)
    years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, step=500)
    job_level = st.slider("Job Level", 1, 5, 2)
    stock_option_level = st.slider("Stock Option Level", 0, 3, 1)

with col2:
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    gender = st.selectbox("Gender", ["Female", "Male"])
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    job_role = st.selectbox("Job Role", [
        "Laboratory Technician", "Sales Representative", "Research Director", "Manager",
        "Healthcare Representative", "Human Resources", "Manufacturing Director",
        "Research Scientist", "Sales Executive"
    ])
    department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])

# ---- Build ONLY the 20 features used in training ----
X_input = pd.DataFrame([{
    'OverTime_Yes': 1 if overtime == "Yes" else 0,
    'TotalWorkingYears': total_working_years,
    'MaritalStatus_Single': 1 if marital_status == "Single" else 0,
    'YearsWithCurrManager': years_with_curr_manager,
    'Age': age,
    'YearsInCurrentRole': years_in_current_role,
    'YearsAtCompany': years_at_company,
    'JobLevel': job_level,
    'MonthlyIncome': monthly_income,
    'StockOptionLevel': stock_option_level,
    'JobRole_Laboratory Technician': 1 if job_role == "Laboratory Technician" else 0,
    'JobRole_Sales Representative': 1 if job_role == "Sales Representative" else 0,
    'BusinessTravel_Travel_Frequently': 1 if business_travel == "Travel_Frequently" else 0,
    'Gender_Male': 1 if gender == "Male" else 0,
    'JobRole_Research Director': 1 if job_role == "Research Director" else 0,
    'JobInvolvement': job_involvement,
    'JobSatisfaction': job_satisfaction,
    'EnvironmentSatisfaction': environment_satisfaction,
    'JobRole_Manager': 1 if job_role == "Manager" else 0,
    'Department_Research & Development': 1 if department == "Research & Development" else 0,
}])

# 🔒 Ensure all features exist
for col in feature_columns:
    if col not in X_input.columns:
        X_input[col] = 0

# 🧼 Reorder to match model
X_input = X_input[feature_columns]

# ✅ Safe scaling
try:
    cols_to_scale = scaler.feature_names_in_
    X_scaled = scaler.transform(X_input[cols_to_scale])
    X_input[cols_to_scale] = X_scaled
except Exception as e:
    st.error("⚠️ Scaling error.")
    st.exception(e)

# ---- Prediction ----
st.markdown("---")
if st.button("🔮 Predict Attrition Risk"):
    try:
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **High Attrition Risk!**\n\nProbability: `{probability:.2%}`")
        else:
            st.success(f"✅ **Low Attrition Risk**\n\nProbability: `{probability:.2%}`")
    except Exception as e:
        st.error("❌ Prediction error.")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Model: Logistic Regression | Features: 20 selected | Scaling: MinMaxScaler")
