import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and feature list
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")  # List of columns model expects

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("👩‍💼 Employee Attrition Predictor")
st.markdown("Predict whether an employee is at risk of attrition based on selected features.")

# --- User Inputs ---
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

# --- Build input DataFrame with ALL model features ---

input_dict = {
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
}

X_input = pd.DataFrame([input_dict])

# Add missing columns with zeros so columns match exactly
for col in selected_features:
    if col not in X_input.columns:
        X_input[col] = 0

# Drop any extra columns not expected by the model
X_input = X_input[selected_features]

# Scale numeric columns
numeric_cols = scaler.feature_names_in_
X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])

# --- Prediction ---

if st.button("🔮 Predict Attrition Risk"):
    try:
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **High Attrition Risk!**\n\nProbability: {probability:.2%}")
        else:
            st.success(f"✅ **Low Attrition Risk**\n\nProbability: {probability:.2%}")
    except Exception as e:
        st.error("❌ Error during prediction.")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Model: Logistic Regression | Features: 20 selected | Scaling: MinMaxScaler")
