import streamlit as st
import pandas as pd
import joblib

# Load model components
model = joblib.load("hr_attrition_model_selected_features.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")  # Must match trained model input

# App configuration
st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("👩‍💼 Employee Attrition Predictor")
st.markdown("Use this tool to estimate whether an employee is at risk of leaving the company based on their profile.")

st.markdown("---")
st.subheader("📋 Employee Profile")

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    years_with_curr_manager = st.slider("Years with Current Manager", 0, 20, 3)
    job_level = st.slider("Job Level", 1, 5, 2)
    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)

with col2:
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)
    stock_option_level = st.slider("Stock Option Level", 0, 3, 1)

    gender = st.selectbox("Gender", ["Female", "Male"])
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    job_role = st.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", 
        "Manufacturing Director", "Research Director", "Research Scientist", 
        "Sales Executive", "Sales Representative"
    ])
    department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])

# --- Feature Vector Construction ---
input_data = {
    'Age': age,
    'TotalWorkingYears': total_working_years,
    'YearsWithCurrManager': years_with_curr_manager,
    'YearsInCurrentRole': years_in_current_role,
    'YearsAtCompany': years_at_company,
    'JobLevel': job_level,
    'MonthlyIncome': monthly_income,
    'StockOptionLevel': stock_option_level,
    'JobInvolvement': job_involvement,
    'JobSatisfaction': job_satisfaction,
    'EnvironmentSatisfaction': environment_satisfaction,
    'Gender_Male': 1 if gender == "Male" else 0,
    'OverTime_Yes': 1 if overtime == "Yes" else 0,
    'MaritalStatus_Single': 1 if marital_status == "Single" else 0,
    'BusinessTravel_Travel_Frequently': 1 if business_travel == "Travel_Frequently" else 0,
    'JobRole_Laboratory Technician': 1 if job_role == "Laboratory Technician" else 0,
    'JobRole_Sales Representative': 1 if job_role == "Sales Representative" else 0,
    'JobRole_Research Director': 1 if job_role == "Research Director" else 0,
    'JobRole_Manager': 1 if job_role == "Manager" else 0,
    'Department_Research & Development': 1 if department == "Research & Development" else 0,
}

# Fill in any missing features with 0 (for robustness)
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

# Convert to DataFrame
X_input = pd.DataFrame([input_data])

# ✅ Fixed scaling section (avoids ValueError)
try:
    # Scale only columns the scaler expects
    columns_to_scale = scaler.feature_names_in_
    X_scaled_part = pd.DataFrame(scaler.transform(X_input[columns_to_scale]), columns=columns_to_scale)

    # Replace scaled values into X_input
    for col in columns_to_scale:
        X_input[col] = X_scaled_part[col]

    # Ensure order matches model training
    X_input = X_input[feature_columns]
except Exception as e:
    st.error("❌ Error during scaling. Please verify scaler and features.")
    st.exception(e)

# --- Prediction ---
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
        st.error("An error occurred while making the prediction.")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Model powered by Logistic Regression using selected features and SMOTE.")
