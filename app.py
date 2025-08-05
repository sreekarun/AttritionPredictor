import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load model
try:
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded!")
except:
    st.error("Model file not found.")
    model = None

# Load scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    st.success("Scaler loaded!")
except:
    scaler = None
    st.warning("Scaler not found.")

# Try to get expected scaler feature names
if scaler and hasattr(scaler, "feature_names_in_"):
    scaler_features = scaler.feature_names_in_.tolist()
else:
    scaler_features = []

# Define all model input features (from training)
model_features = [
    'Age', 'DistanceFromHome', 'EmployeeCount', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager',
    'Gender_Male', 'OverTime_Yes', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales', 'EducationField_Life Sciences',
    'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single'
]

# --- Streamlit UI ---
st.title("Employee Attrition Prediction App")
st.subheader("Fill in employee details to predict the attrition risk.")

# Collect simplified inputs
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
age = st.number_input("Age", min_value=18, max_value=60, value=30)
overtime = st.selectbox("OverTime", ["No", "Yes"])
total_working_years = st.number_input("Total Working Years", 0, 40, 5)
distance_from_home = st.number_input("Distance From Home (km)", 0, 30, 5)
years_at_company = st.number_input("Years at Company", 0, 40, 3)
satisfaction_score = st.slider("Satisfaction Score (0-10)", 0, 10, 7)
remote_stress_score = st.slider("Remote Stress Score (0-10)", 0, 10, 5)

# Map these inputs into model's expected features
input_data = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
    "TotalWorkingYears": total_working_years,
    "DistanceFromHome": distance_from_home,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": satisfaction_score,
    "EnvironmentSatisfaction": remote_stress_score,
    "EmployeeCount": 1,
    "StandardHours": 80
}

# Fill in missing features as 0 (dummies not shown to user)
for feature in model_features:
    if feature not in input_data:
        input_data[feature] = 0

# Create DataFrame
input_df = pd.DataFrame([input_data])

# Scale numerical features if scaler is available
if scaler:
    to_scale = [col for col in scaler_features if col in input_df.columns]
    try:
        input_df[to_scale] = scaler.transform(input_df[to_scale])
    except Exception as e:
        st.error("Scaling failed. Please check your scaler compatibility.")
        st.stop()

# Predict
if model and st.button("Predict Attrition"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Results:")
        if prediction == 1:
            st.error("Prediction: Employee is likely to Attrit")
        else:
            st.success("Prediction: Employee is likely to Stay")
        st.info(f"Attrition Probability: {probability:.2%}")
    except Exception as e:
        st.error("Prediction failed. Check input compatibility with model.")
else:
    st.info("Click the button to predict.")
