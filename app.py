import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'logistic_regression_model.pkl' not found.")
    model = None

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Scaler loaded successfully!")
except FileNotFoundError:
    st.warning("Scaler file 'scaler.pkl' not found.")
    scaler = None

# Try to extract scaler feature names from the scaler itself
if scaler is not None and hasattr(scaler, "feature_names_in_"):
    scaler_feature_names = scaler.feature_names_in_.tolist()
else:
    scaler_feature_names = None
    st.warning("Scaler feature names not found or incompatible.")

st.title("Employee Attrition Prediction")
st.write("Enter employee details to predict attrition:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=60, value=30)
distance_from_home = st.number_input("DistanceFromHome", min_value=1, max_value=29, value=10)
environment_satisfaction = st.selectbox("EnvironmentSatisfaction", [1, 2, 3, 4])
job_involvement = st.selectbox("JobInvolvement", [1, 2, 3, 4])
job_satisfaction = st.selectbox("JobSatisfaction", [1, 2, 3, 4])
monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=20000, value=5000)
num_companies_worked = st.number_input("NumCompaniesWorked", min_value=0, max_value=9, value=2)
stock_option_level = st.selectbox("StockOptionLevel", [0, 1, 2, 3])
total_working_years = st.number_input("TotalWorkingYears", min_value=0, max_value=40, value=5)
years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=3)
years_in_current_role = st.number_input("YearsInCurrentRole", min_value=0, max_value=18, value=2)
years_with_curr_manager = st.number_input("YearsWithCurrManager", min_value=0, max_value=17, value=2)

business_travel = st.selectbox("BusinessTravel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
gender = st.selectbox("Gender", ['Female', 'Male'])
job_role = st.selectbox("JobRole", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                    'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                    'Sales Representative', 'Research Director', 'Human Resources'])
marital_status = st.selectbox("MaritalStatus", ['Single', 'Married', 'Divorced'])
over_time = st.selectbox("OverTime", ['No', 'Yes'])

# Create input dictionary
input_data = {
    'Age': age,
    'DistanceFromHome': distance_from_home,
    'EnvironmentSatisfaction': environment_satisfaction,
    'JobInvolvement': job_involvement,
    'JobSatisfaction': job_satisfaction,
    'MonthlyIncome': monthly_income,
    'NumCompaniesWorked': num_companies_worked,
    'StockOptionLevel': stock_option_level,
    'TotalWorkingYears': total_working_years,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': years_in_current_role,
    'YearsWithCurrManager': years_with_curr_manager,
    'BusinessTravel': business_travel,
    'Department': department,
    'Gender': gender,
    'JobRole': job_role,
    'MaritalStatus': marital_status,
    'OverTime': over_time
}

input_df = pd.DataFrame([input_data])

# One-hot encoding
input_df = pd.get_dummies(input_df, columns=['BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'], drop_first=True)

# Ensure all training columns are present
training_columns = ['Age', 'DistanceFromHome', 'EmployeeCount', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome',
    'NumCompaniesWorked', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently',
    'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
    'Department_Sales', 'Gender_Male', 'JobRole_Human Resources',
    'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'MaritalStatus_Married',
    'MaritalStatus_Single', 'OverTime_Yes']

for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[training_columns]

# Fill static features
input_df['EmployeeCount'] = 1
input_df['StandardHours'] = 80

# Apply scaler
if scaler is not None and scaler_feature_names is not None:
    try:
        input_df[scaler_feature_names] = scaler.transform(input_df[scaler_feature_names])
    except ValueError as e:
        st.error("Scaler error: Input features do not match training features.")
        st.stop()
else:
    st.warning("Scaler or feature names not available. Skipping scaling.")

# Prediction
if model is not None:
    if st.button("Predict Attrition"):
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[:, 1][0]

        st.subheader("Prediction Results:")
        if prediction == 1:
            st.error("Prediction: Employee is likely to Attrit")
        else:
            st.success("Prediction: Employee is likely to Stay")

        st.write(f"Probability of Attrition: {prediction_proba:.4f}")
else:
    st.warning("Model not loaded. Cannot make predictions.")
