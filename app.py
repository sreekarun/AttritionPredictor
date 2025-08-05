import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'logistic_regression_model.pkl' not found.")
    model = None

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Scaler loaded successfully!")
except FileNotFoundError:
    st.warning("Scaler file 'scaler.pkl' not found. Numerical inputs will NOT be scaled.")
    scaler = None

st.title("Employee Attrition Prediction")
st.write("Enter employee details to predict attrition:")

# --- Input fields ---
age = st.number_input("Age", min_value=18, max_value=60, value=30)
distance_from_home = st.number_input("Distance From Home", min_value=1, max_value=29, value=10)
environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=9, value=2)
stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=18, value=2)
years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=17, value=2)

business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
gender = st.selectbox("Gender", ['Female', 'Male'])
job_role = st.selectbox("Job Role", [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative', 'Manager',
    'Sales Representative', 'Research Director', 'Human Resources'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
over_time = st.selectbox("OverTime", ['No', 'Yes'])

# --- Create input dataframe ---
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

df_input = pd.DataFrame([input_data])

# --- One-hot encoding ---
df_input = pd.get_dummies(df_input, columns=['BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'], drop_first=True)

# --- Add missing columns and reorder ---
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

# Add missing columns with 0
for col in training_columns:
    if col not in df_input.columns:
        df_input[col] = 0

# Reorder columns
df_input = df_input[training_columns]

# --- Scale numerical features ---
numerical_cols = ['Age', 'DistanceFromHome', 'EnvironmentSatisfaction',
                  'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome',
                  'NumCompaniesWorked', 'StockOptionLevel', 'TotalWorkingYears',
                  'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']

if scaler is not None:
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
else:
    st.warning("Scaler not loaded, numerical features are NOT scaled.")

# --- Prediction ---
if model is not None:
    if st.button("Predict Attrition"):
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

        st.subheader("Prediction Results:")
        if prediction == 1:
            st.error("Prediction: Employee is likely to attrit.")
        else:
            st.success("Prediction: Employee is likely to stay.")

        st.write(f"Probability of attrition: {proba:.4f}")
else:
    st.warning("Model not loaded. Cannot make predictions.")
