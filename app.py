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
    st.error("Error: Model file 'logistic_regression_model.pkl' not found. Please ensure it's in the same directory.")
    model = None # Set model to None if loading fails

# Load the scaler (assuming you saved it after fitting on the training data)
# If you didn't save the scaler, you would need to re-fit it on a representative dataset
# or use the min/max values from your training data to manually scale.
# For this example, let's assume the scaler was saved as 'scaler.pkl'
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Scaler loaded successfully!")
except FileNotFoundError:
     st.warning("Scaler file 'scaler.pkl' not found. Proceeding without explicit scaling.")
     scaler = None


st.title("Employee Attrition Prediction")

st.write("Enter employee details to predict attrition:")

# Define the input fields based on the features used for training
# You need to replicate the feature engineering and encoding steps here.
# Based on your original notebook, the features after preprocessing were:
# ['Age', 'DistanceFromHome', 'EmployeeCount', 'EnvironmentSatisfaction',
# 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
# 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany',
# 'YearsInCurrentRole', 'YearsWithCurrManager',
# 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
# 'Department_Research & Development', 'Department_Sales',
# 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
# 'JobRole_Manager', 'JobRole_Manufacturing Director',
# 'JobRole_Research Director', 'JobRole_Research Scientist',
# 'JobRole_Sales Executive', 'JobRole_Sales Representative',
# 'MaritalStatus_Married', 'MaritalStatus_Single', 'Gender_Male', 'OverTime_Yes']

# Create input fields for numerical features
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


# Create input fields for categorical features (using selectbox for options)
business_travel = st.selectbox("BusinessTravel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
gender = st.selectbox("Gender", ['Female', 'Male'])
job_role = st.selectbox("JobRole", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                     'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                     'Sales Representative', 'Research Director', 'Human Resources'])
marital_status = st.selectbox("MaritalStatus", ['Single', 'Married', 'Divorced'])
over_time = st.selectbox("OverTime", ['No', 'Yes'])


# Create a dictionary from the input values
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

# Convert the input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])


# --- Preprocessing steps (must match training preprocessing) ---

# Apply one-hot encoding to categorical features
input_df = pd.get_dummies(input_df, columns=['BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'], drop_first=True)

# Ensure all columns from training data are present, fill missing with 0
# This is crucial to match the number of features the model expects
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
       'MaritalStatus_Single', 'OverTime_Yes'] # Add EmployeeCount and StandardHours back if they were in X_train


# Add missing columns with default value 0
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match the order of the training data
input_df = input_df[training_columns]


# Apply scaling to numerical features
# Identify numerical columns AFTER one-hot encoding (excluding dummy variables)
numerical_cols_after_encoding = ['Age', 'DistanceFromHome', 'EnvironmentSatisfaction',
       'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome',
       'NumCompaniesWorked', 'StockOptionLevel',
       'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsWithCurrManager'] # Ensure these are the numerical columns that were scaled during training

if scaler is not None:
    input_df[numerical_cols_after_encoding] = scaler.transform(input_df[numerical_cols_after_encoding])
else:
    st.warning("Scaler not loaded. Numerical features are not scaled.")


# --- Prediction ---
if model is not None:
    if st.button("Predict Attrition"):
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[:, 1][0]

        st.subheader("Prediction Results:")
        if prediction == 1:
            st.error(f"Prediction: Employee is likely to Attrit")
        else:
            st.success(f"Prediction: Employee is likely to Stay")

        st.write(f"Probability of Attrition: {prediction_proba:.4f}")

else:
    st.warning("Model not loaded. Cannot make predictions.")