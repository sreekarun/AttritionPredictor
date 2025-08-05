import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Load Model and Scaler ---
try:
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None

# --- Define expected features based on your training code ---
expected_features = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'Gender_Male', 'OverTime_Yes', 'Attrition_Yes',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Human Resources', 'EducationField_Life Sciences',
    'EducationField_Marketing', 'EducationField_Medical',
    'EducationField_Other', 'EducationField_Technical Degree',
    'JobRole_Healthcare Representative', 'JobRole_Human Resources',
    'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'MaritalStatus_Married',
    'MaritalStatus_Single'
]

# Filter to only include features actually used in your model (excluding dropped columns)
used_features = [
    'Age', 'DistanceFromHome', 'EmployeeCount', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'NumCompaniesWorked', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsWithCurrManager', 'Gender_Male', 'OverTime_Yes',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'JobRole_Human Resources',
    'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'MaritalStatus_Married',
    'MaritalStatus_Single'
]

# --- Streamlit App UI ---
st.title("Employee Attrition Prediction App")
st.subheader("Fill in employee details to predict attrition risk")

# Initialize input dictionary with all expected features set to 0
input_data = {feature: 0 for feature in used_features}

# Main features
col1, col2 = st.columns(2)
with col1:
    input_data['Age'] = st.number_input("Age", min_value=18, max_value=60, value=30)
    input_data['MonthlyIncome'] = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    input_data['TotalWorkingYears'] = st.number_input("Total Working Years", 0, 40, 5)
    input_data['YearsAtCompany'] = st.number_input("Years at Company", 0, 40, 3)
    input_data['DistanceFromHome'] = st.number_input("Distance From Home (km)", 0, 30, 5)
    
with col2:
    input_data['JobSatisfaction'] = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    input_data['EnvironmentSatisfaction'] = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    input_data['JobInvolvement'] = st.slider("Job Involvement (1-4)", 1, 4, 3)
    input_data['JobLevel'] = st.slider("Job Level (1-5)", 1, 5, 2)
    input_data['StockOptionLevel'] = st.slider("Stock Option Level (0-3)", 0, 3, 1)

# Categorical features
st.subheader("Additional Information")
col3, col4 = st.columns(2)
with col3:
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    input_data['OverTime_Yes'] = 1 if overtime == "Yes" else 0
    
    gender = st.selectbox("Gender", ["Female", "Male"])
    input_data['Gender_Male'] = 1 if gender == "Male" else 0
    
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    input_data['BusinessTravel_Travel_Frequently'] = 1 if business_travel == "Travel_Frequently" else 0
    input_data['BusinessTravel_Travel_Rarely'] = 1 if business_travel == "Travel_Rarely" else 0
    
    department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    input_data['Department_Research & Development'] = 1 if department == "Research & Development" else 0
    input_data['Department_Sales'] = 1 if department == "Sales" else 0
    
with col4:
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
    input_data['EducationField_Life Sciences'] = 1 if education_field == "Life Sciences" else 0
    input_data['EducationField_Marketing'] = 1 if education_field == "Marketing" else 0
    input_data['EducationField_Medical'] = 1 if education_field == "Medical" else 0
    input_data['EducationField_Other'] = 1 if education_field == "Other" else 0
    input_data['EducationField_Technical Degree'] = 1 if education_field == "Technical Degree" else 0
    
    job_role = st.selectbox("Job Role", [
        "Research Scientist", "Sales Executive", "Laboratory Technician", 
        "Manager", "Manufacturing Director", "Research Director",
        "Human Resources", "Sales Representative"
    ])
    input_data['JobRole_Research Scientist'] = 1 if job_role == "Research Scientist" else 0
    input_data['JobRole_Sales Executive'] = 1 if job_role == "Sales Executive" else 0
    input_data['JobRole_Laboratory Technician'] = 1 if job_role == "Laboratory Technician" else 0
    input_data['JobRole_Manager'] = 1 if job_role == "Manager" else 0
    input_data['JobRole_Manufacturing Director'] = 1 if job_role == "Manufacturing Director" else 0
    input_data['JobRole_Research Director'] = 1 if job_role == "Research Director" else 0
    input_data['JobRole_Human Resources'] = 1 if job_role == "Human Resources" else 0
    input_data['JobRole_Sales Representative'] = 1 if job_role == "Sales Representative" else 0
    
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    input_data['MaritalStatus_Married'] = 1 if marital_status == "Married" else 0
    input_data['MaritalStatus_Single'] = 1 if marital_status == "Single" else 0

# Set default values for remaining features
input_data['EmployeeCount'] = 1
input_data['StandardHours'] = 80
input_data['NumCompaniesWorked'] = st.number_input("Number of Companies Worked", 0, 10, 2)
input_data['YearsInCurrentRole'] = max(input_data['YearsAtCompany'] - 1, 0)
input_data['YearsWithCurrManager'] = max(input_data['YearsAtCompany'] - 1, 0)

# --- Prediction ---
if model and st.button("Predict Attrition Risk"):
    try:
        # Create DataFrame ensuring correct feature order
        input_df = pd.DataFrame([input_data])[used_features]
        
        # Scale the input data (using the same scaler from training)
        # Note: In a real app, you should load the scaler used during training
        scaler = MinMaxScaler()
        input_scaled = scaler.fit_transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Display results
        st.subheader("Results")
        if prediction[0] == 1:
            st.error(f"🚨 High Risk of Attrition ({prediction_proba[0][1]*100:.1f}% probability)")
        else:
            st.success(f"✅ Low Risk of Attrition ({prediction_proba[0][0]*100:.1f}% probability)")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error("Please verify that all input values are correct and try again.")