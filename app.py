import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- Load model ---
try:
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except:
    st.error("Model file not found.")
    model = None

# --- Load scaler ---
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    st.success("Scaler loaded successfully!")
except:
    scaler = None
    st.warning("Scaler file not found. Predictions will use unscaled features.")

# --- Define model input features from training ---
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

# --- UI ---
st.title("Employee Attrition Prediction App")
st.subheader("Fill in employee details to predict attrition risk")

# Main features
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
    total_working_years = st.number_input("Total Working Years", 0, 40, 5)
    years_at_company = st.number_input("Years at Company", 0, 40, 3)
    distance_from_home = st.number_input("Distance From Home (km)", 0, 30, 5)
    
with col2:
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
    job_level = st.slider("Job Level (1-5)", 1, 5, 2)
    stock_option_level = st.slider("Stock Option Level (0-3)", 0, 3, 1)

# Categorical features
st.subheader("Additional Information")
col3, col4 = st.columns(2)
with col3:
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.selectbox("Department", ["Research & Development", "Sales", "Other"])
    
with col4:
    education_field = st.selectbox("Education Field", [
        "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"
    ])
    job_role = st.selectbox("Job Role", [
        "Research Scientist", "Sales Executive", "Laboratory Technician", "Manager", "Other"
    ])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_companies_worked = st.number_input("Number of Companies Worked", 0, 10, 2)

# --- Build complete input dict ---
input_data = {
    # Numerical features
    "Age": age,
    "MonthlyIncome": monthly_income,
    "TotalWorkingYears": total_working_years,
    "YearsAtCompany": years_at_company,
    "DistanceFromHome": distance_from_home,
    "JobSatisfaction": job_satisfaction,
    "EnvironmentSatisfaction": environment_satisfaction,
    "JobInvolvement": job_involvement,
    "JobLevel": job_level,
    "StockOptionLevel": stock_option_level,
    "NumCompaniesWorked": num_companies_worked,
    "EmployeeCount": 1,  # Default value
    "StandardHours": 80,  # Default value
    "YearsInCurrentRole": years_at_company - 1 if years_at_company > 0 else 0,  # Approximation
    "YearsWithCurrManager": years_at_company - 1 if years_at_company > 0 else 0,  # Approximation
    
    # Categorical features (dummy variables)
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
    "BusinessTravel_Travel_Frequently": 1 if business_travel == "Travel_Frequently" else 0,
    "BusinessTravel_Travel_Rarely": 1 if business_travel == "Travel_Rarely" else 0,
    "Department_Research & Development": 1 if department == "Research & Development" else 0,
    "Department_Sales": 1 if department == "Sales" else 0,
    "EducationField_Life Sciences": 1 if education_field == "Life Sciences" else 0,
    "EducationField_Medical": 1 if education_field == "Medical" else 0,
    "EducationField_Marketing": 1 if education_field == "Marketing" else 0,
    "EducationField_Technical Degree": 1 if education_field == "Technical Degree" else 0,
    "EducationField_Other": 1 if education_field == "Other" else 0,
    "JobRole_Research Scientist": 1 if job_role == "Research Scientist" else 0,
    "JobRole_Sales Executive": 1 if job_role == "Sales Executive" else 0,
    "JobRole_Laboratory Technician": 1 if job_role == "Laboratory Technician" else 0,
    "JobRole_Manager": 1 if job_role == "Manager" else 0,
    "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
    "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
    
    # Set all other expected features to 0
    **{feature: 0 for feature in model_features if feature not in [
        'Age', 'DistanceFromHome', 'EmployeeCount', 'EnvironmentSatisfaction', 
        'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
        'NumCompaniesWorked', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager', 'Gender_Male', 
        'OverTime_Yes', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
        'Department_Research & Development', 'Department_Sales', 'EducationField_Life Sciences',
        'EducationField_Medical', 'EducationField_Marketing', 'EducationField_Technical Degree',
        'EducationField_Other', 'JobRole_Research Scientist', 'JobRole_Sales Executive',
        'JobRole_Laboratory Technician', 'JobRole_Manager', 'MaritalStatus_Married',
        'MaritalStatus_Single'
    ]}
}

# --- Prediction ---
if model and st.button("Predict Attrition Risk"):
    try:
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([input_data])[model_features]
        
        # Scale features if scaler is available
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df
            
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Display results
        st.subheader("Results")
        if prediction[0] == 1:
            st.error(f"High Risk of Attrition ({prediction_proba[0][1]*100:.1f}% probability)")
        else:
            st.success(f"Low Risk of Attrition ({prediction_proba[0][0]*100:.1f}% probability)")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error("Please check that all input values are valid.")