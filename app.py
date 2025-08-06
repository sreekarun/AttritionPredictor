import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model, scaler, and feature columns
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")  # Must match model training

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("👩‍💼 Employee Attrition Predictor")
st.write("Enter employee details to predict if they are likely to leave the company.")

# Collect user input
input_data = {
    'Age': st.slider("Age", 18, 60, 30),
    'DistanceFromHome': st.slider("Distance From Home", 1, 30, 5),
    'MonthlyIncome': st.number_input("Monthly Income", 1000, 20000, 5000, step=500),
    'TotalWorkingYears': st.slider("Total Working Years", 0, 40, 10),
    'YearsAtCompany': st.slider("Years at Company", 0, 40, 5),
    'EmployeeCount': 1,
    'EnvironmentSatisfaction': 3,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobSatisfaction': 3,
    'NumCompaniesWorked': 2,
    'StandardHours': 80,
    'StockOptionLevel': 1,
    'YearsInCurrentRole': 3,
    'YearsWithCurrManager': 3,
    'Gender_Male': 1 if st.selectbox("Gender", ["Female", "Male"]) == "Male" else 0,
    'OverTime_Yes': 1 if st.selectbox("OverTime", ["No", "Yes"]) == "Yes" else 0,
}

# Manually encode select categorical features
bt = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
dept = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
edu = st.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])
jr = st.selectbox("Job Role", ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", 
                               "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"])
ms = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])

# One-hot encode manually
manual_oh = {
    'BusinessTravel_Travel_Frequently': bt == 'Travel_Frequently',
    'BusinessTravel_Travel_Rarely': bt == 'Travel_Rarely',
    'Department_Research & Development': dept == 'Research & Development',
    'Department_Sales': dept == 'Sales',
    'EducationField_Life Sciences': edu == 'Life Sciences',
    'EducationField_Marketing': edu == 'Marketing',
    'EducationField_Medical': edu == 'Medical',
    'EducationField_Other': edu == 'Other',
    'EducationField_Technical Degree': edu == 'Technical Degree',
    'JobRole_Human Resources': jr == 'Human Resources',
    'JobRole_Laboratory Technician': jr == 'Laboratory Technician',
    'JobRole_Manager': jr == 'Manager',
    'JobRole_Manufacturing Director': jr == 'Manufacturing Director',
    'JobRole_Research Director': jr == 'Research Director',
    'JobRole_Research Scientist': jr == 'Research Scientist',
    'JobRole_Sales Executive': jr == 'Sales Executive',
    'JobRole_Sales Representative': jr == 'Sales Representative',
    'MaritalStatus_Married': ms == 'Married',
    'MaritalStatus_Single': ms == 'Single'
}
for col, val in manual_oh.items():
    input_data[col] = 1 if val else 0

# Add any missing columns from training as zeros
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

# Convert and scale
X_input = pd.DataFrame([input_data])
X_input[scaler.feature_names_in_] = scaler.transform(X_input[scaler.feature_names_in_])
X_input = X_input[feature_columns]

# Prediction
if st.button("🔮 Predict Attrition"):
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Attrition Risk (Probability: {probability:.2f})")
