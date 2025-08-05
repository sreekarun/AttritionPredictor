import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.title("Employee Attrition Prediction")

# User input form
with st.form("input_form"):
    st.subheader("Enter Employee Details")
    
    Age = st.slider("Age", 18, 60, 30)
    BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    DistanceFromHome = st.slider("Distance From Home (miles)", 1, 30, 5)
    EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    MonthlyIncome = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
    NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 10, 2)
    PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25, 15)
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
    TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 10, 3)
    YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
    YearsInCurrentRole = st.slider("Years in Current Role", 0, 20, 3)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    YearsWithCurrManager = st.slider("Years with Current Manager", 0, 20, 4)

    submit = st.form_submit_button("Predict")

# Handle form submission
if submit:
    input_dict = {
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "EducationField": EducationField,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "NumCompaniesWorked": NumCompaniesWorked,
        "PercentSalaryHike": PercentSalaryHike,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Add required missing features with default values
    required_columns = ['EmployeeCount', 'JobLevel', 'StandardHours']
    default_values = {'EmployeeCount': 1, 'JobLevel': 1, 'StandardHours': 40}

    for col in required_columns:
        input_df[col] = default_values[col]

    # Handle categorical encoding
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
    if encoder:
        encoded_df = pd.DataFrame(encoder.transform(input_df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
        input_df = input_df.drop(columns=cat_cols)
        input_df = pd.concat([input_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    else:
        st.warning("Encoder not loaded properly.")

    # Ensure column order matches training data
    numerical_cols_after_encoding = input_df.columns  # Assuming same order

    # Scale numerical features
    if scaler:
        input_df[numerical_cols_after_encoding] = scaler.transform(input_df[numerical_cols_after_encoding])
    else:
        st.warning("Scaler not loaded properly.")

    # Predict
    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)[0][1]

    # Output
    if prediction[0] == 1:
        st.error(f"Prediction: Employee is likely to leave. (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"Prediction: Employee is likely to stay. (Probability: {1 - prediction_prob:.2f})")
