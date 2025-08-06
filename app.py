import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Attrition Predictor", layout="wide")

# Function to preprocess uploaded data
def preprocess_data(df):
    # Drop unnecessary columns
    columns_to_drop = [
        'PerformanceRating', 'HourlyRate', 'EmployeeNumber', 'PercentSalaryHike',
        'Education', 'YearsSinceLastPromotion', 'RelationshipSatisfaction',
        'MonthlyRate', 'DailyRate', 'TrainingTimesLastYear', 'WorkLifeBalance'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # One-hot encode specific columns
    df = pd.get_dummies(df, columns=['Gender', 'OverTime', 'Attrition'], drop_first=True)

    # One-hot encode remaining object/categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler

# Upload section
st.title("🔍 Employee Attrition Prediction App")
uploaded_file = st.file_uploader("📁 Upload the HR Employee Attrition CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess
    df_raw = pd.read_csv(uploaded_file)
    df_processed, scaler = preprocess_data(df_raw)

    # Split X and y
    X = df_processed.drop('Attrition_Yes', axis=1)
    y = df_processed['Attrition_Yes']
    feature_columns = X.columns

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_resampled, y_resampled)

    st.success("✅ Model trained successfully on uploaded data.")

    # User input section
    st.header("👤 Enter New Employee Data for Prediction")

    # Collect user input
    input_data = {}
    input_data['Age'] = st.slider('Age', 18, 60, 30)
    input_data['DistanceFromHome'] = st.slider('Distance From Home (miles)', 1, 30, 5)
    input_data['EmployeeCount'] = 1
    input_data['EnvironmentSatisfaction'] = st.slider('Environment Satisfaction', 1, 4, 3)
    input_data['JobInvolvement'] = st.slider('Job Involvement', 1, 4, 3)
    input_data['JobLevel'] = st.selectbox('Job Level', [1, 2, 3, 4, 5])
    input_data['JobSatisfaction'] = st.slider('Job Satisfaction', 1, 4, 3)
    input_data['MonthlyIncome'] = st.number_input('Monthly Income', 1000, 20000, 5000)
    input_data['NumCompaniesWorked'] = st.slider('Number of Companies Worked', 0, 10, 2)
    input_data['StandardHours'] = 80
    input_data['StockOptionLevel'] = st.selectbox('Stock Option Level', [0, 1, 2, 3])
    input_data['TotalWorkingYears'] = st.slider('Total Working Years', 0, 40, 10)
    input_data['YearsAtCompany'] = st.slider('Years at Company', 0, 40, 5)
    input_data['YearsInCurrentRole'] = st.slider('Years in Current Role', 0, 18, 3)
    input_data['YearsWithCurrManager'] = st.slider('Years with Current Manager', 0, 17, 3)

    gender = st.selectbox("Gender", ['Female', 'Male'])
    overtime = st.selectbox("OverTime", ['No', 'Yes'])
    business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    department = st.selectbox("Department", ['Human Resources', 'Research & Development', 'Sales'])
    education_field = st.selectbox("Education Field", ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'])
    job_role = st.selectbox("Job Role", ['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                                         'Manager', 'Manufacturing Director', 'Research Director',
                                         'Research Scientist', 'Sales Executive', 'Sales Representative'])
    marital_status = st.selectbox("Marital Status", ['Divorced', 'Married', 'Single'])

    # Encode input
    encoded_input = pd.DataFrame([input_data])
    encoded_input['Gender_Male'] = 1 if gender == 'Male' else 0
    encoded_input['OverTime_Yes'] = 1 if overtime == 'Yes' else 0

    # One-hot encoding for other categorical fields
    category_map = {
        'BusinessTravel_Travel_Frequently': business_travel == 'Travel_Frequently',
        'BusinessTravel_Travel_Rarely': business_travel == 'Travel_Rarely',
        'Department_Research & Development': department == 'Research & Development',
        'Department_Sales': department == 'Sales',
        'EducationField_Life Sciences': education_field == 'Life Sciences',
        'EducationField_Marketing': education_field == 'Marketing',
        'EducationField_Medical': education_field == 'Medical',
        'EducationField_Other': education_field == 'Other',
        'EducationField_Technical Degree': education_field == 'Technical Degree',
        'JobRole_Human Resources': job_role == 'Human Resources',
        'JobRole_Laboratory Technician': job_role == 'Laboratory Technician',
        'JobRole_Manager': job_role == 'Manager',
        'JobRole_Manufacturing Director': job_role == 'Manufacturing Director',
        'JobRole_Research Director': job_role == 'Research Director',
        'JobRole_Research Scientist': job_role == 'Research Scientist',
        'JobRole_Sales Executive': job_role == 'Sales Executive',
        'JobRole_Sales Representative': job_role == 'Sales Representative',
        'MaritalStatus_Married': marital_status == 'Married',
        'MaritalStatus_Single': marital_status == 'Single'
    }

    for col in category_map:
        encoded_input[col] = 1 if category_map[col] else 0

    # Add missing columns
    for col in feature_columns:
        if col not in encoded_input.columns:
            encoded_input[col] = 0

    # Reorder to match model
    encoded_input = encoded_input[feature_columns]

    # Scale numeric columns
    encoded_input[scaler.feature_names_in_] = scaler.transform(encoded_input[scaler.feature_names_in_])

    # Predict
    if st.button("🔮 Predict Attrition"):
        prediction = model.predict(encoded_input)[0]
        probability = model.predict_proba(encoded_input)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Likely to Leave! (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Likely to Stay (Probability: {probability:.2f})")

else:
    st.info("👆 Please upload the CSV file to continue.")
