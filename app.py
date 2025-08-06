import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Attrition Predictor", layout="centered")

# Load and preprocess the default dataset
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # Drop columns
    columns_to_drop = [
        'PerformanceRating', 'HourlyRate', 'EmployeeNumber', 'PercentSalaryHike',
        'Education', 'YearsSinceLastPromotion', 'RelationshipSatisfaction',
        'MonthlyRate', 'DailyRate', 'TrainingTimesLastYear', 'WorkLifeBalance'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Drop missing or bad Attrition rows (SMOTE fix)
    df = df[df["Attrition"].notna()]

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Gender', 'OverTime', 'Attrition'], drop_first=True)
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale
    numerical_cols = df.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler

# Train the model
@st.cache_resource
def train_model():
    df, scaler = load_and_prepare_data()
    X = df.drop('Attrition_Yes', axis=1)
    y = df['Attrition_Yes']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    model = LogisticRegression(random_state=42)
    model.fit(X_res, y_res)

    return model, scaler, X.columns.tolist()

# App title
st.title("👩‍💼 Employee Attrition Predictor")
st.write("Fill in the employee details below to estimate attrition risk.")

# Load trained model
model, scaler, feature_columns = train_model()

# ---- Input Form ----
input_data = {}

# Required Inputs (some set with defaults)
input_data['Age'] = st.slider("Age", 18, 60, 35)
input_data['DistanceFromHome'] = st.slider("Distance From Home (miles)", 1, 30, 5)
input_data['MonthlyIncome'] = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
input_data['TotalWorkingYears'] = st.slider("Total Working Years", 0, 40, 10)
input_data['YearsAtCompany'] = st.slider("Years at Company", 0, 40, 5)

# Optional / Auto-filled
input_data['EmployeeCount'] = 1
input_data['EnvironmentSatisfaction'] = 3
input_data['JobInvolvement'] = 3
input_data['JobLevel'] = 2
input_data['JobSatisfaction'] = 3
input_data['NumCompaniesWorked'] = 2
input_data['StandardHours'] = 80
input_data['StockOptionLevel'] = 1
input_data['YearsInCurrentRole'] = 3
input_data['YearsWithCurrManager'] = 3

# Binary categorical
input_data['Gender_Male'] = 1 if st.selectbox("Gender", ["Female", "Male"]) == "Male" else 0
input_data['OverTime_Yes'] = 1 if st.selectbox("OverTime", ["No", "Yes"]) == "Yes" else 0

# One-hot categorical
bt = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
dept = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
edu = st.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])
jr = st.selectbox("Job Role", ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", 
                               "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"])
ms = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])

# Manually one-hot encode
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

# Fill in missing feature columns with 0
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

# Prepare and scale input
X_input = pd.DataFrame([input_data])
scaled_cols = scaler.feature_names_in_
X_input[scaled_cols] = scaler.transform(X_input[scaled_cols])
X_input = X_input[feature_columns]

# ---- Prediction ----
if st.button("🔮 Predict Attrition"):
    prediction = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Attrition Risk (Probability: {prob:.2f})")
