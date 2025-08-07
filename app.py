import streamlit as st
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")  # List of feature names

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("👩‍💼 Employee Attrition Predictor")
st.markdown("Predict if an employee is at risk of leaving the company.")

# ---- User Inputs ----
st.subheader("Employee Details")

# Prepare empty dict for inputs
user_input = {}

# Define mapping of feature names to input widgets:
# We'll create inputs only for features in selected_features

# Helper to parse feature types and set widgets
# Assume these 20 features come from your Colab training — adjust ranges based on data understanding

feature_inputs = {
    'Age': lambda: st.slider("Age", 18, 60, 30),
    'TotalWorkingYears': lambda: st.slider("Total Working Years", 0, 40, 10),
    'YearsWithCurrManager': lambda: st.slider("Years with Current Manager", 0, 20, 3),
    'YearsInCurrentRole': lambda: st.slider("Years in Current Role", 0, 20, 3),
    'YearsAtCompany': lambda: st.slider("Years at Company", 0, 40, 5),
    'MonthlyIncome': lambda: st.number_input("Monthly Income", 1000, 20000, 5000, step=500),
    'JobLevel': lambda: st.slider("Job Level", 1, 5, 2),
    'StockOptionLevel': lambda: st.slider("Stock Option Level", 0, 3, 1),
    'JobInvolvement': lambda: st.slider("Job Involvement", 1, 4, 3),
    'JobSatisfaction': lambda: st.slider("Job Satisfaction", 1, 4, 3),
    'EnvironmentSatisfaction': lambda: st.slider("Environment Satisfaction", 1, 4, 3),

    # Binary categorical dummies (1 or 0), get input as selectbox Yes/No or options
    'OverTime_Yes': lambda: 1 if st.selectbox("OverTime", ["No", "Yes"]) == "Yes" else 0,
    'MaritalStatus_Single': lambda: 1 if st.selectbox("Marital Status", ["Divorced", "Married", "Single"]) == "Single" else 0,
    'BusinessTravel_Travel_Frequently': lambda: 1 if st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]) == "Travel_Frequently" else 0,
    'Gender_Male': lambda: 1 if st.selectbox("Gender", ["Female", "Male"]) == "Male" else 0,

    'JobRole_Laboratory Technician': lambda: 1 if st.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", 
        "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
    ]) == "Laboratory Technician" else 0,

    'JobRole_Sales Representative': lambda: 1 if st.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", 
        "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
    ]) == "Sales Representative" else 0,

    'JobRole_Research Director': lambda: 1 if st.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", 
        "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
    ]) == "Research Director" else 0,

    'JobRole_Manager': lambda: 1 if st.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", 
        "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
    ]) == "Manager" else 0,

    'Department_Research & Development': lambda: 1 if st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"]) == "Research & Development" else 0,
}

# To avoid repeated multiple selects for same feature (Job Role), define it once:
job_role_choice = st.selectbox("Job Role", [
    "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", 
    "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
])
department_choice = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
marital_status_choice = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
overtime_choice = st.selectbox("OverTime", ["No", "Yes"])
gender_choice = st.selectbox("Gender", ["Female", "Male"])
business_travel_choice = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

# Gather inputs from user respecting the selected_features list
for feat in selected_features:
    if feat in feature_inputs:
        # For job role, override lambda to use stored choice
        if "JobRole_" in feat:
            user_input[feat] = 1 if feat == f"JobRole_{job_role_choice}" else 0
        elif feat == "Department_Research & Development":
            user_input[feat] = 1 if department_choice == "Research & Development" else 0
        elif feat == "MaritalStatus_Single":
            user_input[feat] = 1 if marital_status_choice == "Single" else 0
        elif feat == "OverTime_Yes":
            user_input[feat] = 1 if overtime_choice == "Yes" else 0
        elif feat == "Gender_Male":
            user_input[feat] = 1 if gender_choice == "Male" else 0
        elif feat == "BusinessTravel_Travel_Frequently":
            user_input[feat] = 1 if business_travel_choice == "Travel_Frequently" else 0
        else:
            # Numerical inputs
            user_input[feat] = feature_inputs[feat]()
    else:
        # If feature missing from inputs, default to zero
        user_input[feat] = 0

X_input = pd.DataFrame([user_input])

# Scale numerical columns only
num_cols = scaler.feature_names_in_
try:
    # Ensure missing scaler columns added as zero
    for col in num_cols:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[num_cols]
    X_scaled = scaler.transform(X_input)
    X_input.loc[:, num_cols] = X_scaled
except Exception as e:
    st.error("⚠️ Scaling error.")
    st.exception(e)

# Predict button
if st.button("🔮 Predict Attrition Risk"):
    try:
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]

        if pred == 1:
            st.error(f"⚠️ **High Attrition Risk!** Probability: {proba:.2%}")
        else:
            st.success(f"✅ **Low Attrition Risk** Probability: {proba:.2%}")
    except Exception as e:
        st.error("❌ Prediction error.")
        st.exception(e)

st.caption("Model trained with selected features | Scaling: MinMaxScaler | Logistic Regression")
