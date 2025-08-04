import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

st.title("Employee Attrition Predictor")

# Example inputs - update as per your features
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
years_at_company = st.slider("Years at Company", 0, 40, 5)
gender_male = st.selectbox("Gender", ["Female", "Male"]) == "Male"
overtime_yes = st.selectbox("OverTime", ["No", "Yes"]) == "Yes"

input_data = np.array([
    age,
    monthly_income,
    years_at_company,
    int(gender_male),
    int(overtime_yes)
]).reshape(1, -1)

if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("Prediction: Likely to leave.")
    else:
        st.success("Prediction: Likely to stay.")
