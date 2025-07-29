import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Risk Prediction")

st.markdown("Fill the details below to check your diabetes risk:")

with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Pregnancies", 0, 20, 1)
        gluc = st.number_input("Glucose", 0, 200, 100)
        bp = st.number_input("Blood Pressure", 0, 150, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 10, 100, 30)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[preg, gluc, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    result = "🟥 High Risk of Diabetes" if prediction[0] else "🟩 Low Risk of Diabetes"
    st.subheader("Prediction Result")
    st.success(result)

    st.subheader("📊 Feature Importance")
    image = Image.open("feature_importance.png")
    st.image(image, caption="Feature Contribution to Prediction")

    st.info("""
    **Interpretation Guide**:
    - **Glucose** and **BMI** often play a strong role in diabetes risk.
    - **Insulin** and **Age** also contribute significantly.
    """)

