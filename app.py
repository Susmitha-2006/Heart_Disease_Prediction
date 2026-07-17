import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# Load Saved Files
# ==========================

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ==========================
# Streamlit Page Settings
# ==========================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="centered"
)
st.markdown("""
<style>
a {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Title
# ==========================

st.title("🫀 Heart Disease Prediction System")

st.write(
    "Enter the patient's medical details below to predict the likelihood of heart disease."
)

st.markdown("---")

# ==========================
# Patient Details
# ==========================

st.header("Patient Details")
# ==========================
# Input Fields
# ==========================

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

trestbps = st.number_input(
    "Resting Blood Pressure (mmHg)",
    min_value=80,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    ["No", "Yes"]
)

restecg = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)
thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thalassemia",
    [0, 1, 2, 3]
)

st.markdown("---")

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Numerical Features
    input_data["age"] = age
    input_data["trestbps"] = trestbps
    input_data["chol"] = chol
    input_data["thalach"] = thalach
    input_data["oldpeak"] = oldpeak

    # Scale Numerical Features
    input_data[
        ["age", "trestbps", "chol", "thalach", "oldpeak"]
    ] = scaler.transform(
        input_data[
            ["age", "trestbps", "chol", "thalach", "oldpeak"]
        ]
    )
        # ==========================
    # One-Hot Encoding
    # ==========================

    # Gender
    gender_value = 1 if sex == "Male" else 0
    input_data[f"sex_{gender_value}"] = 1

    # Chest Pain Type
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    input_data[f"cp_{cp_map[cp]}"] = 1

    # Fasting Blood Sugar
    fbs_value = 1 if fbs == "Yes" else 0
    input_data[f"fbs_{fbs_value}"] = 1

    # Resting ECG
    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    input_data[f"restecg_{restecg_map[restecg]}"] = 1

    # Exercise-Induced Angina
    exang_value = 1 if exang == "Yes" else 0
    input_data[f"exang_{exang_value}"] = 1

    # Slope
    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    input_data[f"slope_{slope_map[slope]}"] = 1

    # Number of Major Vessels
    input_data[f"ca_{ca}"] = 1

    # Thalassemia
    input_data[f"thal_{thal}"] = 1
        # ==========================
    # Prediction
    # ==========================

    prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]

# -------------------------
# Disease Prediction
# -------------------------

if prediction == 0:
    confidence = probabilities[0] * 100
    disease = "No Heart Disease"

    st.success(f"💚 Prediction: {disease}")

else:
    confidence = probabilities[1] * 100
    disease = "Heart Disease"

    st.error(f"❤️ Prediction: {disease}")

# -------------------------
# Risk Score Calculation
# -------------------------

risk_score = 0

# Age
if age >= 60:
    risk_score += 1

# Blood Pressure
if trestbps >= 140:
    risk_score += 1

# Cholesterol
if chol >= 240:
    risk_score += 1

# Fasting Blood Sugar
if fbs == "Yes":
    risk_score += 1

# Exercise-Induced Angina
if exang == "Yes":
    risk_score += 1

# Oldpeak
if oldpeak >= 2:
    risk_score += 1

# Number of Major Vessels
if ca >= 2:
    risk_score += 1

# Chest Pain
if cp == "Asymptomatic":
    risk_score += 2
elif cp == "Non-anginal Pain":
    risk_score += 1

# Maximum Heart Rate
if thalach < 120:
    risk_score += 1

# Model Prediction
if prediction == 1:
    risk_score += 2

# -------------------------
# Final Risk Level
# -------------------------

if risk_score <= 2:
    risk = "🟢 Low Risk"

elif risk_score <= 5:
    risk = "🟡 Medium Risk"

else:
    risk = "🔴 High Risk"

st.write(f"### Confidence: {confidence:.2f}%")
st.write(f"### Estimated Risk Level: {risk}")

    st.markdown("---")
        # ==========================
    # Possible Risk Factors
    # ==========================

    st.subheader("Possible Risk Factors")

    risk_found = False

    if age >= 60:
        risk_found = True
        st.write(f"👴 **Age:** {age} years (Higher age increases heart disease risk)")

    if trestbps > 140:
        risk_found = True
        st.write(f"🩸 **Resting Blood Pressure:** {trestbps} mmHg (High)")

    if chol > 240:
        risk_found = True
        st.write(f"🫀 **Cholesterol:** {chol} mg/dL (High)")

    if oldpeak > 2:
        risk_found = True
        st.write(f"📈 **Oldpeak:** {oldpeak} (High)")

    if exang == "Yes":
        risk_found = True
        st.write("🏃 **Exercise-Induced Angina:** Yes")

    if fbs == "Yes":
        risk_found = True
        st.write("🍬 **Fasting Blood Sugar:** Above 120 mg/dL")

    if not risk_found:
        st.success("✅ No significant risk factors identified from the entered values.")

    st.markdown("---")
        # ==========================
    # Risk-Based Precautions
    # ==========================

    st.subheader("Risk-Based Precautions")

    precautions_given = False

    if chol > 240:
        precautions_given = True
        st.write("🫀 **High Cholesterol**")
        st.write("• Reduce fried, oily, and high-fat foods.")
        st.write("• Eat more fruits, vegetables, and whole grains.")
        st.write("• Exercise regularly (as advised by your doctor).")
        st.write("• Monitor your cholesterol levels regularly.")
        st.write("")

    if trestbps > 140:
        precautions_given = True
        st.write("🩸 **High Blood Pressure**")
        st.write("• Reduce salt intake.")
        st.write("• Avoid stress whenever possible.")
        st.write("• Monitor your blood pressure regularly.")
        st.write("• Take medications as prescribed.")
        st.write("")

    if oldpeak > 2:
        precautions_given = True
        st.write("📈 **High Oldpeak**")
        st.write("• Avoid strenuous physical activity until medically evaluated.")
        st.write("• Consult a cardiologist.")
        st.write("• Follow your doctor's advice regarding further heart tests.")
        st.write("")

    if exang == "Yes":
        precautions_given = True
        st.write("🏃 **Exercise-Induced Angina**")
        st.write("• Avoid heavy exercise until you have medical clearance.")
        st.write("• Stop exercising if chest pain occurs.")
        st.write("• Consult a cardiologist before starting a new exercise program.")
        st.write("")

    if fbs == "Yes":
        precautions_given = True
        st.write("🍬 **High Fasting Blood Sugar**")
        st.write("• Reduce sugar and refined carbohydrate intake.")
        st.write("• Follow a balanced diet.")
        st.write("• Monitor blood sugar regularly.")
        st.write("• Exercise as recommended by your doctor.")
        st.write("")

    if age >= 60:
        precautions_given = True
        st.write("👴 **Age Above 60 Years**")
        st.write("• Schedule regular heart health check-ups.")
        st.write("• Stay physically active according to your doctor's advice.")
        st.write("• Monitor blood pressure and cholesterol routinely.")
        st.write("")

    if not precautions_given:
        st.success("✅ No specific precautions are required based on the entered values.")
        st.write("• Continue maintaining a healthy lifestyle.")
        st.write("• Eat a balanced diet.")
        st.write("• Exercise regularly.")
        st.write("• Go for routine health check-ups.")

    st.markdown("---")
        # ==========================
    # Final Recommendation
    # ==========================

    if prediction == 1:
        st.warning(
            "⚠️ **Disclaimer:** This prediction is generated by a machine learning model and is not a medical diagnosis. "
            "Please consult a qualified healthcare professional for proper medical evaluation and treatment."
        )
    else:
        st.info(
            "💚 The model predicts a low likelihood of heart disease based on the entered values. "
            "Continue maintaining a healthy lifestyle and attend regular health check-ups."
        )

    st.markdown("---")

    st.caption(
        "Developed using Machine Learning (K-Nearest Neighbors) and Streamlit for educational purposes."
    )
