import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- Load Model --------------------
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below.")

# -------------------- User Inputs --------------------

age = st.number_input("Age", 1, 120, 50)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)

chol = st.number_input("Cholesterol", 100, 600, 200)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
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

thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)

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
    ["Upsloping", "Flat", "Downsloping"]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thalassemia",
    [0, 1, 2, 3]
)

# -------------------- Predict Button --------------------

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
    input_data[["age", "trestbps", "chol", "thalach", "oldpeak"]] = scaler.transform(
        input_data[["age", "trestbps", "chol", "thalach", "oldpeak"]]
    )

    # One-Hot Encoding

    input_data[f"sex_{1 if sex=='Male' else 0}"] = 1

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    input_data[f"cp_{cp_map[cp]}"] = 1

    input_data[f"fbs_{1 if fbs=='Yes' else 0}"] = 1

    rest_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    input_data[f"restecg_{rest_map[restecg]}"] = 1

    input_data[f"exang_{1 if exang=='Yes' else 0}"] = 1

    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    input_data[f"slope_{slope_map[slope]}"] = 1

    input_data[f"ca_{ca}"] = 1

    input_data[f"thal_{thal}"] = 1
        # -------------------- Prediction --------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    if probability < 30:
        risk = "🟢 Low Risk"
    elif probability < 70:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.markdown("---")

    if prediction == 1:
        st.error("❤️ Prediction: Heart Disease Detected")
    else:
        st.success("💚 Prediction: No Heart Disease")

    st.write(f"### Confidence: {probability:.2f}%")
    st.write(f"### Risk Level: {risk}")