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

# -------------------- Predict --------------------

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

    # -------- One Hot Encoding --------

    sex_value = 1 if sex == "Male" else 0
    input_data[f"sex_{sex_value}"] = 1

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }

    cp_value = cp_map[cp]
    input_data[f"cp_{cp_value}"] = 1

    fbs_value = 1 if fbs == "Yes" else 0
    input_data[f"fbs_{fbs_value}"] = 1

    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }

    restecg_value = restecg_map[restecg]
    input_data[f"restecg_{restecg_value}"] = 1

    exang_value = 1 if exang == "Yes" else 0
    input_data[f"exang_{exang_value}"] = 1

    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }

    slope_value = slope_map[slope]
    input_data[f"slope_{slope_value}"] = 1

    input_data[f"ca_{ca}"] = 1
    input_data[f"thal_{thal}"] = 1
        # -------------------- Prediction --------------------

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    if prediction == 0:
        confidence = probabilities[0] * 100
        disease = "No Heart Disease"

        if confidence >= 80:
            risk = "🟢 Low Risk"
        else:
            risk = "🟡 Medium Risk"

        st.success(f"💚 Prediction: {disease}")

    else:
        confidence = probabilities[1] * 100
        disease = "Heart Disease"

        if confidence >= 70:
            risk = "🔴 High Risk"
        else:
            risk = "🟡 Medium Risk"

        st.error(f"❤️ Prediction: {disease}")

    st.write(f"### Confidence: {confidence:.2f}%")
    st.write(f"### Risk Level: {risk}")

    st.markdown("---")

    st.subheader("Possible Risk Factors")

    found = False

    if age > 60:
        st.write("• Age above 60 years may increase the risk of heart disease.")
        found = True

    if trestbps > 140:
        st.write("• High resting blood pressure detected.")
        found = True

    if chol > 240:
        st.write("• High cholesterol level detected.")
        found = True

    if oldpeak > 2:
        st.write("• Elevated Oldpeak value detected.")
        found = True

    if exang == "Yes":
        st.write("• Exercise-induced angina is present.")
        found = True

    if not found:
        st.write("• No major risk factors identified from the entered values.")

    st.markdown("---")

    st.subheader("General Precautions")

    st.write("• Maintain a healthy and balanced diet.")
    st.write("• Exercise regularly after consulting a doctor.")
    st.write("• Avoid smoking and limit alcohol consumption.")
    st.write("• Monitor blood pressure and cholesterol regularly.")
    st.write("• Maintain a healthy body weight.")
    st.write("• Get regular health check-ups.")

    if prediction == 1:
        st.warning("⚠️ This prediction is generated by a machine learning model and is not a medical diagnosis. Please consult a healthcare professional.")
    else:
        st.info("✅ Continue following a healthy lifestyle and schedule routine health check-ups.")
