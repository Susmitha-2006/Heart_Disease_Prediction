import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- Load Model --------------------

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below.")

# -------------------- User Inputs --------------------

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
    "Resting Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

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

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
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

    input_data[
        ["age", "trestbps", "chol", "thalach", "oldpeak"]
    ] = scaler.transform(
        input_data[
            ["age", "trestbps", "chol", "thalach", "oldpeak"]
        ]
    )
        # -------------------- One-Hot Encoding --------------------

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
        # -------------------- Possible Risk Factors --------------------

    st.subheader("Possible Risk Factors")

    risk_found = False

    if age >= 60:
        risk_found = True
        st.write("👴 **Age Above 60 Years**")

    if trestbps > 140:
        risk_found = True
        st.write("🩸 **High Resting Blood Pressure Detected**")

    if chol > 240:
        risk_found = True
        st.write("🫀 **High Cholesterol Level Detected**")

    if oldpeak > 2:
        risk_found = True
        st.write("📈 **High Oldpeak Value Detected**")

    if exang == "Yes":
        risk_found = True
        st.write("🏃 **Exercise-Induced Angina Detected**")

    if fbs == "Yes":
        risk_found = True
        st.write("🍬 **High Fasting Blood Sugar Detected**")

    if not risk_found:
        st.success("✅ No major risk factors identified from the entered values.")

    st.markdown("---")

    # -------------------- Risk-Based Precautions --------------------

    st.subheader("Risk-Based Precautions")

    if chol > 240:
        st.write("🫀 **For High Cholesterol:**")
        st.write("• Reduce fried and oily foods.")
        st.write("• Eat more fruits, vegetables and whole grains.")
        st.write("• Exercise regularly.")
        st.write("• Monitor cholesterol periodically.")
        st.write("")

    if trestbps > 140:
        st.write("🩸 **For High Blood Pressure:**")
        st.write("• Reduce salt intake.")
        st.write("• Avoid excessive stress.")
        st.write("• Monitor blood pressure regularly.")
        st.write("• Take medicines as prescribed by your doctor.")
        st.write("")

    if oldpeak > 2:
        st.write("📈 **For High Oldpeak:**")
        st.write("• Avoid heavy physical exertion.")
        st.write("• Consult a cardiologist.")
        st.write("• Follow medical advice for ECG evaluation.")
        st.write("")
    if exang == "Yes":
        st.write("🏃 **For Exercise-Induced Angina:**")
        st.write("• Avoid strenuous physical activities.")
        st.write("• Stop exercising immediately if chest pain occurs.")
        st.write("• Consult a cardiologist before starting an exercise program.")
        st.write("")

    if fbs == "Yes":
        st.write("🍬 **For High Fasting Blood Sugar:**")
        st.write("• Reduce sugar and refined carbohydrate intake.")
        st.write("• Follow a balanced diabetic-friendly diet.")
        st.write("• Exercise regularly as advised by your doctor.")
        st.write("• Monitor your blood sugar levels regularly.")
        st.write("")

    if age >= 60:
        st.write("👴 **For Age Above 60 Years:**")
        st.write("• Schedule regular heart health check-ups.")
        st.write("• Maintain a healthy lifestyle.")
        st.write("• Stay physically active according to your doctor's advice.")
        st.write("")

    if (
        chol <= 240 and
        trestbps <= 140 and
        oldpeak <= 2 and
        exang == "No" and
        fbs == "No" and
        age < 60
    ):
        st.success("✅ No specific precautions are required based on the entered values.")
        st.write("• Continue maintaining a healthy lifestyle.")
        st.write("• Eat a balanced diet.")
        st.write("• Exercise regularly.")
        st.write("• Get routine medical check-ups.")

    st.markdown("---")

    if prediction == 1:
        st.warning(
            "⚠️ This prediction is generated by a machine learning model and is not a medical diagnosis. "
            "Please consult a qualified healthcare professional for proper evaluation."
        )
    else:
        st.info(
            "💚 The prediction indicates a low likelihood of heart disease. "
            "Continue maintaining a healthy lifestyle and attend regular health check-ups."
        )
