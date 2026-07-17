import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# Load Saved Model
# =====================================

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="centered"
)

st.title("🫀 Heart Disease Prediction System")

st.markdown("""
This application predicts whether a patient is likely to have **Heart Disease**
using a trained **Random Forest Machine Learning model**.

Fill in the patient's medical details below and click **Predict**.
""")

st.markdown("---")

# =====================================
# Patient Details
# =====================================

st.header("🩺 Patient Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=45
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
    "Serum Cholesterol (mg/dL)",
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
    "Slope of Peak Exercise ST Segment",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0,1,2,3,4]
)

thal = st.selectbox(
    "Thalassemia",
    [0,1,2,3]
)

st.markdown("---")
# =====================================
# Prediction
# =====================================

if st.button("🔍 Predict"):

    # Create input dataframe with all training features
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # =====================================
    # Numerical Features
    # =====================================

    input_data["age"] = age
    input_data["trestbps"] = trestbps
    input_data["chol"] = chol
    input_data["thalach"] = thalach
    input_data["oldpeak"] = oldpeak

    # Scale numerical columns
    numerical_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )

    # =====================================
    # One Hot Encoding
    # =====================================

    # Gender
    gender = 1 if sex == "Male" else 0
    col = f"sex_{gender}"
    if col in input_data.columns:
        input_data[col] = 1

    # Chest Pain
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }

    col = f"cp_{cp_map[cp]}"
    if col in input_data.columns:
        input_data[col] = 1

    # FBS
    fbs_value = 1 if fbs == "Yes" else 0
    col = f"fbs_{fbs_value}"
    if col in input_data.columns:
        input_data[col] = 1

    # Rest ECG
    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }

    col = f"restecg_{restecg_map[restecg]}"
    if col in input_data.columns:
        input_data[col] = 1

    # Exercise Angina
    exang_value = 1 if exang == "Yes" else 0
    col = f"exang_{exang_value}"
    if col in input_data.columns:
        input_data[col] = 1

    # Slope
    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }

    col = f"slope_{slope_map[slope]}"
    if col in input_data.columns:
        input_data[col] = 1

    # CA
    col = f"ca_{ca}"
    if col in input_data.columns:
        input_data[col] = 1

    # Thal
    col = f"thal_{thal}"
    if col in input_data.columns:
        input_data[col] = 1

    # Final feature order
    input_data = input_data[feature_names]

    # =====================================
    # Prediction
    # =====================================
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 0:

        confidence = probability[0] * 100

        st.success("## 🫀 Prediction : No Heart Disease")

        if confidence >= 90:
            risk = "🟢 Very Low Risk"
        elif confidence >= 70:
            risk = "🟢 Low Risk"
        else:
            risk = "🟡 Medium Risk"

    else:

        confidence = probability[1] * 100

        st.error("## 🫀 Prediction : Heart Disease")

        if confidence >= 90:
            risk = "🔴 Very High Risk"
        elif confidence >= 70:
            risk = "🔴 High Risk"
        else:
            risk = "🟡 Medium Risk"

    #st.write(f"### Confidence : {confidence:.2f}%")
    st.write(f"### Risk Level : {risk}")

    st.markdown("---")
        # =====================================
    # Possible Risk Factors
    # =====================================

    st.subheader("🩺 Possible Risk Factors")

    risk_found = False

    if age >= 60:
        risk_found = True
        st.write("👴 **Age:** Higher age increases the possibility of heart disease.")

    if trestbps > 140:
        risk_found = True
        st.write(f"🩸 **High Blood Pressure:** {trestbps} mmHg")

    if chol > 240:
        risk_found = True
        st.write(f"🧈 **High Cholesterol:** {chol} mg/dL")

    if thalach < 100:
        risk_found = True
        st.write(f"🫀 **Low Maximum Heart Rate:** {thalach}")

    if oldpeak > 2:
        risk_found = True
        st.write(f"📈 **Oldpeak:** {oldpeak}")

    if exang == "Yes":
        risk_found = True
        st.write("🏃 **Exercise-Induced Angina:** Present")

    if fbs == "Yes":
        risk_found = True
        st.write("🍬 **High Fasting Blood Sugar**")

    if cp == "Asymptomatic":
        risk_found = True
        st.write("⚠️ **Asymptomatic Chest Pain:** Higher clinical risk.")

    if ca >= 2:
        risk_found = True
        st.write(f"🫀 **Major Vessels Affected:** {ca}")

    if thal == 3:
        risk_found = True
        st.write("🧬 **Abnormal Thalassemia Result**")

    if not risk_found:
        st.success("✅ No major risk factors detected from the entered values.")

    st.markdown("---")

    # =====================================
    # Precautions
    # =====================================

    st.subheader("💡 Recommended Precautions")

    precautions = []

    if chol > 240:
        precautions.extend([
            "Reduce oily and fried foods.",
            "Eat fruits, vegetables and whole grains.",
            "Exercise regularly."
        ])

    if trestbps > 140:
        precautions.extend([
            "Reduce salt intake.",
            "Monitor blood pressure regularly.",
            "Manage stress."
        ])

    if fbs == "Yes":
        precautions.extend([
            "Reduce sugar intake.",
            "Monitor blood glucose levels.",
            "Maintain a healthy diet."
        ])

    if oldpeak > 2:
        precautions.extend([
            "Consult a cardiologist.",
            "Avoid strenuous physical activity until medically evaluated."
        ])

    if exang == "Yes":
        precautions.extend([
            "Avoid heavy exercise.",
            "Stop activity immediately if chest pain occurs."
        ])

    if age >= 60:
        precautions.extend([
            "Schedule regular heart check-ups.",
            "Maintain an active lifestyle."
        ])

    if prediction == 1:
        precautions.extend([
            "Take medications only as prescribed by your doctor.",
            "Follow up with a cardiologist.",
            "Do not ignore symptoms such as chest pain or breathlessness."
        ])

    precautions = list(dict.fromkeys(precautions))

    if len(precautions) == 0:
        st.success("✅ No special precautions are required.")
        precautions = [
            "Maintain a balanced diet.",
            "Exercise regularly.",
            "Avoid smoking and alcohol.",
            "Drink adequate water.",
            "Sleep for 7–8 hours daily."
        ]

    for item in precautions:
        st.write("•", item)

    st.markdown("---")

    # =====================================
    # Lifestyle Recommendations
    # =====================================

    st.subheader("🥗 Healthy Lifestyle Tips")

    st.write("✅ Eat more fruits and vegetables.")
    st.write("✅ Exercise for at least 30 minutes daily.")
    st.write("✅ Maintain a healthy body weight.")
    st.write("✅ Reduce stress through meditation or yoga.")
    st.write("✅ Quit smoking and avoid alcohol.")
    st.write("✅ Get sufficient sleep every day.")
    st.write("✅ Drink enough water.")
    st.write("✅ Go for regular health check-ups.")

    st.markdown("---")

    # =====================================
    # Final Recommendation
    # =====================================

    if prediction == 1:

        st.warning("""
### ⚠️ Medical Disclaimer

The model predicts a **higher likelihood of heart disease**.

This prediction is generated using a Machine Learning model and **is not a medical diagnosis**.

Please consult a qualified healthcare professional for proper medical evaluation and treatment.
""")

    else:

        st.success("""
###  Good News

The model predicts a **low likelihood of heart disease**.

Continue maintaining a healthy lifestyle and undergo regular health check-ups.
""")

    st.markdown("---")

    st.caption(
        "Heart Disease Prediction System | Developed using Random Forest Classifier and Streamlit"
    )
    
