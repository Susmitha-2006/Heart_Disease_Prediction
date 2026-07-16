# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below.")

age = st.number_input("Age",1,120,50)
sex = st.selectbox("Sex",["Male","Female"])
cp = st.selectbox("Chest Pain Type",["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure",80,250,120)
chol = st.number_input("Cholesterol",100,600,200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",["No","Yes"])
restecg = st.selectbox("Resting ECG",["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"])
thalach = st.number_input("Maximum Heart Rate",50,250,150)
exang = st.selectbox("Exercise Induced Angina",["No","Yes"])
oldpeak = st.number_input("Oldpeak",0.0,10.0,1.0,0.1)
slope = st.selectbox("Slope",["Upsloping","Flat","Downsloping"])
ca = st.selectbox("Number of Major Vessels",[0,1,2,3,4])
thal = st.selectbox("Thalassemia",[0,1,2,3])

if st.button("Predict"):
    input_data = pd.DataFrame(np.zeros((1,len(feature_names))),columns=feature_names)
    input_data["age"]=age
    input_data["trestbps"]=trestbps
    input_data["chol"]=chol
    input_data["thalach"]=thalach
    input_data["oldpeak"]=oldpeak

    input_data[["age","trestbps","chol","thalach","oldpeak"]] = scaler.transform(
        input_data[["age","trestbps","chol","thalach","oldpeak"]]
    )

    input_data[f"sex_{1 if sex=='Male' else 0}"]=1
    input_data[f"cp_{{'Typical Angina':0,'Atypical Angina':1,'Non-anginal Pain':2,'Asymptomatic':3}[cp]}"]=1
    input_data[f"fbs_{1 if fbs=='Yes' else 0}"]=1
    input_data[f"restecg_{{'Normal':0,'ST-T Wave Abnormality':1,'Left Ventricular Hypertrophy':2}[restecg]}"]=1
    input_data[f"exang_{1 if exang=='Yes' else 0}"]=1
    input_data[f"slope_{{'Upsloping':0,'Flat':1,'Downsloping':2}[slope]}"]=1
    input_data[f"ca_{ca}"]=1
    input_data[f"thal_{thal}"]=1

    prediction=model.predict(input_data)[0]
    probs=model.predict_proba(input_data)[0]

    if prediction==0:
        confidence=probs[0]*100
        disease="No Heart Disease"
        risk="🟢 Low Risk" if confidence>=80 else "🟡 Medium Risk"
        st.success(f"💚 Prediction: {disease}")
    else:
        confidence=probs[1]*100
        disease="Heart Disease"
        risk="🟡 Medium Risk" if confidence<60 else "🔴 High Risk"
        st.error(f"❤️ Prediction: {disease}")

    st.write(f"### Confidence: {confidence:.2f}%")
    st.write(f"### Risk Level: {risk}")

    st.subheader("Possible Risk Factors")
    found=False
    if age>60: st.write("• Age above 60 years."); found=True
    if trestbps>140: st.write("• High resting blood pressure."); found=True
    if chol>240: st.write("• High cholesterol."); found=True
    if oldpeak>2: st.write("• Elevated Oldpeak."); found=True
    if exang=="Yes": st.write("• Exercise-induced angina."); found=True
    if not found: st.write("• No major risk factors identified from the entered values.")

    st.subheader("General Precautions")
    for t in [
        "Maintain a balanced diet.",
        "Exercise regularly after medical advice.",
        "Avoid smoking and limit alcohol.",
        "Monitor blood pressure and cholesterol.",
        "Have regular health check-ups."
    ]:
        st.write("• "+t)
