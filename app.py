"""
Heart Disease Prediction - Interactive Machine Learning Web App
-----------------------------------------------------------------
COMSATS University Islamabad, Lahore Campus
Course: Machine Learning Fundamentals (AIC354)
Supervised by: Dr. Rao Muhammad Adeel Nawab

This Streamlit web app allows students, instructors, and healthcare testers
to interact with the trained Support Vector Classifier (SVC) built in
Assignment 2 following the 4-attribute constraint from the UCI Heart Disease dataset.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor | COMSATS AIC354",
    page_icon="❤️",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Load Trained Model and Build Deterministic Label Encoders
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the Jupyter notebook.
    Fitting on the same sorted category lists guarantees identical numerical
    encodings without requiring separate pickle objects.
    """
    encoders = {}
    encoders["Gender"] = LabelEncoder().fit(["Male", "Female"])
    encoders["ChestPain"] = LabelEncoder().fit([
        "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"
    ])
    encoders["ExerciseAngina"] = LabelEncoder().fit(["Yes", "No"])
    encoders["Thalassemia"] = LabelEncoder().fit([
        "Normal", "Fixed Defect", "Reversible Defect"
    ])
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# User Interface
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    **COMSATS University Islamabad, Lahore Campus** — *Machine Learning Fundamentals (AIC354)*  
    **Instructor:** Dr. Rao Muhammad Adeel Nawab | **Assignment 2: End-to-End ML System**

    This application evaluates cardiovascular disease risk using a **Support Vector Classifier (SVC)**
    trained on the **UCI Heart Disease Dataset (Cleveland Clinic Foundation)** using **4 key clinical features**.
    
    *Model Test Accuracy:* **80.0%** on held-out benchmark evaluation.
    """
)

st.divider()

st.subheader("📋 Enter Patient Clinical Profile")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "1. Gender",
        options=["Male", "Female"],
        index=0,
        help="Demographic patient sex"
    )
    chest_pain = st.selectbox(
        "2. Chest Pain Type",
        options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
        index=3,
        help="Symptom classification of chest discomfort"
    )

with col2:
    exercise_angina = st.selectbox(
        "3. Exercise-Induced Angina",
        options=["No", "Yes"],
        index=1,
        help="Presence of angina provoked during exercise or physical exertion"
    )
    thalassemia = st.selectbox(
        "4. Thalassemia Perfusion Test",
        options=["Normal", "Fixed Defect", "Reversible Defect"],
        index=2,
        help="Thallium-201 nuclear myocardial perfusion imaging result"
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
    # Form input dataframe
    user_input = pd.DataFrame({
        "Gender": [gender],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "Thalassemia": [thalassemia],
    })

    # Encode exactly as training pipeline
    encoded = pd.DataFrame({
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
        "Thalassemia": encoders["Thalassemia"].transform(user_input["Thalassemia"]),
    })

    # Model inference
    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### 🚨 Prediction: HEART DISEASE DETECTED")
        st.warning(
            "**Clinical Assessment:** The Support Vector Classifier indicates a elevated risk of coronary heart disease "
            "based on the entered clinical indicators (e.g., provoked angina and perfusion defect patterns). "
            "Immediate formal medical consultation and diagnostic angiogram/stress testing is recommended."
        )
    else:
        st.success("### ✅ Prediction: NO HEART DISEASE DETECTED")
        st.info(
            "**Clinical Assessment:** The Support Vector Classifier does not detect significant coronary narrowing "
            "under this 4-attribute profile. Maintaining routine wellness monitoring and healthy lifestyle is advised."
        )

    with st.expander("🔬 Technical Details: Encoded Feature Vector"):
        st.write("Processed vector passed to Support Vector Classifier:")
        st.dataframe(encoded, hide_index=True)
        st.caption(
            f"Numeric Encoding Codes: "
            f"Gender={encoded['Gender'].iloc[0]}, "
            f"ChestPain={encoded['ChestPain'].iloc[0]}, "
            f"ExerciseAngina={encoded['ExerciseAngina'].iloc[0]}, "
            f"Thalassemia={encoded['Thalassemia'].iloc[0]}"
        )

st.divider()

# Informational Footer
st.caption(
    "📌 **Academic & Teaching Demo:** Built for Assignment 2 of AIC354 Machine Learning Fundamentals. "
    "Dataset: UCI Cleveland Heart Disease dataset. This model is for academic demonstration only and "
    "is not intended as a substitute for professional clinical medical advice."
)
