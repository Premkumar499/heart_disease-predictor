import streamlit as st
import pickle
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Predictor")
st.markdown("Fill in the patient details below to predict the likelihood of heart disease.")

try:
    data = load_model()
    model = data["model"]
    scaler = data["scaler"]
except FileNotFoundError:
    st.error("Model not found. Please run `python train.py` first.")
    st.stop()

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                          format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina",
                                                  2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                               format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0,
                                  value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                             format_func=lambda x: {0: "Downsloping", 1: "Flat", 2: "Upsloping"}[x])
        ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                            format_func=lambda x: {0: "Normal", 1: "Fixed Defect",
                                                    2: "Normal", 3: "Reversible Defect"}[x])

    submitted = st.form_submit_button("Predict", width="stretch")

if submitted:
    features = data["features"]
    input_df = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=features
    )
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.error("⚠️ High Risk: Heart Disease Detected")
    else:
        st.success("✅ Low Risk: No Heart Disease Detected")

    col1, col2 = st.columns(2)
    col1.metric("No Disease Probability", f"{probability[0]:.1%}")
    col2.metric("Heart Disease Probability", f"{probability[1]:.1%}")

    # Show input summary
    with st.expander("Input Summary"):
        summary = pd.DataFrame({
            "Feature": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
                        "Fasting BS", "Rest ECG", "Max HR", "Exercise Angina",
                        "ST Depression", "ST Slope", "Major Vessels", "Thal"],
            "Value": [str(age), "Male" if sex == 1 else "Female", str(cp), str(trestbps), str(chol),
                      "Yes" if fbs == 1 else "No", str(restecg), str(thalach),
                      "Yes" if exang == 1 else "No", str(oldpeak), str(slope), str(ca), str(thal)]
        })
        st.dataframe(summary, width="stretch", hide_index=True)

st.divider()
st.caption("Model: Random Forest | Dataset: UCI Heart Disease (303 samples)")
