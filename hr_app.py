import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="HR Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# Load Model and Scaler
try:
    with open('hr.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('schr.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# App Title
st.title("💼 HR Salary Prediction System")
st.markdown("Predict employee salary using Machine Learning 🚀")

# Sidebar
st.sidebar.header("Input Employee Details")

# User Inputs
age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

experience = st.sidebar.number_input(
    "Experience (Months)",
    min_value=0,
    max_value=600,
    value=12
)

education = st.sidebar.selectbox(
    "Education Level",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "High School",
        1: "Bachelor's Degree",
        2: "Master's Degree",
        3: "PhD"
    }[x]
)

# Prediction Button
if st.button("Predict Salary"):

    # Create input array
    input_data = np.array([[age, experience, education]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict salary
    prediction = model.predict(input_scaled)

    # Display result
    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")

    st.balloons()

# Information Section
st.markdown("---")
st.subheader("📌 Education Level Mapping")
st.write("0 → High School")
st.write("1 → Bachelor's Degree")
st.write("2 → Master's Degree")
st.write("3 → PhD")

st.markdown("---")
st.caption("Built with Streamlit and Machine Learning 🤖")