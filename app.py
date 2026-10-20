import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("model.joblib")

st.set_page_config(page_title="Crop Recommendation")
st.title("🌱 Crop Recommendation System")

st.write(
    "Enter the soil and weather details "
    "to get a crop recommendation."
)

# Collect user inputs
N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
P = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
K = st.number_input("Potassium (K)", 0.0, 250.0, 50.0)

temperature = st.number_input(
    "Temperature (°C)", -10.0, 60.0, 25.0
)

humidity = st.number_input(
    "Humidity (%)", 0.0, 100.0, 70.0
)

ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

rainfall = st.number_input(
    "Rainfall (mm)", 0.0, 500.0, 100.0
)

# Make a prediction
if st.button("Recommend Crop"):

    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    crop = model.predict(input_data)[0]

    st.success(f"Recommended Crop: {crop}")

    st.info(
        "This is a model prediction, not a guarantee "
        "of agricultural suitability or yield."
    )
