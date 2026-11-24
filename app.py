
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained models
crop_model = joblib.load("crop_recommendation_model.pkl")

soil_model = tf.keras.models.load_model(
    "soil_classification_model.keras"
)

# Soil class names
soil_classes = [
    "Alluvial soil",
    "Black Soil",
    "Clay soil",
    "Red soil"
]

st.title("Smart Crop Recommendation and Soil Identification System")

st.write(
    "This system recommends suitable crops "
    "and identifies soil types using Machine Learning and Deep Learning."
)

# ---------------- CROP RECOMMENDATION ----------------

st.header("1. Crop Recommendation")

st.write("Enter the soil and environmental details:")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", value=90.0)
    P = st.number_input("Phosphorus (P)", value=42.0)
    K = st.number_input("Potassium (K)", value=43.0)
    temperature = st.number_input("Temperature (°C)", value=20.8)

with col2:
    humidity = st.number_input("Humidity (%)", value=82.0)
    ph = st.number_input("Soil pH", value=6.5)
    rainfall = st.number_input("Rainfall (mm)", value=202.9)

if st.button("Recommend Crop"):

    input_data = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=[
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ])

    prediction = crop_model.predict(input_data)[0]
    st.session_state["recommended_crop"] = str(prediction).title()

    st.success(f"Recommended Crop: {prediction.title()}")

    if hasattr(crop_model, "predict_proba"):
        probabilities = crop_model.predict_proba(input_data)[0]
        confidence = np.max(probabilities) * 100

        st.write(f"Model confidence: {confidence:.2f}%")

# --------------------------------
# 2. SOIL TYPE IDENTIFICATION
# --------------------------------

st.header("2. Soil Type Identification")

st.write("Upload a soil image to identify its type.")

uploaded_file = st.file_uploader(
    "Choose a soil image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Soil Image", width=300)

    # Resize image
    image = image.resize((224, 224))

    # Convert image into array
    img_array = np.array(image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess image
    img_array = preprocess_input(img_array)

    # Predict soil type
    predictions = soil_model.predict(img_array)

    predicted_index = np.argmax(predictions[0])

    soil_name = soil_classes[predicted_index]
    st.session_state["identified_soil"] = soil_name

    confidence = np.max(predictions[0]) * 100

    st.success(f"Identified Soil Type: {soil_name}")

# --------------------------------
# 3. CROP-SOIL COMPATIBILITY
# --------------------------------

st.header("3. Crop and Soil Compatibility")

compatible_soils = {
    "Rice": ["Alluvial soil", "Clay soil"],
    "Cotton": ["Black Soil"],
    "Wheat": ["Alluvial soil", "Black Soil"],
    "Maize": ["Alluvial soil", "Red soil"],
    "Sugarcane": ["Alluvial soil", "Black Soil"],
    "Chickpea": ["Black Soil", "Alluvial soil"],
    "Coffee": ["Red soil"],
    "Pomegranate": ["Red soil", "Alluvial soil"],
    "Banana": ["Alluvial soil"],
    "Mango": ["Alluvial soil", "Red soil"]
}

crop = st.session_state.get("recommended_crop")
soil = st.session_state.get("identified_soil")

if crop and soil:

    st.write(f"Recommended Crop: {crop}")
    st.write(f"Identified Soil: {soil}")

    suitable = compatible_soils.get(crop)

    if suitable is None:
        st.info("Compatibility information is not available for this crop yet.")

    elif soil in suitable:
        st.success("This crop and soil combination is suitable according to our rule-based advisory.")

    else:
        st.warning("This crop and soil combination may not be suitable according to our rule-based advisory.")

else:
    st.info("Please generate both crop and soil predictions to view compatibility.")

    
