import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Wearable Activity Recognition", layout="centered")

st.title("🏃 Wearable Activity Recognition")
st.write("Upload wearable sensor data (CSV) or enter readings to predict activity.")

# 1. Load the Model
MODEL_PATH = "wearable_activity_model.keras"

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_trained_model()

if model is None:
    st.error(f"Model file '{MODEL_PATH}' was not found in the repository. Please ensure the .keras file is committed and pushed.")
else:
    st.success("Model loaded successfully!")

    # Update these class names to match your dataset classes
    CLASS_NAMES = ["Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs"]

    # 2. File Upload UI
    uploaded_file = st.file_uploader("Choose a sensor reading CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded Data:")
            st.dataframe(df.head())

            if st.button("Classify Activity"):
                # Preprocessing placeholder: convert dataframe to model input shape
                # Adjust (1, -1) / reshape to match your model's expected input dimension
                sensor_data = df.select_dtypes(include=[np.number]).to_numpy()
                
                # Reshape to batch format expected by CNN
                # e.g., if model expects (batch_size, timesteps, features):
                input_tensor = np.expand_dims(sensor_data, axis=0)

                predictions = model.predict(input_tensor)[0]
                top_idx = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class {top_idx}"

                st.subheader(f"Prediction: **{predicted_class}**")
                st.write(f"Confidence: `{predictions[top_idx] * 100:.2f}%`")
                
                st.write("#### Probability Distribution:")
                chart_data = pd.DataFrame({
                    "Activity": CLASS_NAMES[:len(predictions)],
                    "Probability": predictions
                })
                st.bar_chart(chart_data.set_index("Activity"))

        except Exception as e:
            st.error(f"Error processing file: {e}")
