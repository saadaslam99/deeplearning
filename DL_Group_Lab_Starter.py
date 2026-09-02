import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Wearable Activity Recognition", layout="centered")

st.title("🏃 Wearable Activity Recognition")
st.write("Classify wearable sensor time-series data using a 1D Convolutional Neural Network.")

CLASS_NAMES = ['Stationary', 'Walking', 'Running']
MODEL_PATH = "wearable_activity_model.keras"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the model is pushed to the repository.")
else:
    st.success("Model loaded successfully!")

    uploaded_file = st.file_uploader("Upload a sensor CSV (e.g. sample_running.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())

            # Extract 100 time-step features (t000 to t099 or numerical columns)
            time_cols = [c for c in df.columns if c.startswith('t')]
            if len(time_cols) == 100:
                features = df[time_cols].iloc[0].to_numpy()
            else:
                numeric_df = df.select_dtypes(include=[np.number])
                features = numeric_df.to_numpy().flatten()[:100]

            if len(features) < 100:
                st.error(f"Expected at least 100 time-series data points, but found {len(features)}.")
            else:
                # Plot the sensor signal
                st.write("### Sensor Signal (100 timesteps)")
                st.line_chart(features)

                # Prepare input: (1, 100, 1)
                input_tensor = features.reshape(1, 100, 1).astype(np.float32)

                # Run inference
                predictions = model.predict(input_tensor)[0]
                top_idx = int(np.argmax(predictions))
                predicted_activity = CLASS_NAMES[top_idx]
                confidence = predictions[top_idx] * 100

                st.subheader(f"Prediction: **{predicted_activity}** ({confidence:.1f}% confidence)")

                # Probability bar chart
                prob_df = pd.DataFrame({
                    "Activity": CLASS_NAMES,
                    "Probability": predictions
                })
                st.bar_chart(prob_df.set_index("Activity"))

        except Exception as e:
            st.error(f"Error reading file: {e}")
