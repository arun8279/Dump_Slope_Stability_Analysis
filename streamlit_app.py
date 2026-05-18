
import streamlit as st
import joblib
import numpy as np
import pandas as pd # Added pandas import

# Define the model path
MODEL_PATH = '/content/optimized_gradient_boosting_models.pkl' # Use the MODEL_PATH defined in Colab

# Load the pre-trained model
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

st.set_page_config(page_title="Slope Stability Predictor", page_icon="⛰️", layout="centered")

st.title("⛰️ Slope Stability Predictor")
st.markdown("Enter the geological parameters below to predict the Factor of Safety (FoS).")

if model is not None:
    with st.form("prediction_form"):
        st.header("Input Parameters")
        col1, col2 = st.columns(2)
        with col1:
            cohesion = st.number_input("Cohesion (c in kPa)", min_value=0.0, value=15.0, step=0.1, format="%.1f")
            friction_angle = st.number_input("Friction Angle (φ in degrees)", min_value=0.0, value=25.0, step=0.1, format="%.1f")
        with col2:
            slope_angle = st.number_input("Slope Angle (β in degrees)", min_value=0.0, value=38.0, step=0.1, format="%.1f")
            height = st.number_input("Dump Height (H in meters)", min_value=0.0, value=20.0, step=0.1, format="%.1f")

        submitted = st.form_submit_button("Run ML Prediction")

        if submitted:
            features = np.array([[cohesion, friction_angle, slope_angle, height]])

            try:
                raw_prediction = model.predict(features)
                fos = round(float(raw_prediction[0]), 3)

                st.subheader("Prediction Result")
                st.metric(label="Factor of Safety (FoS)", value=f"{fos:.3f}")

                # Prepare data for download
                data_to_download = {
                    'Cohesion (kPa)': [cohesion],
                    'Friction Angle (degrees)': [friction_angle],
                    'Slope Angle (degrees)': [slope_angle],
                    'Dump Height (m)': [height],
                    'Factor of Safety': [fos]
                }
                df_results = pd.DataFrame(data_to_download)
                csv_results = df_results.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_results,
                    file_name='slope_stability_prediction.csv',
                    mime='text/csv',
                    help="Click to download the input parameters and the predicted Factor of Safety."
                )

                if fos >= 1.2:
                    st.success("STATUS: STABLE (SAFE) ✅")
                    st.balloons()
                else:
                    st.error("STATUS: UNSTABLE (CRITICAL) ⚠️")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Model could not be loaded. Please check the `MODEL_PATH` and ensure the file exists.")

