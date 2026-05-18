
import streamlit as st
import joblib
import numpy as np
import pandas as pd # Added pandas import

# Define the model path
MODEL_PATH = '/content/optimized_gradient_boosting_models.pkl' # Use the MODEL_PATH defined in Colab

# Load the pre-trained models (which is a dictionary of models)
@st.cache_resource
def load_models(path):
    try:
        models = joblib.load(path)
        return models
    except Exception as e:
        st.error(f"Error loading models: {{e}}") # Escaped e
        return None

loaded_models = load_models(MODEL_PATH)

st.set_page_config(page_title="Slope Stability Predictor", page_icon="⛰️", layout="centered")

st.title("⛰️ Slope Stability Predictor")
st.markdown("Enter the geological parameters below to predict the Factor of Safety (FoS).")

if loaded_models is not None and isinstance(loaded_models, dict):
    # Dropdown for model selection
    model_names = list(loaded_models.keys())
    selected_model_name = st.selectbox(
        "Select Model:",
        options=model_names,
        index=0, # Default to the first model
        help="Choose the machine learning model for prediction."
    )
    model = loaded_models[selected_model_name]

    with st.form("prediction_form"):
        st.header("Input Parameters")
        col1, col2 = st.columns(2)
        with col1:
            cohesion = st.slider("Cohesion (c in kPa)", min_value=0.0, max_value=100.0, value=15.0, step=0.1, format="%.1f")
            friction_angle = st.slider("Friction Angle (φ in degrees)", min_value=0.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
            slope_angle = st.slider("Slope Angle (β in degrees)", min_value=0.0, max_value=90.0, value=38.0, step=0.1, format="%.1f")
        with col2:
            height = st.slider("Dump Height (H in meters)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.1f")
            unit_weight = st.slider("Unit Weight (γ in kN/m³)", min_value=10.0, max_value=25.0, value=18.0, step=0.1, format="%.1f")
            water_level_ratio = st.slider("Water Level Ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.2f")

        submitted = st.form_submit_button("Run ML Prediction")

        if submitted:
            # Features expected by the model (6 features)
            features = np.array([[cohesion, friction_angle, slope_angle, height, unit_weight, water_level_ratio]])

            try:
                raw_prediction = model.predict(features)
                fos = round(float(raw_prediction[0]), 3)

                st.subheader("Prediction Result")
                st.metric(label=f"Factor of Safety (FoS) using {selected_model_name}", value=f"{{fos:.3f}}") # Escaped selected_model_name and fos

                # Prepare data for download (escaped dictionary braces)
                data_to_download = {
                    'Selected Model': [selected_model_name],
                    'Cohesion (kPa)': [cohesion],
                    'Friction Angle (degrees)': [friction_angle],
                    'Slope Angle (degrees)': [slope_angle],
                    'Dump Height (m)': [height],
                    'Unit Weight (kN/m³)': [unit_weight],
                    'Water Level Ratio': [water_level_ratio],
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
                st.error(f"An error occurred during prediction: {{e}}") # Escaped e
else:
    st.warning("Models could not be loaded. Please check the `MODEL_PATH` and ensure the file exists.")

