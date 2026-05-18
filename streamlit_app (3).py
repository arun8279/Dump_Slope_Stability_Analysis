
import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load pretrained models
# -------------------------------
# Assume you have saved models as .pkl files
models = {
    "GBM": joblib.load("optimized_gradient_boosting_models.pkl")
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Dump Stability Prediction Web App")

st.sidebar.header("Input Features")

# Example input features (adjust to your dataset)
height = st.sidebar.slider("Dump Height (m)", 30, 70, 50)
slope_angle = st.sidebar.slider("Slope Angle (°)", 30, 55, 40)
unit_weight = st.sidebar.slider("Unit Weight (kN/m³)", 18.0, 20.5, 19.0)
cohesion = st.sidebar.slider("Cohesion (kN/m²)", 20, 40, 30)
friction_angle = st.sidebar.slider("Friction Angle (°)", 18, 25, 22)
ru = st.sidebar.slider("Pore Pressure Ratio (ru)", 0.0, 0.5, 0.2)

# Dropdown for model selection
model_choice = st.selectbox("Choose Model", list(models.keys()))

# Prediction button
if st.button("Predict Stability"):
    # Prepare input vector
    input_data = np.array([[height, slope_angle, unit_weight, cohesion, friction_angle, ru]])

    # Get model
    model = models[model_choice]

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"Predicted Factor of Safety (FoS): {prediction:.2f}")

    # Optional classification
    if prediction > 1.3:
        st.write("🟢 Stable")
    elif 1.0 <= prediction <= 1.3:
        st.write("🟡 Critically Stable")
    else:
        st.write("🔴 Unstable")
