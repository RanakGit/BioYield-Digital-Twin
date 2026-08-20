import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

import streamlit as st
# Rest of your app.py code below...
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Any

# Page Configuration
st.set_page_config(page_title="BioYield Digital Twin", page_icon="🧫", layout="wide")

st.title("🧫 BioYield-Predict: Bioprocess Digital Twin")
st.markdown("Predict final fermentation product yield **(g/L)** in real-time based on initial media stoichiometry, biomass kinetics, and environmental stress logs.")

# Load Model
@st.cache_resource
def load_model() -> tuple[Any, list[str]]:
    model_path = Path(__file__).parent / "fermentation_model.pkl"
    features_path = Path(__file__).parent / "model_features.pkl"
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

try:
    model, feature_names = load_model()
    st.sidebar.success("Model Loaded Successfully!")
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(
        "Error loading model files. Please ensure 'fermentation_model.pkl' "
        "and 'model_features.pkl' are in the same directory."
    )
    st.stop()
    raise RuntimeError("Model files could not be loaded") from exc

# Sidebar Inputs
st.sidebar.header("🎛️ Bioreactor Input Parameters")

initial_S = st.sidebar.slider("Initial Substrate S0 (g/L)", 10.0, 60.0, 35.0, step=1.0)
initial_X = st.sidebar.slider("Initial Biomass X0 (g/L)", 0.05, 1.0, 0.25, step=0.05)
delta_X = st.sidebar.slider("Expected Biomass Growth Delta X (g/L)", 1.0, 20.0, 8.0, step=0.5)
min_ph = st.sidebar.slider("Minimum Batch pH Recorded", 4.0, 7.0, 5.5, step=0.1)
do_stress = st.sidebar.slider("DO Stress Duration (<20% DO) in Hours", 0.0, 24.0, 5.0, step=0.5)
y_px_kinetic = st.sidebar.slider("Specific Product Yield Potential Y_px (g/g)", 0.05, 0.5, 0.2, step=0.01)

# Derived Feature Calculation
s0_x0_ratio = initial_S / initial_X

# Input DataFrame
input_data = pd.DataFrame([[
    initial_S, initial_X, s0_x0_ratio, delta_X, min_ph, do_stress, y_px_kinetic
]], columns=feature_names)

# Display Input Summary
st.subheader("1. Batch Run Configuration")
col1, col2, col3, col4 = st.columns(4)
col1.metric("S0 / X0 Ratio", f"{s0_x0_ratio:.1f}")
col2.metric("Min pH", f"{min_ph:.2f}")
col3.metric("DO Stress Time", f"{do_stress:.1f} hrs")
col4.metric("Growth ΔX", f"{delta_X:.1f} g/L")

st.divider()

# Prediction
prediction = model.predict(input_data)[0]

st.subheader("2. Yield Prediction & Operational Insights")
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.metric(label="Predicted Final Product Yield", value=f"{prediction:.3f} g/L")
    
    if do_stress > 12.0:
        st.warning("⚠️ High Oxygen Stress Warning: Hypoxic duration exceeds 12 hours. Consider increasing agitation speed or pure O2 sparging.")
    elif min_ph < 5.0:
        st.error("🚨 Acidification Alert: Batch pH dropped below 5.0. Check alkali feed control.")
    else:
        st.success("✅ Fermentation Parameters within Optimal Operating Window.")

with res_col2:
    # Stoichiometric Efficiency Estimation
    est_conversion = (prediction / (initial_S * y_px_kinetic)) * 100 if initial_S > 0 else 0
    st.metric(label="Estimated Substrate Conversion Efficiency", value=f"{np.clip(est_conversion, 0, 100):.1f}%")