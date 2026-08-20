import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import odeint
import streamlit as st
# Sanitize and bound inputs defensively
initial_S0 = float(np.clip(initial_S0, 10.0, 60.0))
initial_X0 = float(np.clip(initial_X0, 0.05, 1.0))
min_pH = float(np.clip(min_pH, 4.0, 7.5))

# Set page configuration
st.set_page_config(
    page_title="BioYield-Predict | Bioprocess Digital Twin",
    page_icon="🧪",
    layout="wide",
)


# ---------------------------------------------------------
# 1. Model & Data Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("fermentation_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features


try:
    model, feature_names = load_assets()
    st.sidebar.success("Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model files: {e}")
    st.stop()


# ---------------------------------------------------------
# 2. Monod Kinetic ODE Function
# ---------------------------------------------------------
def fermentation_ode(y, t, mu_max, Ks, Y_xs, Y_px):
    X, S, P = y
    mu = mu_max * (S / (Ks + S)) if S > 0 else 0
    dXdt = mu * X
    dSdt = -(1 / Y_xs) * dXdt if S > 0 else 0
    dPdt = Y_px * dXdt if S > 0 else 0
    return [dXdt, dSdt, dPdt]



# ---------------------------------------------------------
# 3. Sidebar Inputs with Reset Button
# ---------------------------------------------------------
st.sidebar.header("⚙️ Bioreactor Input Parameters")

# Define default values in a dictionary
DEFAULTS = {
    "S0": 35.0,
    "X0": 0.25,
    "delta_X": 8.0,
    "min_pH": 5.5,
    "do_stress": 5.0,
    "Y_px": 0.20,
}


# Callback function to clear slider state back to defaults
def reset_to_defaults():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# Add the Reset Button at the top of the sidebar
st.sidebar.button(
    "🔄 Reset Inputs to Defaults",
    on_click=reset_to_defaults,
    use_container_width=True,
)

st.sidebar.divider()

# Sliders connected to session_state keys
initial_S0 = st.sidebar.slider(
    "Initial Substrate S0 (g/L)",
    10.0,
    60.0,
    DEFAULTS["S0"],
    0.5,
    key="S0",
)
initial_X0 = st.sidebar.slider(
    "Initial Biomass X0 (g/L)",
    0.05,
    1.0,
    DEFAULTS["X0"],
    0.05,
    key="X0",
)
delta_X = st.sidebar.slider(
    "Expected Biomass Growth Delta X (g/L)",
    1.0,
    20.0,
    DEFAULTS["delta_X"],
    0.5,
    key="delta_X",
)
min_pH = st.sidebar.slider(
    "Minimum Batch pH Recorded",
    4.0,
    7.5,
    DEFAULTS["min_pH"],
    0.1,
    key="min_pH",
)
do_stress_hours = st.sidebar.slider(
    "DO Stress Duration (<20% DO) in Hours",
    0.0,
    15.0,
    DEFAULTS["do_stress"],
    0.5,
    key="do_stress",
)
Y_px_kinetic = st.sidebar.slider(
    "Specific Product Yield Potential Y_px (g/g)",
    0.05,
    0.5,
    DEFAULTS["Y_px"],
    0.01,
    key="Y_px",
)
# Monod kinetic constants for ODE simulation
mu_max = 0.40  # 1/hr max growth rate
Ks = 1.0  # g/L affinity constant
Y_xs = 0.50  # g biomass / g substrate yield

# ---------------------------------------------------------
# 4. Feature Calculations & Prediction
# ---------------------------------------------------------
s0_x0_ratio = initial_S0 / initial_X0 if initial_X0 > 0 else 0

# Construct input DataFrame matching feature list order
input_data = pd.DataFrame(
    [[
        initial_S0,
        initial_X0,
        s0_x0_ratio,
        delta_X,
        min_pH,
        do_stress_hours,
        Y_px_kinetic,
    ]],
    columns=feature_names,
)

# Model yield prediction
predicted_yield = model.predict(input_data)[0]
substrate_efficiency = (
    (predicted_yield / initial_S0) * 100 if initial_S0 > 0 else 0
)


# ---------------------------------------------------------
# 5. UI Layout - Title & Key Metrics
# ---------------------------------------------------------
st.title("🧪 BioYield-Predict: Bioprocess Digital Twin")
st.markdown(
    "Predict final fermentation product yield (g/L) in real-time based on initial media stoichiometry, biomass kinetics, and environmental stress logs."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("S0 / X0 Ratio", f"{s0_x0_ratio:.1f}")
col2.metric("Min pH", f"{min_pH:.2f}")
col3.metric("DO Stress Time", f"{do_stress_hours:.1f} hrs")
col4.metric("Growth ΔX", f"{delta_X:.1f} g/L")

st.divider()

col_pred1, col_pred2 = st.columns(2)
col_pred1.metric("Predicted Final Product Yield", f"{predicted_yield:.3f} g/L")
col_pred2.metric(
    "Estimated Substrate Conversion Efficiency", f"{substrate_efficiency:.1f}%"
)

if min_pH >= 5.0 and do_stress_hours <= 8.0:
    st.success("✅ Fermentation Parameters within Optimal Operating Window.")
else:
    st.warning(
        "⚠️ Operating parameters outside ideal range: Increased stress detected."
    )

st.divider()

# ---------------------------------------------------------
# 6. FEATURE 1: Interactive Monod Kinematics Plot
# ---------------------------------------------------------
st.subheader("📈 Interactive Monod Kinetic Simulation (24 hrs)")

# Integrate ODEs for 24-hour batch profile
t = np.linspace(0, 24, 100)
ode_solution = odeint(
    fermentation_ode, [initial_X0, initial_S0, 0], t, args=(mu_max, Ks, Y_xs, Y_px_kinetic)
)
sim_df = pd.DataFrame(
    ode_solution, columns=["Biomass (X)", "Substrate (S)", "Product (P)"]
)
sim_df["Time (hr)"] = t

# Plotly interactive figure
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=sim_df["Time (hr)"],
        y=sim_df["Biomass (X)"],
        mode="lines",
        name="Biomass X (g/L)",
        line=dict(color="#2ca02c", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=sim_df["Time (hr)"],
        y=sim_df["Substrate (S)"],
        mode="lines",
        name="Substrate S (g/L)",
        line=dict(color="#d62728", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=sim_df["Time (hr)"],
        y=sim_df["Product (P)"],
        mode="lines",
        name="Product P (g/L)",
        line=dict(color="#1f77b4", width=3),
    )
)

fig.update_layout(
    xaxis_title="Time (hours)",
    yaxis_title="Concentration (g/L)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 7. FEATURE 2 & 3: Model Interpretability & Batch Export
# ---------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    with st.expander("🔍 Model Interpretability & Feature Importances", expanded=True):
        st.write(
            "Relative contribution of each bioreactor variable to the final yield model:"
        )

        # Matplotlib Horizontal Bar Chart
        importances = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values()

        fig_imp, ax = plt.subplots(figsize=(6, 3.5))
        importances.plot(kind="barh", color="#008080", ax=ax)
        ax.set_title("Gradient Boosting Feature Importance", fontsize=10)
        ax.set_xlabel("Importance Score", fontsize=9)
        plt.tight_layout()

        st.pyplot(fig_imp)

with col_right:
    with st.expander("📥 Export Batch Report", expanded=True):
        st.write(
            "Download full parameter logs, simulated 24-hr time-series curves, and predictions for audit compliance."
        )

        # Build summary report DataFrame
        summary_report = pd.DataFrame([{
            "Initial_S0_gL": initial_S0,
            "Initial_X0_gL": initial_X0,
            "S0_X0_Ratio": s0_x0_ratio,
            "Delta_X_gL": delta_X,
            "Min_pH": min_pH,
            "DO_Stress_Hours": do_stress_hours,
            "Y_px_kinetic": Y_px_kinetic,
            "Predicted_Yield_gL": round(predicted_yield, 3),
            "Substrate_Efficiency_Pct": round(substrate_efficiency, 1),
        }])

        # Generate CSV bytes
        csv_buffer = io.StringIO()
        summary_report.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📄 Download Summary Report (CSV)",
            data=csv_data,
            file_name="bioyield_batch_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Generate Full Simulation Time-Series CSV bytes
        sim_csv_buffer = io.StringIO()
        sim_df.to_csv(sim_csv_buffer, index=False)
        sim_csv_data = sim_csv_buffer.getvalue()

        st.download_button(
            label="📈 Download 24-hr Kinetic Profiles (CSV)",
            data=sim_csv_data,
            file_name="bioyield_monod_kinetic_series.csv",
            mime="text/csv",
            use_container_width=True,
        )
