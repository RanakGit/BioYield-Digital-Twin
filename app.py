import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import odeint
import streamlit as st
from supabase import create_client, Client

# Initialize Supabase Client
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# Initialize session state for auth
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------------------------------------------------
# AUTHENTICATION SCREEN (Shown if user is not logged in)
# ---------------------------------------------------------
def render_auth_ui():
    st.markdown("<h2 style='text-align: center;'>🧪 BioYield-Predict</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8E8EA0;'>Sign in to access your digital twin workstation</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="chat-card">', unsafe_allow_html=True)
        
        auth_mode = st.radio("Choose Mode", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
        
        email = st.text_input("Email Address", placeholder="name@company.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        
        if auth_mode == "Login":
            if st.button("Sign In with Email", use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = res.user
                    st.success("Signed in successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
        else:
            if st.button("Create Account", use_container_width=True):
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Account created! Please check your email to confirm.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")
                    
        st.divider()
        
        # Google OAuth Button
        if st.button("🌐 Continue with Google", use_container_width=True):
            try:
                res = supabase.auth.sign_in_with_oauth({"provider": "google"})
                st.info(f"Redirecting to Google authentication...")
            except Exception as e:
                st.error(f"OAuth error: {e}")
                
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN APP GATEWAY
# ---------------------------------------------------------
if st.session_state.user is None:
    render_auth_ui()
    st.stop()  # Stops execution here so unauthenticated users can't see the app below

# --- REST OF YOUR APP.PY CODE RUNS HERE FOR LOGGED-IN USERS ONLY ---


# Set page configuration
st.set_page_config(
    page_title="BioYield-Predict | Bioprocess Digital Twin",
    page_icon="🧪",
    layout="wide",
)

# ---------------------------------------------------------
# ChatGPT Dark Theme Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Backgrounds */
    .stApp {
        background-color: #202123;
        color: #ECECF1;
        font-family: 'Söhne', 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #343541 !important;
        border-right: 1px solid #4d4d4f;
    }
    section[data-testid="stSidebar"] * {
        color: #ECECF1 !important;
    }
    
    /* Metric Cards (ChatGPT Container Style) */
    div[data-testid="stMetric"] {
        background-color: #2A2B32;
        border: 1px solid #3E3F4B;
        padding: 16px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] p {
        color: #8E8EA0 !important;
        font-size: 0.85rem !important;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] div {
        color: #10A37F !important; /* ChatGPT Accent Green */
        font-weight: 600;
    }
    
    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #8E8EA0 !important;
        border-bottom: 2px solid transparent !important;
        font-size: 0.95rem;
        padding: 10px 16px;
    }
    button[aria-selected="true"] {
        color: #ECECF1 !important;
        border-bottom: 2px solid #10A37F !important;
        font-weight: 600;
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        color: #ECECF1 !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    .stCaption {
        color: #8E8EA0 !important;
    }
    
    /* Custom Prompt / Callout Boxes */
    .chat-card {
        background-color: #343541;
        border: 1px solid #4d4d4f;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    
    /* Buttons (ChatGPT Accent Button) */
    .stDownloadButton > button {
        background-color: #10A37F !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: background-color 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #1A7F64 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 1. Model & Data Loaders (CACHED FOR MULTI-USER PERFORMANCE)
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
# 2. Monod Kinetic ODE Function & Caching Layer
# ---------------------------------------------------------
def fermentation_ode(y, t, mu_max, Ks, Y_xs, Y_px):
    X, S, P = y
    mu = mu_max * (S / (Ks + S)) if S > 0 else 0
    dXdt = mu * X
    dSdt = -(1 / Y_xs) * dXdt if S > 0 else 0
    dPdt = Y_px * dXdt if S > 0 else 0
    return [dXdt, dSdt, dPdt]


# Cache ODE integration outputs (up to 100 parameter combinations for 1 hour)
@st.cache_data(max_entries=100, ttl=3600)
def run_cached_ode(X0, S0, Y_px, mu_max, Ks, Y_xs):
    t = np.linspace(0, 24, 100)
    solution = odeint(
        fermentation_ode, [X0, S0, 0], t, args=(mu_max, Ks, Y_xs, Y_px)
    )
    return t, solution


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
# Add input bounds sanitization right here!
initial_S0 = float(np.clip(initial_S0, 10.0, 60.0))
initial_X0 = float(np.clip(initial_X0, 0.05, 1.0))
delta_X = float(np.clip(delta_X, 1.0, 20.0))
min_pH = float(np.clip(min_pH, 4.0, 7.5))
do_stress_hours = float(np.clip(do_stress_hours, 0.0, 15.0))
Y_px_kinetic = float(np.clip(Y_px_kinetic, 0.05, 0.5))

# Derived features & model prediction using the sanitized inputs
s0_x0_ratio = initial_S0 / initial_X0 if initial_X0 > 0 else 0

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
# ---------------------------------------------------------
# 6. FEATURE 1: Interactive Monod Kinematics Plot
# ---------------------------------------------------------
st.subheader("📈 Interactive Monod Kinetic Simulation (24 hrs)")

# Call the cached function instead of recalculating odeint every frame
t, ode_solution = run_cached_ode(
    initial_X0, initial_S0, Y_px_kinetic, mu_max, Ks, Y_xs
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
