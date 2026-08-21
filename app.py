import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import odeint
import streamlit as st
from supabase import create_client, Client

# Set page configuration FIRST
st.set_page_config(
    page_title="BioYield-Predict | Bioprocess Digital Twin",
    page_icon="🧪",
    layout="wide",
)

# ---------------------------------------------------------
# ChatGPT Dark Theme & Enhanced UI Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #202123; color: #ECECF1; font-family: 'Söhne', 'Segoe UI', Roboto, sans-serif; }
    section[data-testid="stSidebar"] { background-color: #343541 !important; border-right: 1px solid #4d4d4f; }
    section[data-testid="stSidebar"] * { color: #ECECF1 !important; }
    
    /* Enhanced Metric Cards */
    div[data-testid="stMetric"] { background-color: #2A2B32; border: 1px solid #3E3F4B; padding: 16px 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); }
    div[data-testid="stMetricLabel"] p { color: #8E8EA0 !important; font-size: 0.85rem !important; font-weight: 500; }
    div[data-testid="stMetricValue"] div { color: #10A37F !important; font-weight: 600; }
    
    /* Tabs & Callout Boxes */
    button[data-baseweb="tab"] { background-color: transparent !important; color: #8E8EA0 !important; border-bottom: 2px solid transparent !important; font-size: 0.95rem; padding: 10px 16px; }
    button[aria-selected="true"] { color: #ECECF1 !important; border-bottom: 2px solid #10A37F !important; font-weight: 600; }
    h1, h2, h3 { color: #ECECF1 !important; font-weight: 600; letter-spacing: -0.02em; }
    .chat-card { background-color: #343541; border: 1px solid #4d4d4f; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }
    
    /* Download & Action Buttons */
    .stDownloadButton > button, div.stButton > button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helper Functions & Supabase Database Operations
# ---------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    # Safely extract URL and KEY from top-level or nested [supabase] secrets
    url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("SUPABASE_KEY")

    if not url or not key:
        st.error("Missing SUPABASE_URL or SUPABASE_KEY in Streamlit Secrets!")
        st.stop()

    # Create client cleanly without invalid options dictionary
    return create_client(url, key)

# Execute client instantiation
supabase = init_supabase()

def save_simulation_run(user_id: str, s0: float, x0: float, ph: float, yield_val: float):
    try:
        data = {
            "user_id": user_id,
            "initial_s0": s0,
            "initial_x0": x0,
            "min_ph": ph,
            "predicted_yield": yield_val
        }
        supabase.table("simulation_runs").insert(data).execute()
        st.toast("Simulation run saved to history!", icon="💾")
    except Exception as e:
        st.error(f"Failed to save simulation to database: {e}")

def fetch_user_history(user_id: str) -> pd.DataFrame:
    try:
        response = supabase.table("simulation_runs").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        if response.data:
            return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
    return pd.DataFrame()

def reset_parameters():
    st.session_state.initial_S0 = 30.0
    st.session_state.initial_X0 = 0.20
    st.session_state.min_pH = 6.8
    st.session_state.sim_time = 48
    st.session_state.Y_px_kinetic = 0.20
    st.session_state.mu_max = 0.40
    st.session_state.Ks = 1.0
    st.session_state.Y_xs = 0.50

# ---------------------------------------------------------
# Session State & Robust OAuth Callback Handler
# ---------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if "baseline_yield" not in st.session_state:
    st.session_state.baseline_yield = None

# 1. Check for 'code' query parameter returned from Google OAuth
if st.session_state.user is None and "code" in st.query_params:
    auth_code = st.query_params["code"]
    try:
        # Pass auth_code directly in dictionary format
        res = supabase.auth.exchange_code_for_session({"auth_code": auth_code})
        if res and hasattr(res, 'user') and res.user:
            st.session_state.user = res.user
        elif res and hasattr(res, 'session') and res.session:
            st.session_state.user = res.session.user
            
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        # If PKCE fails due to missing verifier in Streamlit, clear query params and fallback
        st.query_params.clear()

# 2. Fallback session check (detects active cookie or persisted session)
if st.session_state.user is None:
    try:
        session_res = supabase.auth.get_session()
        if session_res and hasattr(session_res, 'user') and session_res.user:
            st.session_state.user = session_res.user
        elif session_res and hasattr(session_res, 'session') and session_res.session:
            st.session_state.user = session_res.session.user
    except Exception:
        pass

# ---------------------------------------------------------
# AUTHENTICATION SCREEN
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
                    st.success("Account created! Check your email to confirm.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")
                    
        st.divider()
        
        redirect_target = st.secrets.get("REDIRECT_URL") or st.secrets.get("supabase", {}).get("REDIRECT_URL", "http://localhost:8501")

        try:
            auth_response = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {"redirect_to": redirect_target}
            })
            
            if hasattr(auth_response, 'url') and auth_response.url:
                st.link_button("🌐 Continue with Google", auth_response.url, use_container_width=True)
            else:
                st.error("Could not fetch Google login URL. Check Supabase settings.")
        except Exception as e:
            st.error(f"OAuth initialization error: {e}")
                
        st.markdown('</div>', unsafe_allow_html=True)

# Render login UI if user is not authenticated and halt further execution
if st.session_state.user is None:
    render_auth_ui()
    st.stop()

# ---------------------------------------------------------
# 1. Asset & Model Loaders
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
# 2. ENHANCED MONOD KINETIC ODE (Coupled DO & pH Inhibition)
# ---------------------------------------------------------
def fermentation_ode(y, t, mu_max, Ks, Y_xs, Y_px, min_pH, K_o, OUR_max):
    X, S, P, DO = y
    
    # pH inhibition factor (Bell curve centered at optimal pH = 6.8)
    pH_opt = 6.8
    pH_inhibition = np.exp(-0.5 * ((min_pH - pH_opt) / 1.2) ** 2)
    
    # Monod kinetic growth with dual substrate and DO limitation
    do_term = DO / (K_o + DO) if DO > 0 else 0
    s_term = S / (Ks + S) if S > 0 else 0
    
    mu = mu_max * s_term * do_term * pH_inhibition
    
    dXdt = mu * X
    dSdt = -(1 / Y_xs) * dXdt if (S > 0 and Y_xs > 0) else 0
    dPdt = Y_px * dXdt if S > 0 else 0
    
    # Oxygen consumption coupled directly to biomass growth
    dDOdt = -OUR_max * (dXdt / (X + 1e-5)) if DO > 0 else 0
    
    return [dXdt, dSdt, dPdt, dDOdt]

@st.cache_data(max_entries=100, ttl=3600)
def run_cached_ode(X0, S0, Y_px, mu_max, Ks, Y_xs, min_pH, hours):
    t = np.linspace(0, hours, 100)
    initial_DO = 100.0  # Initial % Dissolved Oxygen saturation
    K_o = 2.0           # Critical oxygen half-saturation (% sat)
    OUR_max = 8.5       # Max oxygen uptake rate coefficient
    
    solution = odeint(
        fermentation_ode, [X0, S0, 0, initial_DO], t, args=(mu_max, Ks, Y_xs, Y_px, min_pH, K_o, OUR_max)
    )
    return t, solution

# ---------------------------------------------------------
# 3. Sidebar Inputs & Reset Button
# ---------------------------------------------------------
st.sidebar.header("⚙️ Bioreactor Parameters")

if st.sidebar.button("🔄 Reset to Default Parameters", use_container_width=True):
    reset_parameters()

initial_S0 = st.sidebar.number_input(
    label="Initial Substrate S₀ (g/L)",
    min_value=10.0, max_value=60.0, value=30.0, step=0.5, format="%.2f",
    key="initial_S0",
    help="Starting substrate/sugar concentration in the fermentation medium.",
)

initial_X0 = st.sidebar.number_input(
    label="Initial Biomass X₀ (g/L)",
    min_value=0.05, max_value=1.0, value=0.20, step=0.05, format="%.2f",
    key="initial_X0",
    help="Initial seed culture biomass concentration.",
)

min_pH = st.sidebar.number_input(
    label="Minimum pH",
    min_value=4.0, max_value=7.5, value=6.8, step=0.1, format="%.2f",
    key="min_pH",
    help="Minimum environment pH. Values outside 5.5-7.0 impose metabolic stress.",
)

sim_time = st.sidebar.number_input(
    label="Simulation Time (Hours)",
    min_value=1, max_value=120, value=48, step=1,
    key="sim_time",
    help="Total batch operation duration.",
)

st.sidebar.subheader("🧫 Kinetic Coefficients")
Y_px_kinetic = st.sidebar.number_input("Product Yield (Y_px)", 0.05, 0.50, value=0.20, step=0.01, key="Y_px_kinetic")
mu_max = st.sidebar.number_input("Max Growth Rate (μ_max)", 0.1, 1.0, value=0.40, step=0.05, key="mu_max")
Ks = st.sidebar.number_input("Half-Sat Constant (Ks)", 0.1, 5.0, value=1.0, step=0.1, key="Ks")
Y_xs = st.sidebar.number_input("Biomass Yield (Y_xs)", 0.1, 0.9, value=0.50, step=0.05, key="Y_xs")

# Logout Option in Sidebar
st.sidebar.divider()
if st.sidebar.button("🚪 Sign Out", use_container_width=True):
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.user = None
    st.rerun()

# ---------------------------------------------------------
# 4. Feature Calculations & Prediction Pipeline
# ---------------------------------------------------------
t_span, ode_solution = run_cached_ode(
    initial_X0, initial_S0, Y_px_kinetic, mu_max, Ks, Y_xs, min_pH, sim_time
)

x_vals = ode_solution[:, 0]
s_vals = ode_solution[:, 1]
p_vals = ode_solution[:, 2]
do_vals = ode_solution[:, 3]

x_final = x_vals[-1]
s_final = s_vals[-1]

delta_X = float(x_final - initial_X0)
delta_S = float(initial_S0 - s_final)

CRITICAL_DO_THRESHOLD = 20.0
dt = t_span[1] - t_span[0] if len(t_span) > 1 else 1.0
do_stress_hours = float(np.sum(do_vals < CRITICAL_DO_THRESHOLD) * dt)

initial_S0_clipped = float(np.clip(initial_S0, 10.0, 60.0))
initial_X0_clipped = float(np.clip(initial_X0, 0.05, 1.0))
delta_X_clipped = float(np.clip(delta_X, 1.0, 20.0))
min_pH_clipped = float(np.clip(min_pH, 4.0, 7.5))
do_stress_hours_clipped = float(np.clip(do_stress_hours, 0.0, 15.0))
Y_px_kinetic_clipped = float(np.clip(Y_px_kinetic, 0.05, 0.5))

s0_x0_ratio = initial_S0_clipped / initial_X0_clipped if initial_X0_clipped > 0 else 0.0

input_data = pd.DataFrame(
    [[
        initial_S0_clipped,
        initial_X0_clipped,
        s0_x0_ratio,
        delta_X_clipped,
        min_pH_clipped,
        do_stress_hours_clipped,
        Y_px_kinetic_clipped,
    ]],
    columns=feature_names,
)

predicted_yield = float(model.predict(input_data)[0])
substrate_efficiency = (predicted_yield / initial_S0_clipped) * 100 if initial_S0_clipped > 0 else 0.0

# ---------------------------------------------------------
# 5. UI Layout - Title & Dashboard Tabs
# ---------------------------------------------------------
st.title("🧪 BioYield-Predict: Bioprocess Digital Twin")
st.markdown("Real-time fermentation product yield prediction and Monod kinetic twin analysis.")

tab_dashboard, tab_history = st.tabs(["📊 Simulation Dashboard", "📜 Saved History"])

# ---------------------------------------------------------
# TAB 1: DASHBOARD
# ---------------------------------------------------------
with tab_dashboard:
    col1, col2, col3 = st.columns(3)
    col1.metric("Biomass Produced (ΔX)", f"{delta_X:.2f} g/L", help="Total biomass produced during the run.")
    col2.metric("Substrate Consumed (ΔS)", f"{delta_S:.2f} g/L", help="Total substrate depleted during batch.")
    
    # Calculate yield comparison delta if baseline exists
    yield_delta = None
    if st.session_state.baseline_yield is not None:
        diff = predicted_yield - st.session_state.baseline_yield
        yield_delta = f"{diff:+.3f} g/L vs Baseline"
        
    col3.metric("Predicted Final Yield", f"{predicted_yield:.3f} g/L", delta=yield_delta)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("💾 Save Run to History", use_container_width=True):
            save_simulation_run(st.session_state.user.id, initial_S0, initial_X0, min_pH, predicted_yield)
    with col_btn2:
        if st.button("📌 Set as Baseline Comparison", use_container_width=True):
            st.session_state.baseline_yield = predicted_yield
            st.toast(f"Baseline set to {predicted_yield:.3f} g/L", icon="📌")

    st.divider()

    col_pred1, col_pred2 = st.columns(2)
    col_pred1.metric("Estimated Substrate Efficiency", f"{substrate_efficiency:.1f}%")
    col_pred2.metric("Hypoxia Stress Duration (DO < 20%)", f"{do_stress_hours:.1f} hrs")

    if min_pH >= 5.0 and do_stress_hours <= 8.0:
        st.success("✅ Fermentation parameters within optimal operating window.")
    else:
        st.warning("⚠️ Operating parameters outside ideal range: Metabolic stress detected.")

    st.divider()

    st.subheader(f"📈 Monod Kinetic & Stress Profiles ({sim_time} hrs)")

    sim_df = pd.DataFrame({
        "Time (hr)": t_span,
        "Biomass (X)": x_vals,
        "Substrate (S)": s_vals,
        "Product (P)": p_vals,
        "DO Concentration (%)": do_vals
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim_df["Time (hr)"], y=sim_df["Biomass (X)"], mode="lines", name="Biomass X (g/L)", line=dict(color="#2ca02c", width=3)))
    fig.add_trace(go.Scatter(x=sim_df["Time (hr)"], y=sim_df["Substrate (S)"], mode="lines", name="Substrate S (g/L)", line=dict(color="#d62728", width=3)))
    fig.add_trace(go.Scatter(x=sim_df["Time (hr)"], y=sim_df["Product (P)"], mode="lines", name="Product P (g/L)", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=sim_df["Time (hr)"], y=sim_df["DO Concentration (%)"], mode="lines", name="Dissolved O₂ (%)", line=dict(color="#ff7f0e", width=2, dash="dash")))

    fig.update_layout(
        xaxis_title="Time (hours)",
        yaxis_title="Concentration (g/L) / DO (%)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        with st.expander("🔍 Model Feature Importances", expanded=True):
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
            fig_imp, ax = plt.subplots(figsize=(6, 3.5))
            importances.plot(kind="barh", color="#10A37F", ax=ax)
            ax.set_title("Gradient Boosting Feature Importance", fontsize=10)
            ax.set_xlabel("Importance Score", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_imp)

    with col_right:
        with st.expander("📥 Export Batch Audit Report", expanded=True):
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

            csv_buffer = io.StringIO()
            summary_report.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📄 Download Summary Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name="bioyield_batch_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

            sim_csv_buffer = io.StringIO()
            sim_df.to_csv(sim_csv_buffer, index=False)
            
            st.download_button(
                label="📈 Download Kinetic Profiles (CSV)",
                data=sim_csv_buffer.getvalue(),
                file_name="bioyield_monod_kinetic_series.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------
# TAB 2: SAVED SIMULATION HISTORY
# ---------------------------------------------------------
with tab_history:
    st.subheader("📜 Your Saved Bioreactor Runs")
    st.write("View past batch simulations stored securely in your Supabase database.")
    
    if st.button("🔄 Refresh History", use_container_width=False):
        st.rerun()

    history_df = fetch_user_history(st.session_state.user.id)
    
    if not history_df.empty:
        display_df = history_df[["created_at", "initial_s0", "initial_x0", "min_ph", "predicted_yield"]].copy()
        display_df.columns = ["Timestamp", "Initial S₀ (g/L)", "Initial X₀ (g/L)", "Min pH", "Predicted Yield (g/L)"]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        hist_csv = io.StringIO()
        display_df.to_csv(hist_csv, index=False)
        st.download_button("📥 Download History Log (CSV)", hist_csv.getvalue(), "simulation_history.csv", "text/csv")
    else:
        st.info("No saved runs found. Click 'Save Run to History' on the main dashboard to save your first batch!")
