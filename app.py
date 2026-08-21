import os
import time
import urllib.parse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
import streamlit as st
from supabase import create_client, Client

# ==========================================
# 1. PAGE CONFIGURATION & INITIAL STATE
# ==========================================
st.set_page_config(
    page_title="BioYield Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session States safely
if "run_history" not in st.session_state:
    st.session_state["run_history"] = []
if "baseline_yield" not in st.session_state:
    st.session_state["baseline_yield"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None

# ==========================================
# 2. SUPABASE AUTH & DB INITIALIZATION
# ==========================================
@st.cache_resource
def init_supabase() -> Client | None:
    """Initialize Supabase client securely from secrets."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]  # Ensure this is the ANON public key
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase configuration missing or invalid: {str(e)}")
        return None

supabase: Client | None = init_supabase()

def get_current_base_url() -> str:
    """Dynamically determine current app URL for OAuth redirects."""
    try:
        if "REDIRECT_URL" in st.secrets:
            return st.secrets["REDIRECT_URL"].rstrip("/")
    except Exception:
        pass
    
    # Fallback default for Streamlit Cloud / local testing
    return "http://localhost:8501"

def handle_oauth_callback():
    """Extract and set user session token on OAuth redirect."""
    if not supabase:
        return
    query_params = st.query_params
    if "access_token" in query_params:
        access_token = query_params["access_token"]
        try:
            user_response = supabase.auth.get_user(access_token)
            if user_response and user_response.user:
                st.session_state["user"] = user_response.user
        except Exception as e:
            st.sidebar.error(f"Authentication failed: {str(e)}")

handle_oauth_callback()

# ==========================================
# 3. SUPABASE DB PERSISTENCE HELPERS
# ==========================================
def save_run_to_supabase(run_data: dict):
    """Persist simulation run to Supabase DB if user is authenticated."""
    if not supabase or not st.session_state.get("user"):
        return
    
    try:
        payload = {
            "user_id": st.session_state["user"].id,
            "run_name": run_data["run_name"],
            "batch_time_hr": float(run_data["batch_time_hr"]),
            "final_biomass": float(run_data["final_biomass"]),
            "final_product": float(run_data["final_product"]),
            "yield_p_s": float(run_data["yield_p_s"]),
            "volumetric_prod": float(run_data["volumetric_prod"]),
            "timestamp": run_data["timestamp"]
        }
        supabase.table("simulation_runs").insert(payload).execute()
    except Exception as e:
        # Gracefully handle DB logging failure without breaking UI
        st.sidebar.warning(f"Note: Could not sync run to database ({str(e)})")

def fetch_user_run_history():
    """Load historical runs from Supabase into session state."""
    if not supabase or not st.session_state.get("user"):
        return
    try:
        response = (
            supabase.table("simulation_runs")
            .select("*")
            .eq("user_id", st.session_state["user"].id)
            .order("created_at", desc=True)
            .execute()
        )
        if response.data:
            st.session_state["run_history"] = [
                {
                    "run_name": r["run_name"],
                    "batch_time_hr": r["batch_time_hr"],
                    "final_biomass": r["final_biomass"],
                    "final_product": r["final_product"],
                    "yield_p_s": r["yield_p_s"],
                    "volumetric_prod": r["volumetric_prod"],
                    "timestamp": r["timestamp"]
                }
                for r in response.data
            ]
    except Exception:
        pass  # Fall back to session state in memory

# ==========================================
# 4. KINETIC MODEL & HEAVY CACHED CALCULATIONS
# ==========================================
def bioprocess_ode_system(state, t, mu_max, Ks, Y_xs, Y_ps, alpha, beta):
    """
    Monod-based Kinetic Differential Equations:
    dX/dt = mu * X
    dS/dt = - (1/Y_xs) * mu * X - (1/Y_ps) * qp * X
    dP/dt = qp * X  where qp = alpha * mu + beta
    """
    X, S, P = state
    # Guard against negative concentration values in ODE solver
    S_pos = max(0.0, S)
    X_pos = max(0.0, X)
    
    mu = mu_max * (S_pos / (Ks + S_pos))
    qp = alpha * mu + beta

    dXdt = mu * X_pos
    dSdt = - (1.0 / Y_xs) * mu * X_pos - (1.0 / Y_ps) * qp * X_pos
    dPdt = qp * X_pos

    return [dXdt, dSdt, dPdt]

@st.cache_data(show_spinner=False)
def run_bioprocess_simulation(
    X0: float,
    S0: float,
    P0: float,
    t_max: float,
    steps: int,
    mu_max: float,
    Ks: float,
    Y_xs: float,
    Y_ps: float,
    alpha: float,
    beta: float,
) -> tuple[pd.DataFrame, dict]:
    """Execute ODE integration with memoization and robust error catching."""
    t = np.linspace(0, t_max, steps)
    initial_state = [X0, S0, P0]

    try:
        solution = odeint(
            bioprocess_ode_system,
            initial_state,
            t,
            args=(mu_max, Ks, Y_xs, Y_ps, alpha, beta),
            mxstep=5000
        )
    except Exception as e:
        raise RuntimeError(f"ODE integration non-convergence: {str(e)}")

    df = pd.DataFrame(solution, columns=["Biomass (X)", "Substrate (S)", "Product (P)"])
    df["Time (hr)"] = t

    # Compute key KPIs safely
    final_p = max(0.0, float(df["Product (P)"].iloc[-1]))
    final_x = max(0.0, float(df["Biomass (X)"].iloc[-1]))
    consumed_s = max(1e-6, S0 - float(df["Substrate (S)"].iloc[-1]))

    yield_p_s = final_p / consumed_s
    volumetric_prod = final_p / t_max if t_max > 0 else 0.0

    kpis = {
        "final_biomass": final_x,
        "final_product": final_p,
        "consumed_substrate": consumed_s,
        "yield_p_s": yield_p_s,
        "volumetric_prod": volumetric_prod,
        "batch_time_hr": t_max
    }

    return df, kpis

@st.cache_data(show_spinner="Computing 2D Parameter Sensitivity Matrix...")
def run_productivity_heatmap_sweep(
    S0: float,
    X0: float,
    P0: float,
    t_max: float,
    steps: int,
    mu_max_range: tuple[float, float],
    S0_range: tuple[float, float],
    Ks: float,
    Y_xs: float,
    Y_ps: float,
    alpha: float,
    beta: float,
    grid_size: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cached computation of 15x15 parameter space sweep.
    Prevents thread locking and freezing on slider updates.
    """
    mu_vals = np.linspace(mu_max_range[0], mu_max_range[1], grid_size)
    s0_vals = np.linspace(S0_range[0], S0_range[1], grid_size)
    prod_matrix = np.zeros((grid_size, grid_size))

    for i, s0_val in enumerate(s0_vals):
        for j, mu_val in enumerate(mu_vals):
            try:
                _, kpis = run_bioprocess_simulation(
                    X0=X0, S0=s0_val, P0=P0, t_max=t_max, steps=steps,
                    mu_max=mu_val, Ks=Ks, Y_xs=Y_xs, Y_ps=Y_ps,
                    alpha=alpha, beta=beta
                )
                prod_matrix[i, j] = kpis["volumetric_prod"]
            except Exception:
                prod_matrix[i, j] = 0.0

    return prod_matrix, mu_vals, s0_vals

# ==========================================
# 5. SIDEBAR: AUTHENTICATION & CONTROLS
# ==========================================
with st.sidebar:
    st.title("🧬 Control Panel")
    
    # --- Authentication Block ---
    st.subheader("Account & Sync")
    if st.session_state.get("user"):
        user_email = st.session_state["user"].email
        st.success(f"Logged in as:\n`{user_email}`")
        if st.button("Log Out", use_container_width=True):
            if supabase:
                supabase.auth.sign_out()
            st.session_state["user"] = None
            st.session_state["run_history"] = []
            st.rerun()
    else:
        st.info("Log in to sync runs across devices.")
        if supabase:
            base_url = get_current_base_url()
            redirect_to = f"{base_url}"
            try:
                auth_url_resp = supabase.auth.get_user() # Validate
            except Exception:
                pass
            
            login_url = f"{st.secrets['SUPABASE_URL']}/auth/v1/authorize?provider=google&redirect_to={urllib.parse.quote(redirect_to)}"
            st.markdown(
                f'<a href="{login_url}" target="_self" style="display: block; text-align: center; background-color: #FF4B4B; color: white; padding: 8px 16px; border-radius: 8px; text-decoration: none; font-weight: bold;">Log in with Google</a>',
                unsafe_allow_html=True
            )
        else:
            st.caption("⚠️ Auth currently offline (Supabase keys required).")

    st.divider()

    # --- Simulation Parameters ---
    st.subheader("Initial Conditions")
    X0 = st.number_input("Initial Biomass X0 (g/L)", min_value=0.01, max_value=50.0, value=0.5, step=0.1)
    S0 = st.number_input("Initial Substrate S0 (g/L)", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
    P0 = st.number_input("Initial Product P0 (g/L)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    t_max = st.slider("Batch Duration (hr)", min_value=5, max_value=120, value=48, step=1)
    
    st.subheader("Kinetic Parameters")
    mu_max = st.slider("Max Specific Growth Rate μ_max (1/hr)", 0.05, 1.5, 0.35, 0.01)
    Ks = st.number_input("Monod Constant Ks (g/L)", 0.01, 10.0, 0.5, 0.05)
    Y_xs = st.slider("Yield Biomass/Substrate Y_X/S (g/g)", 0.1, 0.9, 0.5, 0.05)
    Y_ps = st.slider("Yield Product/Substrate Y_P/S (g/g)", 0.01, 0.9, 0.2, 0.02)
    
    with st.expander("Luedeking-Piret Parameters"):
        alpha = st.number_input("Growth-associated α", 0.0, 5.0, 0.1, 0.05)
        beta = st.number_input("Non-growth-associated β (1/hr)", 0.0, 1.0, 0.01, 0.005)

# Load user historical runs on first auth
if st.session_state.get("user") and not st.session_state["run_history"]:
    fetch_user_run_history()

# Run main simulation safely
try:
    df_sim, current_kpis = run_bioprocess_simulation(
        X0=X0, S0=S0, P0=P0, t_max=float(t_max), steps=200,
        mu_max=mu_max, Ks=Ks, Y_xs=Y_xs, Y_ps=Y_ps, alpha=alpha, beta=beta
    )
except Exception as err:
    st.error(f"Simulation Error: {str(err)}")
    st.stop()

# Auto-set initial baseline if empty
if st.session_state["baseline_yield"] is None:
    st.session_state["baseline_yield"] = current_kpis["yield_p_s"]

# ==========================================
# 6. MAIN APPLICATION LAYOUT & TABS
# ==========================================
st.title("🧪 Fermentation Digital Twin")
st.caption("Real-time Monod & Luedeking-Piret Kinetics Simulator")

# --- Top Metric Dashboard ---
col1, col2, col3, col4 = st.columns(4)

delta_yield = None
if st.session_state["baseline_yield"]:
    delta_val = current_kpis["yield_p_s"] - st.session_state["baseline_yield"]
    delta_yield = f"{delta_val:+.3f} g/g vs baseline"

col1.metric("Final Biomass (X)", f"{current_kpis['final_biomass']:.2f} g/L")
col2.metric("Final Product (P)", f"{current_kpis['final_product']:.2f} g/L")
col3.metric("Yield (Y_P/S)", f"{current_kpis['yield_p_s']:.3f} g/g", delta=delta_yield)
col4.metric("Volumetric Prod.", f"{current_kpis['volumetric_prod']:.3f} g/L/hr")

st.divider()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Time-Series Dynamics", 
    "📊 Yield & Baseline Benchmarking", 
    "🔥 Parameter Sensitivity", 
    "📁 Run History"
])

# --- TAB 1: TIME-SERIES DYNAMICS ---
with tab1:
    st.subheader("Fermentation Trajectory")
    
    fig_dynamics = px.line(
        df_sim, 
        x="Time (hr)", 
        y=["Biomass (X)", "Substrate (S)", "Product (P)"],
        labels={"value": "Concentration (g/L)", "variable": "Species"},
        color_discrete_map={"Biomass (X)": "#2E7D32", "Substrate (S)": "#D84315", "Product (P)": "#1565C0"}
    )
    fig_dynamics.update_layout(hovermode="x unified", height=450)
    st.plotly_chart(fig_dynamics, use_container_width=True)

# --- TAB 2: YIELD & BENCHMARKING ---
with tab2:
    st.subheader("Yield Optimization & Benchmarking")
    
    col_b1, col_b2 = st.columns([1, 2])
    
    with col_b1:
        st.markdown("### Control Actions")
        if st.button("Set Current Run as Baseline", use_container_width=True):
            st.session_state["baseline_yield"] = current_kpis["yield_p_s"]
            st.success("Baseline updated!")
            st.rerun()

        run_name = st.text_input("Run Label", value=f"Run #{len(st.session_state['run_history']) + 1}")
        if st.button("Save Run Record", use_container_width=True):
            record = {
                "run_name": run_name,
                "batch_time_hr": t_max,
                "final_biomass": round(current_kpis["final_biomass"], 2),
                "final_product": round(current_kpis["final_product"], 2),
                "yield_p_s": round(current_kpis["yield_p_s"], 4),
                "volumetric_prod": round(current_kpis["volumetric_prod"], 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state["run_history"].append(record)
            save_run_to_supabase(record)
            st.success(f"Saved '{run_name}'!")
            st.rerun()

    with col_b2:
        st.markdown("### Conversion & Yield Breakdown")
        df_kpi_viz = pd.DataFrame({
            "Metric": ["Substrate Consumed", "Biomass Produced", "Product Formed"],
            "Amount (g/L)": [
                current_kpis["consumed_substrate"], 
                current_kpis["final_biomass"] - X0, 
                current_kpis["final_product"] - P0
            ]
        })
        fig_bar = px.bar(
            df_kpi_viz, 
            x="Metric", 
            y="Amount (g/L)", 
            color="Metric",
            text_auto=".2f",
            color_discrete_sequence=["#FF7043", "#66BB6A", "#42A5F5"]
        )
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: PARAMETER SENSITIVITY ---
with tab3:
    st.subheader("2D Volumetric Productivity Surface (μ_max vs S0)")
    st.caption("Uses cached matrix calculations to ensure rapid interaction.")

    prod_matrix, mu_vals, s0_vals = run_productivity_heatmap_sweep(
        S0=S0, X0=X0, P0=P0, t_max=float(t_max), steps=100,
        mu_max_range=(0.05, 1.2), S0_range=(10.0, 200.0),
        Ks=Ks, Y_xs=Y_xs, Y_ps=Y_ps, alpha=alpha, beta=beta,
        grid_size=15
    )

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=prod_matrix,
        x=np.round(mu_vals, 2),
        y=np.round(s0_vals, 1),
        colorscale="Viridis",
        colorbar=dict(title="Prod. (g/L/hr)")
    ))
    fig_heatmap.update_layout(
        xaxis_title="Max Specific Growth Rate μ_max (1/hr)",
        yaxis_title="Initial Substrate S0 (g/L)",
        height=450
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# --- TAB 4: RUN HISTORY ---
with tab4:
    st.subheader("Historical Runs & Saved Benchmarks")
    
    if st.session_state["run_history"]:
        df_history = pd.DataFrame(st.session_state["run_history"])
        st.dataframe(
            df_history, 
            use_container_width=True,
            column_config={
                "yield_p_s": st.column_config.NumberColumn("Yield Y_P/S (g/g)", format="%.4f"),
                "volumetric_prod": st.column_config.NumberColumn("Vol. Prod (g/L/hr)", format="%.4f")
            }
        )
        
        if st.button("Clear Local History"):
            st.session_state["run_history"] = []
            st.rerun()
    else:
        st.info("No saved runs in this session yet. Save a run in the 'Yield & Baseline Benchmarking' tab.")
