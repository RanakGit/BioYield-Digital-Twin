import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
from supabase import create_client, Client

# Set page configuration
st.set_page_config(page_title="BioYield Digital Twin", layout="wide")

# ---------------------------------------------------------
# 1. SUPABASE AUTHENTICATION & SESSION SETUP
# ---------------------------------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# Session State Initialization
if "user" not in st.session_state:
    st.session_state.user = None

if "baseline_yield" not in st.session_state:
    st.session_state.baseline_yield = 5.0  # Default baseline yield (g/L)

# ---------------------------------------------------------
# 2. OAUTH CALLBACK HANDLER & PKCE GUARD
# ---------------------------------------------------------
if st.session_state.user is None and "code" in st.query_params:
    auth_code = st.query_params["code"]
    try:
        res = supabase.auth.exchange_code_for_session({"auth_code": auth_code})
        if res and hasattr(res, 'user') and res.user:
            st.session_state.user = res.user
        elif res and hasattr(res, 'session') and res.session:
            st.session_state.user = res.session.user
            
        st.query_params.clear()
        st.rerun()
    except Exception:
        # Fallback for Streamlit Cloud refresh verifier clearing
        st.query_params.clear()

# Active Session Persistence Check
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
# 3. AUTHENTICATION UI LAYER
# ---------------------------------------------------------
st.title("🧫 BioYield Digital Twin: Advanced Bioprocess Studio")

if st.session_state.user is None:
    st.info("Welcome! Please log in to access the bioprocess digital twin environment.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Email / Password Sign In")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Log In / Sign Up", type="primary"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                if hasattr(res, 'user') and res.user:
                    st.session_state.user = res.user
                    st.success("Logged in successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Authentication error: {e}")

    with col2:
        st.subheader("Single Sign-On")
        if st.button("Sign in with Google", type="secondary"):
            try:
                response = supabase.auth.sign_in_with_oauth(
                    {
                        "provider": "google",
                        "options": {
                            "redirect_to": "https://bioyield-digital-twin-wgn36c9vv3sheccq2fe9un.streamlit.app"
                        }
                    }
                )
                if response and hasattr(response, 'url'):
                    st.link_button("Proceed to Google Login", response.url, type="primary")
            except Exception as e:
                st.error(f"Error initiating Google Login: {e}")

    st.markdown("---")
    st.stop()  # Halt execution until authenticated

# Sidebar Logout once authenticated
user_email = getattr(st.session_state.user, 'email', 'Authenticated User')
st.sidebar.write(f"Logged in as: **{user_email}**")
if st.sidebar.button("Log Out"):
    st.session_state.user = None
    st.rerun()

st.markdown("Real-time bioprocess simulation, stoichiometric balancing, and parameter optimization.")

# ---------------------------------------------------------
# 4. SIDEBAR: ORGANIZED DATA INPUT & PRESETS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Bioprocess Configuration")

preset = st.sidebar.selectbox(
    "Organism / Process Preset",
    ["Custom", "Saccharomyces cerevisiae (Yeast)", "Escherichia coli (Recombinant)"]
)

if preset == "Saccharomyces cerevisiae (Yeast)":
    default_mu_max, default_Ks, default_Yxs, default_S0 = 0.4, 0.2, 0.5, 50.0
elif preset == "Escherichia coli (Recombinant)":
    default_mu_max, default_Ks, default_Yxs, default_S0 = 0.6, 0.05, 0.45, 30.0
else:
    default_mu_max, default_Ks, default_Yxs, default_S0 = 0.35, 0.1, 0.4, 40.0

with st.sidebar.expander("🔬 Kinetic Parameters", expanded=True):
    mu_max = st.slider("Max Specific Growth Rate (μ_max, 1/h)", 0.05, 1.0, default_mu_max, 0.01)
    Ks = st.number_input("Monod Constant (Ks, g/L)", 0.01, 5.0, default_Ks, 0.05)
    Y_xs = st.slider("Biomass Yield (Y_x/s, g/g)", 0.1, 0.8, default_Yxs, 0.02)
    Y_ps = st.slider("Product Yield (Y_p/s, g/g)", 0.05, 0.6, 0.2, 0.02)

with st.sidebar.expander("🌡️ Operating Conditions", expanded=True):
    mode = st.radio("Reactor Mode", ["Batch", "Fed-Batch / Continuous (Chemostat)"])
    D = st.slider("Dilution Rate (D, 1/h)", 0.0, 0.8, 0.1 if mode != "Batch" else 0.0, 0.02)
    S0 = st.number_input("Substrate Feed (S0, g/L)", 5.0, 200.0, default_S0, 5.0)
    temp = st.slider("Temperature (°C)", 20, 45, 30)
    ph = st.slider("pH Level", 4.0, 9.0, 6.8, 0.1)
    agitation = st.slider("Agitation Speed (RPM)", 100, 1000, 400, 50)

# Input Validation Alerts
if S0 > 150.0:
    st.warning("⚠️ **High Substrate Concentration:** Substrate levels above 150 g/L may trigger substrate inhibition (Haldane kinetics) or osmotic stress.")
if mode != "Batch" and D >= mu_max:
    st.error("🚨 **Washout Risk:** Dilution rate (D) is greater than or equal to μ_max. Biomass will wash out of the reactor!")

# ---------------------------------------------------------
# 5. MAIN INTERFACE TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Dynamic Kinetics", 
    "⚖️ Stoichiometry & Yields", 
    "🗺️ Sensitivity & Optimization",
    "📥 Batch Report & Export"
])

# =========================================================
# TAB 1: DYNAMIC FERMENTATION & KINETIC CURVES
# =========================================================
with tab1:
    st.subheader("Interactive Fermentation Profile")

    def bioprocess_model(y, t, mu_max, Ks, Y_xs, Y_ps, D, S0):
        X, S, P = y
        mu = mu_max * (S / (Ks + S)) if S > 0 else 0
        dXdt = (mu - D) * X
        dSdt = D * (S0 - S) - (mu * X / Y_xs)
        dPdt = (Y_ps * mu * X) - (D * P)
        return [max(0, dXdt), dSdt, max(0, dPdt)]

    X0, S0_init, P0 = 0.5, S0 if mode == "Batch" else S0/2, 0.0
    t = np.linspace(0, 48, 200)
    
    solution = odeint(bioprocess_model, [X0, S0_init, P0], t, args=(mu_max, Ks, Y_xs, Y_ps, D, S0))
    df_sim = pd.DataFrame(solution, columns=["Biomass (X)", "Substrate (S)", "Product (P)"])
    df_sim["Time (h)"] = t

    final_X = df_sim["Biomass (X)"].iloc[-1]
    final_P = df_sim["Product (P)"].iloc[-1]
    
    # Calculate percentage change against session state baseline yield
    baseline_P = st.session_state.baseline_yield
    delta_p = ((final_P - baseline_P) / baseline_P) * 100 if baseline_P > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Biomass (X)", f"{final_X:.2f} g/L")
    col2.metric("Final Product (P)", f"{final_P:.2f} g/L", delta=f"{delta_p:+.1f}% vs Baseline")
    col3.metric("Substrate Conversion", f"{((S0_init - df_sim['Substrate (S)'].iloc[-1])/S0_init)*100:.1f}%")
    col4.metric("Volumetric Productivity", f"{final_P/48:.3f} g/L·h")

    if st.button("Save Current Product Yield as New Session Baseline"):
        st.session_state.baseline_yield = final_P
        st.success(f"Updated baseline yield to {final_P:.2f} g/L")
        st.rerun()

    # Plotly Interactive Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sim["Time (h)"], y=df_sim["Biomass (X)"], mode='lines', name='Biomass X (g/L)', line=dict(color='#2ca02c', width=3)))
    fig.add_trace(go.Scatter(x=df_sim["Time (h)"], y=df_sim["Substrate (S)"], mode='lines', name='Substrate S (g/L)', line=dict(color='#d62728', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=df_sim["Time (h)"], y=df_sim["Product (P)"], mode='lines', name='Product P (g/L)', line=dict(color='#1f77b4', width=3)))

    fig.update_layout(
        title="Real-Time State Trajectories over 48 Hours",
        xaxis_title="Time (hours)",
        yaxis_title="Concentration (g/L)",
        hovermode="x unified",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2: YIELD & STOICHIOMETRIC BALANCE CALCULATOR
# =========================================================
with tab2:
    st.subheader("Elemental & Stoichiometric Mass Balance")
    st.markdown("Calculates theoretical yield limits based on elemental chemical formulas ($CH_aO_bN_c$).")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Substrate Formula (e.g., Glucose: $CH_2O$)**")
        c_s = st.number_input("C (Substrate)", value=1.0)
        h_s = st.number_input("H (Substrate)", value=2.0)
        o_s = st.number_input("O (Substrate)", value=1.0)
        n_s = st.number_input("N (Substrate)", value=0.0)

    with c2:
        st.write("**Biomass Formula (e.g., Dry Cell Weight: $CH_{1.8}O_{0.5}N_{0.2}$)**")
        c_x = st.number_input("C (Biomass)", value=1.0)
        h_x = st.number_input("H (Biomass)", value=1.8)
        o_x = st.number_input("O (Biomass)", value=0.5)
        n_x = st.number_input("N (Biomass)", value=0.2)

    MW_S = c_s*12.011 + h_s*1.008 + o_s*15.999 + n_s*14.007
    MW_X = c_x*12.011 + h_x*1.008 + o_x*15.999 + n_x*14.007

    max_Y_xs_mol = c_x / c_s
    max_Y_xs_mass = max_Y_xs_mol * (MW_X / MW_S)

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Substrate Mol. Weight", f"{MW_S:.2f} g/C-mol")
    res_col2.metric("Biomass Mol. Weight", f"{MW_X:.2f} g/C-mol")
    res_col3.metric("Max Theoretical Y_x/s Limit", f"{max_Y_xs_mass:.3f} g/g")

    if Y_xs > max_Y_xs_mass:
        st.error(f"⚠️ Selected Yield ($Y_{{x/s}} = {Y_xs}$) exceeds stoichiometric limit ($^{{max}}Y_{{x/s}} = {max_Y_xs_mass:.3f}$)! Adjust yield parameters.")

# =========================================================
# TAB 3: SENSITIVITY & TRADE-OFF MATRIX
# =========================================================
with tab3:
    st.subheader("Process Sensitivity Heatmap")
    st.markdown("Simulates trade-offs between **Agitation (RPM)**, **Temperature (°C)**, Yield, and estimated **Energy Cost**.")

    temps = np.linspace(25, 42, 10)
    agitations = np.linspace(200, 800, 10)
    T_grid, A_grid = np.meshgrid(temps, agitations)

    Yield_matrix = Y_xs * np.exp(-((T_grid - 32)**2)/50) * (1 - np.exp(-A_grid/300))
    Power_cost_matrix = (A_grid / 400)**3 * 1.5 + (T_grid - 25)*0.1

    plot_type = st.radio("Select Response Surface Metric:", ["Biomass Yield (g/g)", "Energy Cost Index (kW)"])

    if plot_type == "Biomass Yield (g/g)":
        fig_heat = px.imshow(
            Yield_matrix, 
            x=np.round(temps, 1), 
            y=np.round(agitations, 0),
            labels=dict(x="Temperature (°C)", y="Agitation (RPM)", color="Yield (g/g)"),
            color_continuous_scale="Viridis"
        )
    else:
        fig_heat = px.imshow(
            Power_cost_matrix, 
            x=np.round(temps, 1), 
            y=np.round(agitations, 0),
            labels=dict(x="Temperature (°C)", y="Agitation (RPM)", color="Power Index"),
            color_continuous_scale="Plasma"
        )

    fig_heat.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# TAB 4: BATCH REPORT & EXPORT
# =========================================================
with tab4:
    st.subheader("Export Run Data & Parameters")
    st.markdown("Download generated simulation trajectories and batch parameters for record-keeping or downstream analytical modeling.")

    st.dataframe(df_sim.head(10), use_container_width=True)

    csv_data = df_sim.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Full Batch Trajectory CSV",
        data=csv_data,
        file_name=f"bioprocess_run_{preset.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        type="primary"
    )
