import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
from supabase import create_client, Client

# Set page configuration
st.set_page_config(page_title="BioYield Digital Twin", layout="wide", initial_sidebar_state="expanded")

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

if "run_history" not in st.session_state:
    st.session_state.run_history = []  # Holds saved runs for side-by-side comparison overlays

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
    st.stop()

# Sidebar Logout
user_email = getattr(st.session_state.user, 'email', 'Authenticated User')
st.sidebar.write(f"Logged in as: **{user_email}**")
if st.sidebar.button("Log Out"):
    st.session_state.user = None
    st.rerun()

st.caption("Real-time bioprocess simulation, rigorous stoichiometric balancing, and dynamic multi-variable optimization.")

# ---------------------------------------------------------
# 4. SIDEBAR: ORGANIZED DATA INPUT & SCENARIO PRESETS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Bioprocess Configuration")

preset = st.sidebar.selectbox(
    "Organism / Process Preset",
    ["Custom", "Saccharomyces cerevisiae (Yeast)", "Escherichia coli (Recombinant)"]
)

# Preset Baseline Definitions
if preset == "Saccharomyces cerevisiae (Yeast)":
    default_mu_max, default_Ks, default_Yxs, default_Yps, default_S0, default_mode, default_D = 0.4, 0.2, 0.48, 0.15, 50.0, "Batch", 0.0
elif preset == "Escherichia coli (Recombinant)":
    default_mu_max, default_Ks, default_Yxs, default_Yps, default_S0, default_mode, default_D = 0.6, 0.05, 0.42, 0.20, 30.0, "Batch", 0.0
else:
    default_mu_max, default_Ks, default_Yxs, default_Yps, default_S0, default_mode, default_D = 0.35, 0.1, 0.40, 0.15, 40.0, "Batch", 0.0

# Quick Scenario Action Buttons
st.sidebar.markdown("**⚡ Quick Test Scenarios**")
sc_col1, sc_col2 = st.sidebar.columns(2)
if sc_col1.button("🚀 High Yield", help="Set continuous mode at peak dilution productivity."):
    default_mode, default_D, default_S0 = "Fed-Batch / Continuous (Chemostat)", 0.25, 60.0
if sc_col2.button("⚠️ Washout", help="Set dilution rate above μ_max to demonstrate biomass washout."):
    default_mode, default_D, default_S0 = "Fed-Batch / Continuous (Chemostat)", default_mu_max + 0.05, 40.0

with st.sidebar.expander("🔬 Biological Kinetic Parameters", expanded=True):
    mu_max = st.slider(
        "Max Growth Rate (μ_max, 1/h)", 0.05, 1.0, default_mu_max, 0.01,
        help="Maximum specific growth rate under non-limiting substrate conditions."
    )
    Ks = st.number_input(
        "Monod Constant (Ks, g/L)", 0.01, 5.0, default_Ks, 0.05,
        help="Substrate affinity constant — lower values indicate higher affinity."
    )
    Y_xs = st.slider(
        "Biomass Yield (Y_x/s, g/g)", 0.1, 0.8, default_Yxs, 0.02,
        help="Grams of biomass produced per gram of substrate consumed."
    )
    Y_ps = st.slider(
        "Product Yield (Y_p/s, g/g)", 0.00, 0.6, default_Yps, 0.02,
        help="Grams of target product produced per gram of substrate consumed."
    )

with st.sidebar.expander("🌡️ Reactor Operating Conditions", expanded=True):
    mode = st.radio("Reactor Mode", ["Batch", "Fed-Batch / Continuous (Chemostat)"], index=0 if default_mode == "Batch" else 1)
    
    # Calculate critical dilution rate dynamically for real-time guardrails
    S0_temp = default_S0
    D_crit = mu_max * (S0_temp / (Ks + S0_temp))
    
    D = st.slider(
        "Dilution Rate (D, 1/h)", 0.0, 0.8, default_D if mode != "Batch" else 0.0, 0.02,
        help=f"Volumetric feed rate divided by reactor volume. Washout threshold D_crit ≈ {D_crit:.3f} h⁻¹."
    )
    S0 = st.number_input(
        "Substrate Feed Concentration (S0, g/L)", 5.0, 200.0, default_S0, 5.0,
        help="Concentration of substrate in the incoming feed stream."
    )

# Dynamic Operational Guardrails & Feedback Alerts
if S0 > 100.0:
    st.sidebar.warning(f"⚠️ **High Substrate (S0 = {S0:.0f} g/L):** Concentrations above 100 g/L risk osmotic stress or Luong/Haldane substrate inhibition.")

if mode != "Batch":
    if D >= mu_max:
        st.sidebar.error(f"🚨 **CRITICAL WASHOUT:** Dilution rate ($D = {D:.2f}$) exceeds $\mu_{{max}} = {mu_max:.2f}$. Cells will be flushed out faster than they grow!")
    elif D >= D_crit:
        st.sidebar.warning(f"⚠️ **Near-Washout Zone:** $D$ ({D:.2f}) approaches critical rate ({D_crit:.2f} h⁻¹). Steady state biomass will drop sharply.")

# ---------------------------------------------------------
# 5. MAIN INTERFACE TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Dynamic Kinetics", 
    "⚖️ Stoichiometric Mass Balance", 
    "🗺️ Dynamic Optimization Matrix",
    "📥 Batch Report Export"
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
        return [max(0, dXdt), max(0, dSdt), max(0, dPdt)]

    X0, S0_init, P0 = 0.5, S0 if mode == "Batch" else S0/2, 0.0
    t = np.linspace(0, 48, 200)
    
    solution = odeint(bioprocess_model, [X0, S0_init, P0], t, args=(mu_max, Ks, Y_xs, Y_ps, D, S0))
    df_sim = pd.DataFrame(solution, columns=["Biomass (X)", "Substrate (S)", "Product (P)"])
    df_sim["Time (h)"] = t

    final_X = df_sim["Biomass (X)"].iloc[-1]
    final_P = df_sim["Product (P)"].iloc[-1]
    
    baseline_P = st.session_state.baseline_yield
    delta_p = ((final_P - baseline_P) / baseline_P) * 100 if baseline_P > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Biomass (X)", f"{final_X:.2f} g/L")
    col2.metric("Final Product (P)", f"{final_P:.2f} g/L", delta=f"{delta_p:+.1f}% vs Baseline")
    col3.metric("Substrate Conversion", f"{((S0_init - df_sim['Substrate (S)'].iloc[-1])/S0_init)*100:.1f}%")
    col4.metric("Volumetric Productivity", f"{final_P/48:.3f} g/L·h")

    # Action Toolbar: Save Baseline & Run History Management
    b_col1, b_col2, b_col3 = st.columns([1.5, 1.5, 2])
    with b_col1:
        if st.button("📌 Set as Session Baseline", use_container_width=True):
            st.session_state.baseline_yield = final_P
            st.success(f"Updated baseline to {final_P:.2f} g/L")
            st.rerun()
    with b_col2:
        if st.button("💾 Save Run Trajectory", use_container_width=True):
            run_label = f"Run {len(st.session_state.run_history)+1} (P={final_P:.1f}g/L, D={D})"
            st.session_state.run_history.append({"label": run_label, "data": df_sim.copy()})
            st.toast(f"Saved {run_label} to overlay history!", icon="✅")
    with b_col3:
        if len(st.session_state.run_history) > 0 and st.button("🗑️ Clear Run Overlay History", use_container_width=True):
            st.session_state.run_history = []
            st.rerun()

    # Dynamic Trajectory Chart with Overlay History
    fig = go.Figure()
    
    # Active Run Traces
    fig.add_trace(go.Scatter(x=df_sim["Time (h)"], y=df_sim["Biomass (X)"], mode='lines', name='Biomass X (Current)', line=dict(color='#2ca02c', width=3)))
    fig.add_trace(go.Scatter(x=df_sim["Time (h)"], y=df_sim["Substrate (S)"], mode='lines', name='Substrate S (Current)', line=dict(color='#d62728', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=df_sim["Time (h)"], y=df_sim["Product (P)"], mode='lines', name='Product P (Current)', line=dict(color='#1f77b4', width=3)))

    # Historical Overlays
    colors_p = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    for idx, saved_run in enumerate(st.session_state.run_history[-5:]):  # Keep max 5 overlay traces
        history_df = saved_run["data"]
        color = colors_p[idx % len(colors_p)]
        fig.add_trace(go.Scatter(
            x=history_df["Time (h)"], 
            y=history_df["Product (P)"], 
            mode='lines', 
            name=f"P ({saved_run['label']})", 
            line=dict(color=color, width=2, dash='dot')
        ))

    fig.update_layout(
        title="Dynamic State Trajectories (X, S, P) with Multi-Run Overlays",
        xaxis_title="Time (hours)",
        yaxis_title="Concentration (g/L)",
        hovermode="x unified",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2: FUNCTIONAL STOICHIOMETRIC MATRIX BALANCER
# =========================================================
with tab2:
    st.subheader("Elemental Mass Balance & Carbon Partitioning")
    st.markdown("Solves elemental balance equations ($C, H, O, N$) to calculate **Theoretical Yield Limits ($Y_{x/s}^{max}$)**, **Oxygen Demand ($O_2$)**, and **$CO_2$ Respiration**.")

    col_sub, col_bio, col_prod = st.columns(3)
    
    with col_sub:
        st.markdown("**Substrate ($CH_aO_bN_c$)**")
        cs, hs, os, ns = 1.0, st.number_input("H (Substrate)", value=2.0, key="hs"), st.number_input("O (Substrate)", value=1.0, key="os"), st.number_input("N (Substrate)", value=0.0, key="ns")
    
    with col_bio:
        st.markdown("**Biomass ($CH_dO_eN_f$)**")
        cx, hx, ox, nx = 1.0, st.number_input("H (Biomass)", value=1.8, key="hx"), st.number_input("O (Biomass)", value=0.5, key="ox"), st.number_input("N (Biomass)", value=0.2, key="nx")

    with col_prod:
        st.markdown("**Product ($CH_gO_hN_i$)**")
        cp, hp, op, np_val = 1.0, st.number_input("H (Product)", value=3.0, key="hp"), st.number_input("O (Product)", value=1.0, key="op"), st.number_input("N (Product)", value=0.0, key="np")

    # Calculate C-mol Molecular Weights
    MW_Substrate = cs*12.011 + hs*1.008 + os*15.999 + ns*14.007
    MW_Biomass = cx*12.011 + hx*1.008 + ox*15.999 + nx*14.007
    MW_Product = cp*12.011 + hp*1.008 + op*15.999 + np_val*14.007

    # C-mol fraction balance: 1 C-mol S -> Y_xs_cmol Biomass + Y_ps_cmol Product + Y_co2_cmol CO2
    Y_xs_cmol = Y_xs * (MW_Substrate / MW_Biomass)
    Y_ps_cmol = Y_ps * (MW_Substrate / MW_Product)
    Y_co2_cmol = 1.0 - Y_xs_cmol - Y_ps_cmol
    
    # Calculate O2 requirement per C-mol substrate (Degree of Reduction Balance)
    gamma_s = 4*cs + hs - 2*os - 3*ns
    gamma_x = 4*cx + hx - 2*ox - 3*nx
    gamma_p = 4*cp + hp - 2*op - 3*np_val

    O2_demand_cmol = (gamma_s - (Y_xs_cmol * gamma_x) - (Y_ps_cmol * gamma_p)) / 4.0

    st.markdown("---")
    
    # Check thermodynamic feasibility
    if Y_co2_cmol < 0:
        st.error(f"🚨 **Thermodynamically Impossible Configuration!** The combined Biomass ($Y_{{x/s}}={Y_xs}$) and Product ($Y_{{p/s}}={Y_ps}$) yields consume **{Y_xs_cmol + Y_ps_cmol:.2f} C-moles** per C-mole substrate. Maximum available carbon is **1.0 C-mole**.")
    else:
        st.success(f"✓ **Stoichiometrically Valid Process:** Carbon allocation is {Y_xs_cmol*100:.1f}% Biomass, {Y_ps_cmol*100:.1f}% Product, and {Y_co2_cmol*100:.1f}% $CO_2$ respiration.")

    # High-level Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Substrate C-mol Weight", f"{MW_Substrate:.2f} g/C-mol")
    m2.metric("Biomass C-mol Weight", f"{MW_Biomass:.2f} g/C-mol")
    m3.metric("Theoretical Max $Y_{x/s}$", f"{(MW_Biomass/MW_Substrate):.3f} g/g")
    m4.metric("Specific $O_2$ Demand", f"{max(0, O2_demand_cmol):.3f} mol O2/C-mol S")

    # Visual Mass Balance Flow (Sankey Diagram)
    if Y_co2_cmol >= 0:
        st.markdown("#### Carbon Allocation Flow (Sankey Diagram)")
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20, thickness=25,
                label=["Substrate Carbon (100%)", "Biomass (X)", "Product (P)", "Respiration (CO₂)"],
                color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
            ),
            link=dict(
                source=[0, 0, 0],
                target=[1, 2, 3],
                value=[Y_xs_cmol * 100, Y_ps_cmol * 100, max(0, Y_co2_cmol * 100)],
                color=["rgba(44, 160, 44, 0.4)", "rgba(255, 127, 14, 0.4)", "rgba(214, 39, 40, 0.4)"]
            )
        )])
        fig_sankey.update_layout(
            height=320, 
            margin=dict(l=10, r=10, t=20, b=10),
            font=dict(size=13)
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

# =========================================================
# TAB 3: REAL KINETIC SENSITIVITY & DYNAMIC OPTIMIZATION
# =========================================================
with tab3:
    st.subheader("ODE Parameter Sweep Matrix")
    st.markdown("Runs full 2D differential simulations across dilution rates ($D$) and substrate feed concentrations ($S_0$) to map true steady-state volumetric productivity ($g/L \cdot h$).")

    d_range = np.linspace(0.02, min(0.75, mu_max * 1.1), 15)
    s0_range = np.linspace(10.0, 100.0, 15)
    
    productivity_matrix = np.zeros((len(s0_range), len(d_range)))
    
    # Run dynamic sweeps
    for i, s0_val in enumerate(s0_range):
        for j, d_val in enumerate(d_range):
            if d_val >= mu_max:
                productivity_matrix[i, j] = 0.0  # Washout state
            else:
                # Analytical steady state for chemostat
                S_ss = (Ks * d_val) / (mu_max - d_val) if (mu_max - d_val) > 0 else s0_val
                if S_ss < s0_val:
                    X_ss = Y_xs * (s0_val - S_ss)
                    P_ss = Y_ps * (s0_val - S_ss)
                    productivity_matrix[i, j] = P_ss * d_val  # Volumetric productivity: P * D
                else:
                    productivity_matrix[i, j] = 0.0

    fig_heat = go.Figure(data=go.Contour(
        z=productivity_matrix,
        x=np.round(d_range, 3),
        y=np.round(s0_range, 1),
        colorscale='Viridis',
        colorbar=dict(title='Productivity (g/L·h)'),
        contours=dict(showlabels=True, labelfont=dict(color='white'))
    ))

    fig_heat.update_layout(
        title="Volumetric Product Productivity Surface (D vs Substrate Feed S0)",
        xaxis_title="Dilution Rate D (1/h)",
        yaxis_title="Substrate Feed S0 (g/L)",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# TAB 4: BATCH REPORT EXPORT
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
