import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint

# Set page configuration
st.set_page_config(page_title="BioYield Digital Twin", layout="wide")

st.title("🧫 BioYield Digital Twin: Advanced Bioprocess Studio")
st.markdown("Real-time bioprocess simulation, stoichiometric balancing, and parameter optimization.")

# ---------------------------------------------------------
# 1. SIDEBAR: ORGANIZED DATA INPUT & PRESETS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Bioprocess Configuration")

# Preset selection
preset = st.sidebar.selectbox(
    "Organism / Process Preset",
    ["Custom", "Saccharomyces cerevisiae (Yeast)", "Escherichia coli (Recombinant)"]
)

# Set defaults based on preset
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

# ---------------------------------------------------------
# INPUT VALIDATION ALERTS
# ---------------------------------------------------------
if S0 > 150.0:
    st.warning("⚠️ **High Substrate Concentration:** Substrate levels above 150 g/L may trigger substrate inhibition (Haldane kinetics) or osmotic stress.")
if mode != "Batch" and D >= mu_max:
    st.error("🚨 **Washout Risk:** Dilution rate (D) is greater than or equal to μ_max. Biomass will wash out of the reactor!")

# ---------------------------------------------------------
# MAIN INTERFACE TABS
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

    # ODE Model Formulation
    def bioprocess_model(y, t, mu_max, Ks, Y_xs, Y_ps, D, S0):
        X, S, P = y
        # Monod Kinetics
        mu = mu_max * (S / (Ks + S)) if S > 0 else 0
        
        # Mass Balances
        dXdt = (mu - D) * X
        dSdt = D * (S0 - S) - (mu * X / Y_xs)
        dPdt = (Y_ps * mu * X) - (D * P)
        
        return [max(0, dXdt), dSdt, max(0, dPdt)]

    # Initial conditions
    X0, S0_init, P0 = 0.5, S0 if mode == "Batch" else S0/2, 0.0
    t = np.linspace(0, 48, 200) # 48 hours simulation
    
    # Solve ODEs
    solution = odeint(bioprocess_model, [X0, S0_init, P0], t, args=(mu_max, Ks, Y_xs, Y_ps, D, S0))
    df_sim = pd.DataFrame(solution, columns=["Biomass (X)", "Substrate (S)", "Product (P)"])
    df_sim["Time (h)"] = t

    # Key Performance Metrics Cards
    final_X = df_sim["Biomass (X)"].iloc[-1]
    final_P = df_sim["Product (P)"].iloc[-1]
    baseline_P = 5.0 # Benchmark baseline for comparison
    delta_p = ((final_P - baseline_P) / baseline_P) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Biomass (X)", f"{final_X:.2f} g/L")
    col2.metric("Final Product (P)", f"{final_P:.2f} g/L", delta=f"{delta_p:+.1f}% vs Baseline")
    col3.metric("Substrate Conversion", f"{((S0_init - df_sim['Substrate (S)'].iloc[-1])/S0_init)*100:.1f}%")
    col4.metric("Volumetric Productivity", f"{final_P/48:.3f} g/L·h")

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

    # Molecular weights calculation
    MW_S = c_s*12.011 + h_s*1.008 + o_s*15.999 + n_s*14.007
    MW_X = c_x*12.011 + h_x*1.008 + o_x*15.999 + n_x*14.007

    # Maximum theoretical carbon yield (C-mol/C-mol)
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

    # Generate parameter sweep grid
    temps = np.linspace(25, 42, 10)
    agitations = np.linspace(200, 800, 10)
    T_grid, A_grid = np.meshgrid(temps, agitations)

    # Simple response surface models for yield and power consumption
    # Yield peaks around 32°C and benefits from higher agitation (oxygenation)
    Yield_matrix = Y_xs * np.exp(-((T_grid - 32)**2)/50) * (1 - np.exp(-A_grid/300))
    
    # Power consumption P ∝ N^3 * D^5 (proportional to agitation RPM^3)
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

    # Summary table
    st.dataframe(df_sim.head(10), use_container_width=True)

    csv_data = df_sim.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Full Batch Trajectory CSV",
        data=csv_data,
        file_name=f"bioprocess_run_{preset.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        type="primary"
    )
