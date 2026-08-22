import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import datetime
import html
import io
import requests

# ==========================================
# 1. PAGE & INDUSTRIAL DESIGN SYSTEM (CSS)
# ==========================================
st.set_page_config(
    page_title="BioTwin Pro | Enterprise Bioprocess Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    
    .brand-title {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .brand-badge {
        background: linear-gradient(135deg, #0284C7, #0D9488);
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .kpi-container {
        display: grid;
        grid-content-columns: repeat(auto-fit, minmax(200px, 1fr));
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        flex: 1;
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-left: 4px solid #0284C7;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        transition: all 0.2s ease-in-out;
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.7;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }

    .section-banner {
        background: rgba(128, 128, 128, 0.05);
        border-left: 3px solid #0D9488;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AUTHENTICATION & MULTI-TENANCY GATE
# ==========================================
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.markdown("## 🧬 BioTwin Pro Enterprise Login")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        # Using a form prevents Streamlit from refreshing mid-typing
        with st.form("login_form"):
            tenant_key = st.text_input("Enterprise License Key / API Token", type="password")
            organization = st.text_input("Organization / Facility ID", "BioProcess Corp Alpha")
            submit_button = st.form_submit_button("Authenticate Workspace", use_container_width=True)
            
            if submit_button:
                # .strip() removes any accidental trailing/leading spaces
                clean_key = tenant_key.strip()
                
                VALID_KEYS = ["demo", "biotwin_enterprise_secret_key", "admin"]
                
                if clean_key in VALID_KEYS:
                    st.session_state['authenticated'] = True
                    st.session_state['org'] = organization.strip()
                    st.success("Authorized! Loading Digital Twin Workspace...")
                    st.rerun()
                else:
                    st.error("Invalid Enterprise Token. Try 'demo' or 'admin'.")
                    
    st.stop()

# ==========================================
# 3. ADVANCED NUMERICAL COMPUTATION ENGINE
# ==========================================
def haldane_fedbatch_model(y, t, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2, F_in, S_feed):
    X, S, P, DO, V = max(0, y[0]), max(0, y[1]), max(0, y[2]), max(0, y[3]), max(1e-3, y[4])
    
    # Haldane kinetic expression with substrate inhibition
    mu = mu_max * S / (Ks + S + ((S**2) / Ki)) if S > 0 else 0.0
    D = F_in / V
    
    dXdt = (mu - D) * X
    dSdt = D * (S_feed - S) - ((1.0 / Y_xs) * mu * X) if S > 0 else D * (S_feed - S)
    dPdt = -D * P + (Y_ps * mu * X) + (alpha * mu * X) + (beta * X)
    
    # Oxygen Mass Transfer Balance (OTR - OUR)
    OTR = kla * (C_star - DO)
    OUR = q_O2 * X * 1000.0  # mg/L/h
    dDOdt = OTR - OUR - (D * DO)
    dVdt = F_in
    
    return [dXdt, dSdt, dPdt, dDOdt, dVdt]

@st.cache_data(ttl=3600)
def run_advanced_simulation(X0, S0, P0, DO0, V0, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2, F_in, S_feed, batch_time, n_points=300):
    t = np.linspace(0, batch_time, n_points)
    try:
        sol = odeint(
            haldane_fedbatch_model, 
            [X0, S0, P0, DO0, V0], 
            t, 
            args=(mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2, F_in, S_feed)
        )
        df = pd.DataFrame({
            'Time (hr)': t,
            'Biomass X (g/L)': np.clip(sol[:, 0], 0, None),
            'Substrate S (g/L)': np.clip(sol[:, 1], 0, None),
            'Product P (g/L)': np.clip(sol[:, 2], 0, None),
            'Dissolved Oxygen DO (mg/L)': np.clip(sol[:, 3], 0, None),
            'Reactor Volume V (L)': sol[:, 4]
        })
        return df, True
    except Exception as e:
        return pd.DataFrame(), False

# ==========================================
# 4. ORGANISM PRESETS & DATABASE ENGINE
# ==========================================
ORGANISM_PRESETS = {
    "E. coli Recombinant Protein": {
        "mu_max": 0.65, "Ks": 0.20, "Ki": 150.0, "Y_xs": 0.50, "Y_ps": 0.18,
        "alpha": 0.12, "beta": 0.02, "X0": 0.15, "S0": 30.0, "batch_time": 18.0,
        "kla": 180.0, "F_in": 0.05, "S_feed": 200.0
    },
    "S. cerevisiae (Bioethanol)": {
        "mu_max": 0.42, "Ks": 0.45, "Ki": 80.0, "Y_xs": 0.14, "Y_ps": 0.46,
        "alpha": 0.05, "beta": 0.01, "X0": 0.50, "S0": 110.0, "batch_time": 32.0,
        "kla": 120.0, "F_in": 0.0, "S_feed": 0.0
    },
    "CHO Cell Line (mAb Expression)": {
        "mu_max": 0.038, "Ks": 0.12, "Ki": 300.0, "Y_xs": 0.62, "Y_ps": 0.28,
        "alpha": 0.18, "beta": 0.004, "X0": 0.20, "S0": 18.0, "batch_time": 120.0,
        "kla": 25.0, "F_in": 0.005, "S_feed": 50.0
    },
    "Custom / Lab Sandbox": {
        "mu_max": 0.45, "Ks": 0.25, "Ki": 100.0, "Y_xs": 0.45, "Y_ps": 0.20,
        "alpha": 0.05, "beta": 0.01, "X0": 0.10, "S0": 25.0, "batch_time": 24.0,
        "kla": 100.0, "F_in": 0.0, "S_feed": 0.0
    }
}

# ==========================================
# 5. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Enterprise Twin Config")
    st.caption(f"Connected Workspace: **{st.session_state.get('org', 'Default')}**")
    preset_choice = st.selectbox("Active Strain Profile", list(ORGANISM_PRESETS.keys()))
    preset = ORGANISM_PRESETS[preset_choice]

    st.markdown("---")
    st.markdown("#### Kinetic & Inhibition Parameters")
    
    default_mu = st.session_state.get('fitted_mu', float(preset["mu_max"]))
    mu_max = st.slider("μ_max (Max Growth Rate, 1/h)", 0.01, 1.50, default_mu, 0.01)
    Ks = st.slider("Ks (Affinity Constant, g/L)", 0.01, 2.00, float(preset["Ks"]), 0.01)
    Ki = st.number_input("Ki (Haldane Substrate Inhibition, g/L)", 1.0, 1000.0, float(preset["Ki"]))
    Y_xs = st.slider("Y_x/s (Biomass Yield, g/g)", 0.05, 0.90, float(preset["Y_xs"]), 0.01)
    Y_ps = st.slider("Y_p/s (Product Yield, g/g)", 0.00, 0.90, float(preset["Y_ps"]), 0.01)

    st.markdown("---")
    st.markdown("#### Fed-Batch & Mass Transfer (kL a)")
    kla = st.slider("kL a (Mass Transfer Coeff, 1/h)", 5.0, 500.0, float(preset["kla"]))
    F_in = st.number_input("Substrate Feed Rate F_in (L/h)", 0.0, 5.0, float(preset["F_in"]), 0.01)
    S_feed = st.number_input("Feed Substrate Conc S_feed (g/L)", 0.0, 1000.0, float(preset["S_feed"]), 10.0)

    st.markdown("---")
    st.markdown("#### Initial Reactor States")
    X0 = st.number_input("Initial Biomass X₀ (g/L)", 0.01, 50.0, float(preset["X0"]), 0.05)
    S0 = st.number_input("Initial Substrate S₀ (g/L)", 0.1, 500.0, float(preset["S0"]), 1.0)
    batch_time = st.slider("Batch Duration (Hours)", 4.0, 200.0, float(preset["batch_time"]), 1.0)

    alpha, beta = preset["alpha"], preset["beta"]

# ==========================================
# 6. EXECUTE SIMULATION & METRICS
# ==========================================
sim_df, success = run_advanced_simulation(
    X0, S0, 0.0, 7.0, 1.0, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, 
    kla, 7.0, 0.15, F_in, S_feed, batch_time
)

if not success or sim_df.empty:
    st.error("⚠️ ODE Solver Numerical Instability. Verify kinetic constants.")
    st.stop()

final_X = sim_df['Biomass X (g/L)'].iloc[-1]
final_P = sim_df['Product P (g/L)'].iloc[-1]
final_V = sim_df['Reactor Volume V (L)'].iloc[-1]
min_DO = sim_df['Dissolved Oxygen DO (mg/L)'].min()
consumed_S = (S0 + (F_in * S_feed * batch_time)) - sim_df['Substrate S (g/L)'].iloc[-1]
vol_prod = (final_P * final_V) / batch_time if batch_time > 0 else 0
overall_yield = (final_P / consumed_S) if consumed_S > 0 else 0

# ==========================================
# 7. HEADER & KPI DASHBOARD
# ==========================================
st.markdown(f"""
<div class="brand-header">
    <div class="brand-title">
        <span>🧬 BioTwin Pro</span>
        <span class="brand-badge">Enterprise v3.0</span>
    </div>
    <div style="font-size: 0.85rem; opacity: 0.8;">
        Strain: <b>{preset_choice}</b> | Mode: <b>{'Fed-Batch' if F_in > 0 else 'Batch'}</b> | System Status: <span style="color:#0D9488; font-weight:700;">● Engine Operational</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card" style="border-left-color: #0D9488;">
        <div class="kpi-label">Final Biomass (X)</div>
        <div class="kpi-value">{final_X:.2f} <span style="font-size:0.9rem;">g/L</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #0284C7;">
        <div class="kpi-label">Final Product (P)</div>
        <div class="kpi-value" style="color: #0284C7;">{final_P:.2f} <span style="font-size:0.9rem;">g/L</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #D97706;">
        <div class="kpi-label">Volumetric Productivity</div>
        <div class="kpi-value">{vol_prod:.3f} <span style="font-size:0.9rem;">g/h</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #8B5CF6;">
        <div class="kpi-label">Min Dissolved O₂</div>
        <div class="kpi-value" style="color: {'#E11D48' if min_DO < 1.0 else '#0D9488'};">{min_DO:.2f} <span style="font-size:0.9rem;">mg/L</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 8. TAB WORKFLOW SYSTEM
# ==========================================
tab_twin, tab_fitting, tab_sensitivity, tab_report = st.tabs([
    "📈 Dynamic Digital Twin",
    "🔬 Parameter Estimation Engine",
    "🎯 Response Surface Sweep",
    "📄 Report Generator & Export"
])

# --- TAB 1: DIGITAL TWIN ---
with tab_twin:
    col_chart, col_stats = st.columns([3, 1])
    with col_chart:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Biomass X (g/L)'], name="Biomass (X)", line=dict(color="#0D9488", width=3.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Substrate S (g/L)'], name="Substrate (S)", line=dict(color="#E11D48", width=2.5, dash='dash')), secondary_y=False)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Product P (g/L)'], name="Product (P)", line=dict(color="#0284C7", width=3.5)), secondary_y=True)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Dissolved Oxygen DO (mg/L)'], name="Dissolved O₂", line=dict(color="#8B5CF6", width=2, dash='dot')), secondary_y=True)

        fig.update_layout(template="none", height=480, hovermode="x unified", legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
        grid_style = dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)")
        fig.update_xaxes(title_text="<b>Duration (Hours)</b>", **grid_style)
        fig.update_yaxes(title_text="<b>Biomass & Substrate (g/L)</b>", secondary_y=False, **grid_style)
        fig.update_yaxes(title_text="<b>Product (g/L) / DO (mg/L)</b>", secondary_y=True, **grid_style)

        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("#### Reactor Status")
        st.markdown(f"""
        **Working Volume:** `{final_V:.2f} L`  
        **Substrate Consumed:** `{consumed_S:.1f} g`  
        **Oxygen Limitation:** `{'CRITICAL' if min_DO < 1.0 else 'Optimal'}`  
        """)
        st.download_button(
            "📥 Export Simulation Data",
            sim_df.to_csv(index=False),
            f"BioTwin_Enterprise_Run_{preset_choice.replace(' ', '_')}.csv",
            "text/csv",
            use_container_width=True
        )

# --- TAB 2: PARAMETER ESTIMATION ---
with tab_fitting:
    st.markdown("""
    <div class="section-banner">
        <h4>🔬 Non-Linear Regression & Parameter Fitting Engine</h4>
        <p>Upload laboratory CSVs or paste raw run data to regress kinetic rates using SciPy's Levenberg-Marquardt optimizer.</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_mode = st.radio("Choose Input Method:", ["📄 Upload CSV File", "✍️ Paste Raw Text / Excel Data"], horizontal=True)
    exp_df = None
    
    if input_mode == "📄 Upload CSV File":
        uploaded_file = st.file_uploader("Select Fermentation Run CSV", type=["csv", "txt"])
        if uploaded_file is not None:
            try: exp_df = pd.read_csv(uploaded_file)
            except Exception as e: st.error(f"Error: {e}")
    else:
        raw_text = st.text_area("Paste Tabular Data:", height=140, placeholder="Time,Biomass,Substrate\n0,0.15,30.0\n2,0.32,28.5")
        if raw_text.strip():
            try:
                sep = '\t' if '\t' in raw_text else ','
                exp_df = pd.read_csv(io.StringIO(raw_text.strip()), sep=sep)
            except Exception as e: st.error(f"Error parsing text: {e}")

    if exp_df is None:
        exp_df = pd.DataFrame({
            'Time': [0, 2, 4, 6, 8, 12, 16, 20, 24],
            'Biomass': [0.15, 0.32, 0.78, 1.85, 3.90, 7.20, 8.10, 8.35, 8.40],
            'Substrate': [30.0, 28.5, 25.1, 19.8, 12.0, 3.2, 0.5, 0.1, 0.0]
        })

    st.dataframe(exp_df, height=140, use_container_width=True)

    if 'Time' in exp_df.columns and 'Biomass' in exp_df.columns:
        try:
            t_data = exp_df['Time'].astype(float).values
            x_data = exp_df['Biomass'].astype(float).values
            
            def fit_growth(t, mu_est, x0_est):
                return np.minimum(x0_est * np.exp(mu_est * t), max(x_data))
            
            popt, _ = curve_fit(fit_growth, t_data[:5], x_data[:5], p0=[0.4, x_data[0]])
            fitted_mu = popt[0]
            
            c1, c2 = st.columns(2)
            with c1: st.metric("Regressed μ_max", f"{fitted_mu:.4f} h⁻¹")
            with c2:
                if st.button("⚡ Sync Regressed μ_max to Twin Model", use_container_width=True):
                    st.session_state['fitted_mu'] = float(fitted_mu)
                    st.success("Synchronized!")
                    st.rerun()

            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(x=t_data, y=x_data, mode='markers', name='Lab Data', marker=dict(size=9, color='#E11D48')))
            t_smooth = np.linspace(0, max(t_data), 100)
            fig_fit.add_trace(go.Scatter(x=t_smooth, y=fit_growth(t_smooth, *popt), mode='lines', name='Regressed Fit', line=dict(color='#0D9488', dash='dash')))
            fig_fit.update_layout(template="none", height=320, title="<b>Experimental Points vs Fitted Growth Model</b>")
            st.plotly_chart(fig_fit, use_container_width=True)
        except Exception as e: st.error(f"Regression error: {e}")

# --- TAB 3: RESPONSE SURFACE ---
with tab_sensitivity:
    st.markdown("#### 🎯 Multi-Parameter Yield Optimization Surface")
    res = st.slider("Sweep Grid Resolution", 4, 12, 6)
    mu_range = np.linspace(0.05, 1.0, res)
    S0_range = np.linspace(10.0, 150.0, res)
    matrix = np.zeros((len(S0_range), len(mu_range)))
    
    for i, s_val in enumerate(S0_range):
        for j, m_val in enumerate(mu_range):
            r_df, _ = run_advanced_simulation(X0, s_val, 0.0, 7.0, 1.0, m_val, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, 7.0, 0.15, F_in, S_feed, batch_time)
            matrix[i, j] = r_df['Product P (g/L)'].iloc[-1] if not r_df.empty else 0
            
    fig_sens = go.Figure(data=go.Heatmap(z=matrix, x=np.round(mu_range, 2), y=np.round(S0_range, 1), colorscale='Tealgrn'))
    fig_sens.update_layout(template="none", height=400, xaxis_title="μ_max (1/h)", yaxis_title="Initial Substrate S₀ (g/L)")
    st.plotly_chart(fig_sens, use_container_width=True)

# --- TAB 4: COMPLIANCE REPORT GENERATOR ---
with tab_report:
    st.markdown("#### 📄 Executive Batch Verification Audit Report")
    report_html = f"""
    <div style="background:#FFF; color:#000; padding:24px; border:1px solid #CCC; font-family:sans-serif;">
        <h2 style="color:#0284C7; margin-0;">BioTwin Enterprise Audit Report</h2>
        <p><b>Facility:</b> {st.session_state.get('org', 'Default Facility')} | <b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d')}</p>
        <hr/>
        <ul>
            <li><b>Strain Profile:</b> {preset_choice}</li>
            <li><b>Final Yield (P):</b> {final_P:.2f} g/L</li>
            <li><b>Total Volume (V):</b> {final_V:.2f} L</li>
            <li><b>Volumetric Mass Transfer (kL a):</b> {kla} h⁻¹</li>
        </ul>
    </div>
    """
    st.components.v1.html(report_html, height=220)
    st.download_button("📥 Download HTML Compliance Certificate", report_html, "BioTwin_Batch_Audit.html", "text/html", use_container_width=True)
