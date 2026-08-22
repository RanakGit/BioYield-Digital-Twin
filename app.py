import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import datetime
import html

# ==========================================
# 1. PAGE & INDUSTRIAL DESIGN SYSTEM (CSS)
# ==========================================
st.set_page_config(
    page_title="BioTwin Pro | Enterprise Bioprocess Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Industrial SaaS Styling System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container & Layout */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Top Brand Navigation Header */
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

    /* Dashboard Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-left: 4px solid #0284C7;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
        border-color: rgba(2, 132, 199, 0.5);
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-color);
        opacity: 0.65;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        color: var(--text-color);
        line-height: 1.1;
    }

    .kpi-unit {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.7;
        margin-left: 0.2rem;
    }

    .kpi-sub {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    .sub-positive { color: #0D9488; }
    .sub-neutral { color: #0284C7; }
    .sub-warning { color: #D97706; }

    /* Custom Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px 8px 0 0;
        padding: 0 18px;
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.7;
        background: transparent;
        border: 1px solid transparent;
        transition: all 0.15s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #0284C7 !important;
        opacity: 1.0 !important;
        border-bottom: 3px solid #0284C7 !important;
        background: rgba(2, 132, 199, 0.06);
    }

    /* Custom Section Banner */
    .section-banner {
        background: rgba(128, 128, 128, 0.05);
        border-left: 3px solid #0D9488;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.25rem;
    }

    .section-banner h4 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .section-banner p {
        margin: 0.2rem 0 0 0;
        font-size: 0.82rem;
        opacity: 0.75;
    }

    /* Print / Export helper */
    @media print {
        section[data-testid="stSidebar"], .stButton, header {
            display: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. ODE NUMERICAL ENGINE
# ==========================================
def monod_model(y, t, mu_max, Ks, Y_xs, Y_ps, alpha, beta):
    X, S, P = max(0, y[0]), max(0, y[1]), max(0, y[2])
    mu = mu_max * S / (Ks + S) if S > 0 else 0.0
    
    dXdt = mu * X
    dSdt = -(1.0 / Y_xs) * dXdt if S > 0 else 0.0
    dPdt = (Y_ps * dXdt) + (alpha * dXdt) + (beta * X)
    
    return [dXdt, dSdt, dPdt]

@st.cache_data(ttl=3600)
def run_simulation(X0, S0, P0, mu_max, Ks, Y_xs, Y_ps, alpha, beta, batch_time, n_points=300):
    t = np.linspace(0, batch_time, n_points)
    try:
        sol = odeint(monod_model, [X0, S0, P0], t, args=(mu_max, Ks, Y_xs, Y_ps, alpha, beta))
        df = pd.DataFrame({
            'Time (hr)': t,
            'Biomass X (g/L)': np.clip(sol[:, 0], 0, None),
            'Substrate S (g/L)': np.clip(sol[:, 1], 0, None),
            'Product P (g/L)': np.clip(sol[:, 2], 0, None)
        })
        return df, True
    except Exception:
        return pd.DataFrame(), False


# ==========================================
# 3. ORGANISM PRESETS & CONSTANTS
# ==========================================
ORGANISM_PRESETS = {
    "E. coli Recombinant Protein": {
        "mu_max": 0.65, "Ks": 0.20, "Y_xs": 0.50, "Y_ps": 0.18,
        "alpha": 0.12, "beta": 0.02, "X0": 0.15, "S0": 30.0, "batch_time": 18.0
    },
    "S. cerevisiae (Bioethanol)": {
        "mu_max": 0.42, "Ks": 0.45, "Y_xs": 0.14, "Y_ps": 0.46,
        "alpha": 0.05, "beta": 0.01, "X0": 0.50, "S0": 110.0, "batch_time": 32.0
    },
    "CHO Cell Line (mAb Expression)": {
        "mu_max": 0.038, "Ks": 0.12, "Y_xs": 0.62, "Y_ps": 0.28,
        "alpha": 0.18, "beta": 0.004, "X0": 0.20, "S0": 18.0, "batch_time": 120.0
    },
    "Custom / Lab Sandbox": {
        "mu_max": 0.45, "Ks": 0.25, "Y_xs": 0.45, "Y_ps": 0.20,
        "alpha": 0.05, "beta": 0.01, "X0": 0.10, "S0": 25.0, "batch_time": 24.0
    }
}


# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Twin Configuration")
    preset_choice = st.selectbox("Active Strain Preset", list(ORGANISM_PRESETS.keys()))
    preset = ORGANISM_PRESETS[preset_choice]

    st.markdown("---")
    st.markdown("#### Kinetic Parameters")
    
    # Check session state overrides from fitting engine
    default_mu = st.session_state.get('fitted_mu', float(preset["mu_max"]))
    
    mu_max = st.slider("μ_max (Max Specific Growth Rate, 1/h)", 0.01, 1.50, default_mu, 0.01)
    Ks = st.slider("Ks (Substrate Affinity Constant, g/L)", 0.01, 2.00, float(preset["Ks"]), 0.01)
    Y_xs = st.slider("Y_x/s (Biomass Yield Coefficient, g/g)", 0.05, 0.90, float(preset["Y_xs"]), 0.01)
    Y_ps = st.slider("Y_p/s (Product Yield Coefficient, g/g)", 0.00, 0.90, float(preset["Y_ps"]), 0.01)

    st.markdown("---")
    st.markdown("#### Initial Reactor States")
    X0 = st.number_input("Initial Biomass X₀ (g/L)", 0.01, 50.0, float(preset["X0"]), 0.05)
    S0 = st.number_input("Initial Substrate S₀ (g/L)", 0.1, 500.0, float(preset["S0"]), 1.0)
    batch_time = st.slider("Batch Duration (Hours)", 4.0, 200.0, float(preset["batch_time"]), 1.0)

    alpha = preset["alpha"]
    beta = preset["beta"]

# ==========================================
# 5. EXECUTE COMPUTATION & KPI ENGINE
# ==========================================
sim_df, success = run_simulation(X0, S0, 0.0, mu_max, Ks, Y_xs, Y_ps, alpha, beta, batch_time)

if not success or sim_df.empty:
    st.error("⚠️ ODE Solver Numerical Instability. Check kinetic constant boundaries.")
    st.stop()

# Compute Metrics
final_X = sim_df['Biomass X (g/L)'].iloc[-1]
final_P = sim_df['Product P (g/L)'].iloc[-1]
consumed_S = S0 - sim_df['Substrate S (g/L)'].iloc[-1]
vol_prod = final_P / batch_time if batch_time > 0 else 0
overall_yield = (final_P / consumed_S) if consumed_S > 0 else 0
specific_growth = mu_max * (sim_df['Substrate S (g/L)'].iloc[-1] / (Ks + sim_df['Substrate S (g/L)'].iloc[-1]))

# ==========================================
# 6. HEADER & DASHBOARD METRICS
# ==========================================
st.markdown(f"""
<div class="brand-header">
    <div class="brand-title">
        <span>🧬 BioTwin Pro</span>
        <span class="brand-badge">Enterprise v2.5</span>
    </div>
    <div style="font-size: 0.85rem; opacity: 0.8;">
        Strain Profile: <b>{preset_choice}</b> | Status: <span style="color:#0D9488; font-weight:700;">● Engine Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI Cards Display
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card" style="border-left-color: #0D9488;">
        <div class="kpi-label">Final Biomass Concentration (X)</div>
        <div class="kpi-value">{final_X:.2f}<span class="kpi-unit">g/L</span></div>
        <div class="kpi-sub sub-positive">▲ +{(final_X - X0):.2f} g/L net growth</div>
    </div>
    <div class="kpi-card" style="border-left-color: #0284C7;">
        <div class="kpi-label">Target Product Yield (P)</div>
        <div class="kpi-value" style="color: #0284C7;">{final_P:.2f}<span class="kpi-unit">g/L</span></div>
        <div class="kpi-sub sub-neutral">● Expression Target</div>
    </div>
    <div class="kpi-card" style="border-left-color: #D97706;">
        <div class="kpi-label">Volumetric Productivity</div>
        <div class="kpi-value">{vol_prod:.3f}<span class="kpi-unit">g/L/h</span></div>
        <div class="kpi-sub sub-warning">★ Space-Time Yield</div>
    </div>
    <div class="kpi-card" style="border-left-color: #8B5CF6;">
        <div class="kpi-label">Overall Product Yield (Y_p/s)</div>
        <div class="kpi-value">{overall_yield:.3f}<span class="kpi-unit">g/g</span></div>
        <div class="kpi-sub sub-neutral">◆ Substrate Conversion Efficiency</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 7. TAB WORKFLOW SYSTEM
# ==========================================
tab_twin, tab_fitting, tab_sensitivity, tab_report = st.tabs([
    "📈 Dynamic Digital Twin",
    "🔬 Parameter Estimation Engine",
    "🎯 Response Surface Sweep",
    "📄 Report Generator & Export"
])


# ------------------------------------------
# TAB 1: DYNAMIC DIGITAL TWIN
# ------------------------------------------
with tab_twin:
    col_chart, col_stats = st.columns([3, 1])
    
    with col_chart:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Biomass (X) - Emerald
        fig.add_trace(go.Scatter(
            x=sim_df['Time (hr)'], y=sim_df['Biomass X (g/L)'],
            name="Biomass (X)",
            line=dict(color="#0D9488", width=3.5)
        ), secondary_y=False)
        
        # Substrate (S) - Crimson Dashed
        fig.add_trace(go.Scatter(
            x=sim_df['Time (hr)'], y=sim_df['Substrate S (g/L)'],
            name="Substrate (S)",
            line=dict(color="#E11D48", width=2.5, dash='dash')
        ), secondary_y=False)
        
        # Product (P) - Deep Cyan
        fig.add_trace(go.Scatter(
            x=sim_df['Time (hr)'], y=sim_df['Product P (g/L)'],
            name="Product (P)",
            line=dict(color="#0284C7", width=3.5)
        ), secondary_y=True)

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=480,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,
                font=dict(size=12, family="Inter")
            ),
            hovermode="x unified"
        )

        grid_color = "rgba(128, 128, 128, 0.15)"
        fig.update_xaxes(title_text="<b>Batch Duration (Hours)</b>", showgrid=True, gridcolor=grid_color)
        fig.update_yaxes(title_text="<b>Biomass & Substrate (g/L)</b>", secondary_y=False, showgrid=True, gridcolor=grid_color)
        fig.update_yaxes(title_text="<b>Product Concentration (g/L)</b>", secondary_y=True, showgrid=True, gridcolor=grid_color)

        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("#### Reactor Analytics")
        st.markdown(f"""
        **Substrate Conversion:**  
        `{((consumed_S / S0) * 100):.1f}%`
        
        **Final Specific Growth Rate:**  
        `{specific_growth:.4f} h⁻¹`
        
        **Peak Biomass Productivity:**  
        `{((final_X - X0) / batch_time):.3f} g/L/h`
        """)
        st.markdown("---")
        st.download_button(
            "📥 Export Run CSV Data",
            sim_df.to_csv(index=False),
            f"BioTwin_Run_{preset_choice.replace(' ', '_')}.csv",
            "text/csv",
            use_container_width=True
        )


# ------------------------------------------
# TAB 2: PARAMETER ESTIMATION ENGINE
# ------------------------------------------
with tab_fitting:
    st.markdown("""
    <div class="section-banner">
        <h4>🔬 Non-Linear Regression & Parameter Fitting Engine</h4>
        <p>Upload offline laboratory sample readings (.csv) to automatically estimate empirical kinetic constants using SciPy curve fitting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_upload, col_preview = st.columns([1, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader("Select Experimental Fermentation Run CSV", type=["csv"])
        
        # Template Generator
        demo_t = np.array([0, 2, 4, 6, 8, 12, 16, 20, 24])
        demo_x = np.array([0.15, 0.32, 0.78, 1.85, 3.90, 7.20, 8.10, 8.35, 8.40])
        demo_s = np.array([30.0, 28.5, 25.1, 19.8, 12.0, 3.2, 0.5, 0.1, 0.0])
        demo_df = pd.DataFrame({'Time': demo_t, 'Biomass': demo_x, 'Substrate': demo_s})
        
        st.download_button("📥 Download Laboratory Data Template", demo_df.to_csv(index=False), "lab_run_template.csv", "text/csv")

    with col_preview:
        if uploaded_file is not None:
            exp_df = pd.read_csv(uploaded_file)
            st.markdown("##### Uploaded Sample Data Preview")
            st.dataframe(exp_df, height=180, use_container_width=True)
        else:
            exp_df = demo_df
            st.info("ℹ️ Showing default demo lab dataset. Upload your own CSV above.")

    # Execute Non-Linear Fitting Engine
    st.markdown("---")
   # ------------------------------------------
# REGRESSION COMPUTATION ENGINE (FIXED)
# ------------------------------------------
st.markdown("### 📊 Non-Linear Regression Results")

if 'Time' in exp_df.columns and 'Biomass' in exp_df.columns:
    try:
        t_data = exp_df['Time'].astype(float).values
        x_data = exp_df['Biomass'].astype(float).values
        
        # Capped growth model to prevent runaway exponential curves
        def fit_growth(t, mu_est, x0_est):
            # Capped at realistic max biomass (~8.5 g/L) based on data upper bound
            x_max = max(x_data) if len(x_data) > 0 else 10.0
            raw_exp = x0_est * np.exp(mu_est * t)
            return np.minimum(raw_exp, x_max)
        
        # Fit on exponential phase points (before stationary phase)
        fit_idx = min(5, len(t_data))
        popt, _ = curve_fit(fit_growth, t_data[:fit_idx], x_data[:fit_idx], p0=[0.4, x_data[0]])
        fitted_mu = popt[0]
        
        # R² Calculation
        residuals = x_data[:fit_idx] - fit_growth(t_data[:fit_idx], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((x_data[:fit_idx] - np.mean(x_data[:fit_idx]))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Estimated Growth Rate (μ_max)", f"{fitted_mu:.4f} h⁻¹")
        with c2:
            st.metric("Regression Fit Precision (R²)", f"{r_squared:.4f}")
        with c3:
            if st.button("⚡ Apply Fitted μ_max to Twin", use_container_width=True):
                st.session_state['fitted_mu'] = float(fitted_mu)
                st.success(f"Updated μ_max to {fitted_mu:.4f} h⁻¹!")
                st.rerun()

        # Visual Fit Plot with Forced High-Contrast Styling
        fig_fit = go.Figure()
        
        fig_fit.add_trace(go.Scatter(
            x=t_data, y=x_data, 
            mode='markers', 
            name='Lab Samples (Biomass)', 
            marker=dict(size=10, color='#E11D48')
        ))
        
        t_smooth = np.linspace(0, max(t_data), 100)
        fig_fit.add_trace(go.Scatter(
            x=t_smooth, y=fit_growth(t_smooth, *popt), 
            mode='lines', 
            name='Regressed Monod Growth Curve', 
            line=dict(color='#0D9488', width=2.5, dash='dash')
        ))
        
        # Explicit Font & Axis Color Overrides for High Legibility
        fig_fit.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=380,
            title=dict(
                text="<b>Experimental Biomass Points vs. Regression Model</b>",
                font=dict(color="#0F172A", size=15)
            ),
            font=dict(color="#0F172A", family="Inter"),
            legend=dict(
                font=dict(color="#0F172A", size=12),
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            )
        )
        
        grid_style = dict(showgrid=True, gridcolor="rgba(100, 100, 100, 0.15)", tickfont=dict(color="#0F172A"))
        fig_fit.update_xaxes(title=dict(text="<b>Time (Hours)</b>", font=dict(color="#0F172A")), **grid_style)
        fig_fit.update_yaxes(title=dict(text="<b>Biomass Concentration (g/L)</b>", font=dict(color="#0F172A")), **grid_style)

        st.plotly_chart(fig_fit, use_container_width=True)

    except Exception as e:
        st.error(f"Regression error: {str(e)}")
else:
    st.warning("⚠️ Column headers missing! Data must contain 'Time' and 'Biomass' headers.")


# ------------------------------------------
# TAB 3: RESPONSE SURFACE SWEEP
# ------------------------------------------
with tab_sensitivity:
    st.markdown("""
    <div class="section-banner">
        <h4>🎯 Multi-Parameter Sensitivity Analysis</h4>
        <p>Explore yield optimization surfaces across combinations of growth rate (μ_max) and initial substrate feeding (S₀).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_sweep_controls, col_sweep_plot = st.columns([1, 2.5])
    
    with col_sweep_controls:
        st.markdown("##### Sweep Resolution Settings")
        resolution = st.slider("Grid Resolution (Points)", 5, 15, 8)
        s0_max_sweep = st.number_input("Max Substrate S₀ for Sweep (g/L)", 20.0, 300.0, 100.0)
        
    with col_sweep_plot:
        mu_range = np.linspace(0.05, 1.0, resolution)
        S0_range = np.linspace(10.0, s0_max_sweep, resolution)
        matrix = np.zeros((len(S0_range), len(mu_range)))
        
        for i, s_val in enumerate(S0_range):
            for j, m_val in enumerate(mu_range):
                res, _ = run_simulation(X0, s_val, 0.0, m_val, Ks, Y_xs, Y_ps, alpha, beta, batch_time)
                matrix[i, j] = res['Product P (g/L)'].iloc[-1] if not res.empty else 0
                
        fig_sens = go.Figure(data=go.Heatmap(
            z=matrix, 
            x=np.round(mu_range, 2), 
            y=np.round(S0_range, 1),
            colorscale='Tealgrn',
            colorbar=dict(title="<b>Final Product (g/L)</b>")
        ))

        fig_sens.update_layout(
            title="Product Yield Surface Contour",
            xaxis_title="<b>μ_max Growth Rate (1/hr)</b>",
            yaxis_title="<b>Initial Substrate S₀ (g/L)</b>",
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=440
        )
        st.plotly_chart(fig_sens, use_container_width=True)


# ------------------------------------------
# TAB 4: EXECUTIVE REPORT GENERATOR
# ------------------------------------------
with tab_report:
    st.markdown("""
    <div class="section-banner">
        <h4>📄 Automated Compliance & Batch Report Generator</h4>
        <p>Generate clean, printable executive HTML reports formatted for lab records and team reviews.</p>
    </div>
    """, unsafe_allow_html=True)
    
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        report_title = st.text_input("Run Identifier / Report Title", f"Fermentation Batch Audit - {preset_choice}")
    with r_col2:
        engineer_name = st.text_input("Lead Process Engineer", "Dr. Alex Chen")

    report_html = f"""
    <div style="background-color: #FFFFFF; color: #0F172A; padding: 32px; border-radius: 10px; border: 1px solid #CBD5E1; font-family: Arial, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #0284C7; padding-bottom: 12px; margin-bottom: 20px;">
            <div>
                <h2 style="margin: 0; color: #0284C7;">{html.escape(report_title)}</h2>
                <p style="margin: 4px 0 0 0; color: #64748B; font-size: 0.85rem;">BioTwin Pro Enterprise Verification Audit</p>
            </div>
            <div style="text-align: right; font-size: 0.8rem; color: #64748B;">
                <div><b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div><b>Engineer:</b> {html.escape(engineer_name)}</div>
            </div>
        </div>

        <h3 style="color: #0F172A; font-size: 1.05rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 6px;">1. Process Performance Summary</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem;">
            <tr style="background-color: #F8FAFC;">
                <th style="padding: 10px; border: 1px solid #E2E8F0; text-align: left;">Metric Description</th>
                <th style="padding: 10px; border: 1px solid #E2E8F0; text-align: left;">Simulated Value</th>
                <th style="padding: 10px; border: 1px solid #E2E8F0; text-align: left;">Unit</th>
            </tr>
            <tr>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0;">Final Biomass (X)</td>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0; font-weight: bold;">{final_X:.2f}</td>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0;">g/L</td>
            </tr>
            <tr>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0;">Final Target Product (P)</td>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0; font-weight: bold; color: #0284C7;">{final_P:.2f}</td>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0;">g/L</td>
            </tr>
            <tr>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0;">Volumetric Productivity</td>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0; font-weight: bold;">{vol_prod:.3f}</td>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0;">g/L/h</td>
            </tr>
            <tr>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0;">Overall Yield (Y_p/s)</td>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0; font-weight: bold;">{overall_yield:.3f}</td>
                <td style="padding: 8px 10px; border: 1px solid #E2E8F0;">g/g</td>
            </tr>
        </table>

        <h3 style="color: #0F172A; font-size: 1.05rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; margin-top: 24px;">2. Applied Kinetic Parameters</h3>
        <ul style="font-size: 0.88rem; line-height: 1.6; color: #334155;">
            <li><b>Max Growth Rate (μ_max):</b> {mu_max:.4f} h⁻¹</li>
            <li><b>Substrate Affinity Constant (Ks):</b> {Ks:.2f} g/L</li>
            <li><b>Biomass Yield (Y_x/s):</b> {Y_xs:.2f} g/g</li>
            <li><b>Product Yield (Y_p/s):</b> {Y_ps:.2f} g/g</li>
            <li><b>Total Batch Duration:</b> {batch_time:.1f} Hours</li>
        </ul>
    </div>
    """

    st.components.v1.html(report_html, height=380, scrolling=True)
    
    st.download_button(
        label="📥 Export Downloadable Batch Summary HTML Report",
        data=report_html,
        file_name=f"BioTwin_Report_{datetime.date.today()}.html",
        mime="text/html",
        use_container_width=True
    )
