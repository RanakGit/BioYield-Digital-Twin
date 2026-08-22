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
# 1. PAGE & THEME-AWARE STYLING CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="BioTwin Pro | Enterprise Bioprocess Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme-Aware SaaS UI Injection
st.markdown("""
<style>
    /* Theme-aware CSS utilizing Streamlit native variables */
    .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Glassmorphism Metric Cards - Works in Light & Dark Mode */
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #06B6D4;
        box-shadow: 0 10px 15px -3px rgba(6, 182, 212, 0.15);
    }
    
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-color);
    }
    
    .metric-sub {
        font-size: 0.75rem;
        color: #10B981;
        font-weight: 600;
    }

    /* Tab Customization Adapts to Light/Dark Mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.7;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #06B6D4 !important;
        opacity: 1.0 !important;
        border-bottom: 3px solid #06B6D4 !important;
        background: rgba(6, 182, 212, 0.08);
    }

    /* Print / PDF Mode Helper */
    @media print {
        section[data-testid="stSidebar"], .stButton, header {
            display: none !important;
        }
        .stApp {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. NUMERICAL KINETIC MODEL ENGINE
# ==========================================
def monod_model(y, t, mu_max, Ks, Y_xs, Y_ps, alpha, beta):
    """
    Generalized Monod-type fermentation ODE System.
    y = [Biomass (X), Substrate (S), Product (P)]
    """
    X, S, P = max(0, y[0]), max(0, y[1]), max(0, y[2])
    
    # Specific growth rate
    mu = mu_max * S / (Ks + S) if S > 0 else 0.0
    
    dXdt = mu * X
    dSdt = -(1.0 / Y_xs) * dXdt if S > 0 else 0.0
    dPdt = (Y_ps * dXdt) + (alpha * dXdt) + (beta * X)
    
    return [dXdt, dSdt, dPdt]

@st.cache_data(ttl=3600)
def run_simulation(X0, S0, P0, mu_max, Ks, Y_xs, Y_ps, alpha, beta, batch_time, n_points=250):
    """Cached solver for speed across high-density slider adjustments."""
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
    except Exception as e:
        return pd.DataFrame(), False


# ==========================================
# 3. PRESET ORGANISM PROFILES
# ==========================================
ORGANISM_PRESETS = {
    "E. coli Recombinant Protein": {
        "mu_max": 0.65, "Ks": 0.20, "Y_xs": 0.50, "Y_ps": 0.15,
        "alpha": 0.10, "beta": 0.02, "X0": 0.1, "S0": 25.0, "batch_time": 18.0
    },
    "S. cerevisiae (Ethanol)": {
        "mu_max": 0.40, "Ks": 0.50, "Y_xs": 0.12, "Y_ps": 0.45,
        "alpha": 0.05, "beta": 0.01, "X0": 0.5, "S0": 100.0, "batch_time": 30.0
    },
    "CHO Cell Line (Monoclonal Ab)": {
        "mu_max": 0.035, "Ks": 0.10, "Y_xs": 0.65, "Y_ps": 0.25,
        "alpha": 0.20, "beta": 0.005, "X0": 0.2, "S0": 15.0, "batch_time": 120.0
    },
    "Custom / Manual Sandbox": {
        "mu_max": 0.45, "Ks": 0.25, "Y_xs": 0.45, "Y_ps": 0.20,
        "alpha": 0.05, "beta": 0.01, "X0": 0.1, "S0": 20.0, "batch_time": 24.0
    }
}


# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## 🧬 BioTwin Pro")
st.sidebar.caption("Enterprise Bioprocess Digital Twin v2.5")
st.sidebar.markdown("---")

preset_choice = st.sidebar.selectbox("⚡ Quick Strain Presets", list(ORGANISM_PRESETS.keys()))
preset = ORGANISM_PRESETS[preset_choice]

st.sidebar.markdown("### ⚙️ Kinetic Parameters")
mu_max = st.sidebar.slider("μ_max (Max Growth Rate, 1/h)", 0.01, 1.50, float(preset["mu_max"]), 0.01)
Ks = st.sidebar.slider("Ks (Affinity Constant, g/L)", 0.01, 2.00, float(preset["Ks"]), 0.01)
Y_xs = st.sidebar.slider("Y_x/s (Yield Biomass/Substrate)", 0.05, 0.90, float(preset["Y_xs"]), 0.01)
Y_ps = st.sidebar.slider("Y_p/s (Yield Product/Substrate)", 0.00, 0.90, float(preset["Y_ps"]), 0.01)

st.sidebar.markdown("### 🧪 Operational Initial Values")
X0 = st.sidebar.number_input("Initial Biomass X₀ (g/L)", 0.01, 50.0, float(preset["X0"]), 0.1)
S0 = st.sidebar.number_input("Initial Substrate S₀ (g/L)", 0.1, 500.0, float(preset["S0"]), 1.0)
batch_time = st.sidebar.slider("Batch Duration (hr)", 4.0, 200.0, float(preset["batch_time"]), 1.0)

# Product Kinetics
alpha = preset["alpha"]
beta = preset["beta"]

# ==========================================
# 5. CORE COMPUTATION & KPI ENGINE
# ==========================================
sim_df, success = run_simulation(X0, S0, 0.0, mu_max, Ks, Y_xs, Y_ps, alpha, beta, batch_time)

if not success or sim_df.empty:
    st.error("⚠️ ODE Numerical Divergence. Please adjust parameter boundaries.")
    st.stop()

# Key Performance Indicators
final_X = sim_df['Biomass X (g/L)'].iloc[-1]
final_P = sim_df['Product P (g/L)'].iloc[-1]
consumed_S = S0 - sim_df['Substrate S (g/L)'].iloc[-1]
vol_prod = final_P / batch_time if batch_time > 0 else 0
overall_yield = (final_P / consumed_S) if consumed_S > 0 else 0


# ==========================================
# 6. MAIN DASHBOARD HEADER & METRICS
# ==========================================
st.title("🧫 Bioprocess Digital Twin Dashboard")
st.caption(f"Active Profile: **{preset_choice}** | Solved via High-Precision ODE Integration")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Final Biomass (X)</div>
        <div class="metric-value">{final_X:.2f} <span style="font-size:1rem;">g/L</span></div>
        <div class="metric-sub">▲ {(final_X - X0):.2f} g/L net growth</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Final Product (P)</div>
        <div class="metric-value" style="color: var(--accent-cyan);">{final_P:.2f} <span style="font-size:1rem;">g/L</span></div>
        <div class="metric-sub">Target Expression</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Volumetric Productivity</div>
        <div class="metric-value">{vol_prod:.3f} <span style="font-size:1rem;">g/L/h</span></div>
        <div class="metric-sub">Space-Time Yield</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Overall Product Yield</div>
        <div class="metric-value">{overall_yield:.3f} <span style="font-size:1rem;">g/g</span></div>
        <div class="metric-sub">P / Substrate Consumed</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# 7. TAB NAVIGATION WORKFLOW
# ==========================================
tab_twin, tab_fitting, tab_sensitivity, tab_report = st.tabs([
    "📈 Dynamic Twin Simulation",
    "🔬 Data Fitting Engine (CSV)",
    "🎯 Sensitivity & Parameter Sweep",
    "📄 Executive Report & Export"
])


# ------------------------------------------
# TAB 1: DYNAMIC SIMULATION
# ------------------------------------------
with tab_twin:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=sim_df['Time (hr)'], y=sim_df['Biomass X (g/L)'],
        name="Biomass (X)", line=dict(color="#10B981", width=3)
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=sim_df['Time (hr)'], y=sim_df['Substrate S (g/L)'],
        name="Substrate (S)", line=dict(color="#F43F5E", width=2, dash='dash')
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=sim_df['Time (hr)'], y=sim_df['Product P (g/L)'],
        name="Product (P)", line=dict(color="#06B6D4", width=3)
    ), secondary_y=True)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.6)',
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Batch Duration (Hours)", gridcolor="#334155")
    fig.update_yaxes(title_text="Biomass & Substrate (g/L)", gridcolor="#334155", secondary_y=False)
    fig.update_yaxes(title_text="Product Concentration (g/L)", gridcolor="#334155", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------
# TAB 2: DATA FITTING ENGINE (HIGH VALUE)
# ------------------------------------------
with tab_fitting:
    st.markdown("### 🔬 Automated Monod Parameter Fitting")
    st.caption("Upload experimental offline batch run data to extract empirical μ_max and Ks constants.")
    
    uploaded_file = st.file_uploader("Upload Experimental Run CSV", type=["csv"])
    
    # Template CSV download helper
    template_df = pd.DataFrame({
        'Time': [0, 2, 4, 6, 8, 12, 16, 20],
        'Biomass': [0.1, 0.25, 0.6, 1.4, 2.8, 5.1, 6.2, 6.4],
        'Substrate': [20.0, 19.1, 17.5, 14.2, 9.8, 2.5, 0.4, 0.1]
    })
    st.download_button("📥 Download CSV Data Template", template_df.to_csv(index=False), "fermentation_data_template.csv", "text/csv")
    
    if uploaded_file is not None:
        try:
            exp_df = pd.read_csv(uploaded_file)
            st.success("✅ CSV Successfully Loaded")
            
            col_t, col_x = st.columns(2)
            with col_t:
                st.dataframe(exp_df.head(6), use_container_width=True)
            
            # Simple Monod Growth Analytical Curve Fitting for Demonstration
            def fit_monod_growth(t, mu_max_fit, X0_fit):
                return X0_fit * np.exp(mu_max_fit * t)
            
            t_data = exp_df['Time'].values
            x_data = exp_df['Biomass'].values
            
            popt, _ = curve_fit(fit_monod_growth, t_data[:5], x_data[:5], p0=[0.4, x_data[0]])
            mu_fit = popt[0]
            
            # R-squared computation
            residuals = x_data[:5] - fit_monod_growth(t_data[:5], *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((x_data[:5] - np.mean(x_data[:5]))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            with col_x:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Exponential μ_max</div>
                    <div class="metric-value" style="color:var(--accent-emerald);">{mu_fit:.3f} h⁻¹</div>
                    <div class="metric-sub">R² Fit Precision: {r_squared:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("⚡ Apply Fitted μ_max to Active Twin"):
                    st.session_state['custom_mu'] = mu_fit
                    st.rerun()

        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")


# ------------------------------------------
# TAB 3: SENSITIVITY SWEEP
# ------------------------------------------
with tab_sensitivity:
    st.markdown("### 🎯 Parameter Sensitivity Heatmap")
    st.caption("Evaluate overall product yield across varying combinations of μ_max and Initial Substrate (S₀).")
    
    mu_range = np.linspace(0.1, 1.0, 8)
    S0_range = np.linspace(10.0, 100.0, 8)
    
    matrix = np.zeros((len(S0_range), len(mu_range)))
    
    for i, s_val in enumerate(S0_range):
        for j, m_val in enumerate(mu_range):
            res, _ = run_simulation(X0, s_val, 0.0, m_val, Ks, Y_xs, Y_ps, alpha, beta, batch_time)
            matrix[i, j] = res['Product P (g/L)'].iloc[-1] if not res.empty else 0
            
    fig_sens = go.Figure(data=go.Heatmap(
        z=matrix, x=np.round(mu_range, 2), y=np.round(S0_range, 1),
        colorscale='Viridis', colorbar=dict(title="Final Product (g/L)")
    ))
    fig_sens.update_layout(
        title="Product Yield Response Surface",
        xaxis_title="μ_max (1/hr)",
        yaxis_title="Initial Substrate S₀ (g/L)",
        template="plotly_dark", height=450
    )
    st.plotly_chart(fig_sens, use_container_width=True)


# ------------------------------------------
# TAB 4: EXECUTIVE REPORT EXPORT
# ------------------------------------------
with tab_report:
    st.markdown("### 📄 Run Summary Export")
    st.caption("Generate a clean, printable executive PDF/HTML report for compliance and lab records.")
    
    report_title = st.text_input("Report Title", f"Bioprocess Run Summary - {preset_choice}")
    engineer_name = st.text_input("Lead Process Engineer", "Dr. Alex Chen")
    
    report_html = f"""
    <div style="background-color: white; color: black; padding: 30px; border-radius: 8px; font-family: Arial, sans-serif;">
        <h2 style="color: #0F172A; margin-bottom: 5px;">{html.escape(report_title)}</h2>
        <p style="color: #64748B; font-size: 0.9rem;">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Author: {html.escape(engineer_name)}</p>
        <hr style="border: 0.5px solid #CBD5E1;">
        
        <h3>1. Kinetic Parameters</h3>
        <ul>
            <li><b>Max Growth Rate (μ_max):</b> {mu_max} h⁻¹</li>
            <li><b>Saturation Constant (Ks):</b> {Ks} g/L</li>
            <li><b>Yield Biomass/Substrate (Y_x/s):</b> {Y_xs} g/g</li>
            <li><b>Yield Product/Substrate (Y_p/s):</b> {Y_ps} g/g</li>
        </ul>
        
        <h3>2. Run Performance Summary</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="background-color: #F1F5F9;">
                <th style="padding: 8px; border: 1px solid #CBD5E1; text-align: left;">Metric</th>
                <th style="padding: 8px; border: 1px solid #CBD5E1; text-align: left;">Value</th>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #CBD5E1;">Final Biomass (X)</td>
                <td style="padding: 8px; border: 1px solid #CBD5E1;">{final_X:.2f} g/L</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #CBD5E1;">Final Product (P)</td>
                <td style="padding: 8px; border: 1px solid #CBD5E1;">{final_P:.2f} g/L</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #CBD5E1;">Volumetric Productivity</td>
                <td style="padding: 8px; border: 1px solid #CBD5E1;">{vol_prod:.3f} g/L/h</td>
            </tr>
        </table>
    </div>
    """
    
    st.components.v1.html(report_html, height=350, scrolling=True)
    
    st.download_button(
        label="📥 Download Printable HTML Executive Report",
        data=report_html,
        file_name=f"Bioprocess_Report_{datetime.date.today()}.html",
        mime="text/html"
    )
