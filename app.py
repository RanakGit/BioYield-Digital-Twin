import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import datetime
import io
import sqlite3
import hashlib
import math

# ==========================================
# 1. PAGE & INDUSTRIAL DESIGN SYSTEM (CSS)
# ==========================================
st.set_page_config(
    page_title="BioTwin Pro Enterprise v4.0",
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

    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .brand-title {
        font-size: 1.4rem;
        font-weight: 700;
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
    }

    .kpi-container {
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
        padding: 1rem 1.25rem;
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
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
# 2. AUTHENTICATION & DATABASE SYSTEM
# ==========================================
def init_user_db():
    conn = sqlite3.connect("biotwin_users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            facility TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username: str, facility: str, password: str) -> tuple[bool, str]:
    if not username.strip() or not password.strip():
        return False, "Username and password cannot be empty."
    conn = sqlite3.connect("biotwin_users.db")
    cursor = conn.cursor()
    try:
        hashed = hash_password(password)
        cursor.execute("INSERT INTO users (username, facility, password_hash) VALUES (?, ?, ?)",
                       (username.strip().lower(), facility.strip(), hashed))
        conn.commit()
        conn.close()
        return True, "Account created successfully! You can now log in."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists. Please choose another or Log In."

def authenticate_user(username: str, password: str) -> tuple[bool, str, str]:
    conn = sqlite3.connect("biotwin_users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT facility, password_hash FROM users WHERE username = ?", (username.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        facility, stored_hash = row
        if stored_hash == hash_password(password):
            return True, facility, "Login successful!"
        return False, "", "Incorrect password."
    return False, "", "Username not found. Please Sign Up first."

init_user_db()

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.markdown("## 🧬 Welcome to BioTwin Pro Enterprise")
    st.caption("Access the Bioprocess Digital Twin & Physical Simulation Engine")
    
    auth_tab1, auth_tab2 = st.tabs(["🔒 Log In", "📝 Sign Up (New Account)"])
    
    with auth_tab1:
        with st.form("login_form"):
            login_user = st.text_input("Username or Email")
            login_pass = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Log In", use_container_width=True)
            if login_btn:
                success, facility, msg = authenticate_user(login_user, login_pass)
                if success:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = login_user.strip().lower()
                    st.session_state['org'] = facility
                    st.success("Authorized! Loading Digital Twin...")
                    st.rerun()
                else:
                    st.error(msg)
                    
    with auth_tab2:
        with st.form("signup_form"):
            new_user = st.text_input("Choose Username / Email")
            new_facility = st.text_input("Facility / Organization Name", "BioProcess Corp")
            new_pass = st.text_input("Create Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            signup_btn = st.form_submit_button("Create Account & Sign Up", use_container_width=True)
            if signup_btn:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match!")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters long.")
                else:
                    created, msg = register_user(new_user, new_facility, new_pass)
                    if created: st.success(msg)
                    else: st.error(msg)
    st.stop()

# ==========================================
# 3. EXTENDED PHYSICS & KINETICS ENGINE
# ==========================================
def temp_factor(T, T_opt, T_min, T_max):
    if T <= T_min or T >= T_max: return 0.0
    num = (T - T_max) * ((T - T_min)**2)
    den = (T_opt - T_min) * ((T_opt - T_min)*(T - T_opt) - (T_opt - T_max)*(T_opt + T_min - 2.0*T))
    return max(0.0, num / den) if den != 0 else 0.0

def ph_factor(pH, pH_opt, width):
    return math.exp(-((pH - pH_opt) / width)**2)

def enterprise_v4_model(y, t, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2,
                        T_curr, T_opt, T_min, T_max, pH_curr, pH_opt, pH_width,
                        alpha_A, beta_A, A_crit,
                        feed_policy, F0, mu_set, DO_thresh, S_feed):
    
    X, S, P, A, DO, V = max(0, y[0]), max(0, y[1]), max(0, y[2]), max(0, y[3]), max(0, y[4]), max(1e-3, y[5])
    
    # 1. Environmental & Byproduct Adjustments
    g_T = temp_factor(T_curr, T_opt, T_min, T_max)
    g_pH = ph_factor(pH_curr, pH_opt, pH_width)
    byproduct_inh = max(0.0, 1.0 - (A / A_crit)) if A_crit > 0 else 1.0
    
    # 2. Kinetic Specific Growth Rate
    mu_base = mu_max * S / (Ks + S + ((S**2) / Ki)) if S > 0 else 0.0
    mu_eff = mu_base * g_T * g_pH * byproduct_inh
    
    # 3. Dynamic Feeding Control Logic
    if feed_policy == "Exponential":
        F_in = F0 * math.exp(mu_set * t)
    elif feed_policy == "DO-Stat":
        DO_pct = (DO / C_star) * 100.0
        F_in = F0 * 3.0 if DO_pct > DO_thresh else F0 * 0.1
    else:  # Constant
        F_in = F0
        
    D = F_in / V
    
    # 4. Differential Equations
    dXdt = (mu_eff - D) * X
    dSdt = D * (S_feed - S) - ((1.0 / Y_xs) * mu_eff * X) if S > 0 else D * (S_feed - S)
    dPdt = -D * P + (Y_ps * mu_eff * X) + (alpha * mu_eff * X) + (beta * X)
    dAdt = -D * A + (alpha_A * mu_eff * X) + (beta_A * X)  # Luedeking-Piret Byproduct
    
    OTR = kla * (C_star - DO)
    OUR = q_O2 * X * 1000.0
    dDOdt = OTR - OUR - (D * DO)
    dVdt = F_in
    
    return [dXdt, dSdt, dPdt, dAdt, dDOdt, dVdt]

@st.cache_data(ttl=3600)
def run_v4_simulation(X0, S0, P0, A0, DO0, V0, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2,
                       T_curr, T_opt, T_min, T_max, pH_curr, pH_opt, pH_width,
                       alpha_A, beta_A, A_crit,
                       feed_policy, F0, mu_set, DO_thresh, S_feed, batch_time, n_points=300):
    t = np.linspace(0, batch_time, n_points)
    try:
        sol = odeint(
            enterprise_v4_model, [X0, S0, P0, A0, DO0, V0], t,
            args=(mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2,
                  T_curr, T_opt, T_min, T_max, pH_curr, pH_opt, pH_width,
                  alpha_A, beta_A, A_crit, feed_policy, F0, mu_set, DO_thresh, S_feed)
        )
        return pd.DataFrame({
            'Time (hr)': t,
            'Biomass X (g/L)': np.clip(sol[:, 0], 0, None),
            'Substrate S (g/L)': np.clip(sol[:, 1], 0, None),
            'Product P (g/L)': np.clip(sol[:, 2], 0, None),
            'Byproduct A (g/L)': np.clip(sol[:, 3], 0, None),
            'Dissolved Oxygen DO (mg/L)': np.clip(sol[:, 4], 0, None),
            'Reactor Volume V (L)': sol[:, 5]
        }), True
    except Exception as e:
        return pd.DataFrame(), False

# --- EXTENDED KALMAN FILTER (EKF) FOR SENSOR TELEMETRY ---
def run_extended_kalman_filter(time_pts, noisy_biomass, noisy_DO=None, q_noise=0.01, r_noise=0.15):
    """
    EKF State Estimator: Filters sensor noise and estimates hidden substrate state S(t)
    """
    n = len(time_pts)
    x_hat = np.zeros((n, 2))  # [Biomass_est, Substrate_est]
    x_hat[0] = [noisy_biomass[0], 25.0]  # Initial state guess
    P_cov = np.eye(2) * 0.1
    Q = np.eye(2) * q_noise
    R = np.eye(1) * r_noise
    
    for k in range(1, n):
        dt = time_pts[k] - time_pts[k-1]
        if dt <= 0:
            dt = 1e-4  # Prevent zero division if timestamps duplicate
            
        # 1. Predict step
        X_prev, S_prev = x_hat[k-1]
        mu_est = 0.45 * S_prev / (0.25 + S_prev) if S_prev > 0 else 0.0
        X_pred = X_prev + (mu_est * X_prev) * dt
        S_pred = max(0.0, S_prev - (1.0 / 0.45) * mu_est * X_prev * dt)
        
        # 2. Linearize Jacobian F
        den = (0.25 + S_prev)**2
        dmu_dS = (0.45 * 0.25 / den) if den > 0 else 0.0
        
        F_mat = np.array([
            [1.0 + mu_est * dt, X_prev * dmu_dS * dt],
            [-(1.0 / 0.45) * mu_est * dt, 1.0 - (1.0 / 0.45) * X_prev * dmu_dS * dt]
        ])
        
        P_pred = F_mat @ P_cov @ F_mat.T + Q
        
        # 3. Update step with noisy biomass measurement z
        z = noisy_biomass[k]
        H = np.array([[1.0, 0.0]])
        y_residual = float(z - (H @ np.array([X_pred, S_pred]))[0])
        
        S_res = (H @ P_pred @ H.T) + R
        K_gain = (P_pred @ H.T) / S_res[0, 0]  # Shape (2, 1)
        
        # Flatten vector operation to avoid dimensional mismatch in Python max()
        correction = (K_gain.flatten() * y_residual)
        updated_state = np.array([X_pred, S_pred]) + correction
        
        x_hat[k] = [max(0.0, float(updated_state[0])), max(0.0, float(updated_state[1]))]
        P_cov = (np.eye(2) - K_gain @ H) @ P_pred
        
    return x_hat[:, 0], x_hat[:, 1]
# ==========================================
# 4. ORGANISM PRESETS & SIDEBAR CONFIG
# ==========================================
ORGANISM_PRESETS = {
    "E. coli Recombinant Protein": {
        "mu_max": 0.65, "Ks": 0.20, "Ki": 150.0, "Y_xs": 0.50, "Y_ps": 0.18,
        "alpha": 0.12, "beta": 0.02, "X0": 0.15, "S0": 30.0, "batch_time": 18.0,
        "kla": 180.0, "F0": 0.05, "S_feed": 200.0, "T_opt": 37.0, "pH_opt": 7.0,
        "alpha_A": 0.15, "beta_A": 0.02, "A_crit": 12.0
    },
    "S. cerevisiae (Bioethanol)": {
        "mu_max": 0.42, "Ks": 0.45, "Ki": 80.0, "Y_xs": 0.14, "Y_ps": 0.46,
        "alpha": 0.05, "beta": 0.01, "X0": 0.50, "S0": 110.0, "batch_time": 32.0,
        "kla": 120.0, "F0": 0.0, "S_feed": 0.0, "T_opt": 30.0, "pH_opt": 5.0,
        "alpha_A": 0.08, "beta_A": 0.01, "A_crit": 85.0
    },
    "CHO Cell Line (mAb Expression)": {
        "mu_max": 0.038, "Ks": 0.12, "Ki": 300.0, "Y_xs": 0.62, "Y_ps": 0.28,
        "alpha": 0.18, "beta": 0.004, "X0": 0.20, "S0": 18.0, "batch_time": 120.0,
        "kla": 25.0, "F0": 0.005, "S_feed": 50.0, "T_opt": 36.5, "pH_opt": 7.2,
        "alpha_A": 0.22, "beta_A": 0.005, "A_crit": 5.0
    }
}  

with st.sidebar:
    st.markdown("### ⚙️ Enterprise Twin Config")
    st.caption(f"Connected Facility: **{st.session_state.get('org', 'Default Facility')}**")
    
    # --- CUSTOM STRAIN & ENTERPRISE OVERRIDE ---
    use_custom_strain = st.checkbox("✍️ Enter Custom Strain / Strain ID", value=False)
    
    if use_custom_strain:
        custom_strain_name = st.text_input("Custom Strain / Organism Name", "Wild-Type E. coli K-12")
        custom_strain_id = st.text_input("Batch / Lot ID", "LOT-2026-0822-X")
        custom_target_product = st.text_input("Target Molecule / Product", "Therapeutic Protein B")
        preset_choice = f"{custom_strain_name} ({custom_strain_id})"
        preset = ORGANISM_PRESETS["E. coli Recombinant Protein"]  # Default baseline parameters
    else:
        preset_choice = st.selectbox("Active Strain Profile", list(ORGANISM_PRESETS.keys()))
        preset = ORGANISM_PRESETS[preset_choice]
        custom_target_product = "Target Biomaterial / Recombinant Product"

    st.markdown("---")
    st.markdown("#### Kinetic Parameters")
    default_mu = st.session_state.get('fitted_mu', float(preset["mu_max"]))
    mu_max = st.slider("μ_max (Max Growth Rate, 1/h)", 0.01, 1.50, default_mu, 0.01)
    Ks = st.slider("Ks (Affinity Constant, g/L)", 0.01, 2.00, float(preset["Ks"]), 0.01)
    Ki = st.number_input("Ki (Haldane Substrate Inh, g/L)", 1.0, 1000.0, float(preset["Ki"]))
    Y_xs = st.slider("Y_x/s (Biomass Yield, g/g)", 0.05, 0.90, float(preset["Y_xs"]), 0.01)
    Y_ps = st.slider("Y_p/s (Product Yield, g/g)", 0.00, 0.90, float(preset["Y_ps"]), 0.01)

    st.markdown("---")
    st.markdown("#### 🌡️ Temperature & pH Controls")
    T_curr = st.slider("Operating Temperature (°C)", 15.0, 45.0, float(preset["T_opt"]), 0.5)
    pH_curr = st.slider("Operating pH", 3.0, 10.0, float(preset["pH_opt"]), 0.1)
    T_opt, T_min, T_max = float(preset["T_opt"]), 15.0, 45.0
    pH_opt, pH_width = float(preset["pH_opt"]), 1.5

    st.markdown("---")
    st.markdown("#### 🧪 Byproduct Kinetics (Luedeking-Piret)")
    alpha_A = st.number_input("Growth Byproduct Rate α_A", 0.0, 2.0, float(preset["alpha_A"]), 0.01)
    beta_A = st.number_input("Non-Growth Byproduct Rate β_A", 0.0, 0.5, float(preset["beta_A"]), 0.001)
    A_crit = st.number_input("Critical Byproduct Limit A_crit (g/L)", 1.0, 200.0, float(preset["A_crit"]), 1.0)

    st.markdown("---")
    st.markdown("#### 🚰 Feeding Strategy Policy")
    feed_policy = st.selectbox("Feed Policy Mode", ["Constant", "Exponential", "DO-Stat"])
    F0 = st.number_input("Base Feed Rate F0 (L/h)", 0.0, 5.0, float(preset["F0"]), 0.001)
    mu_set = st.number_input("Target Exponential μ_set (1/h)", 0.01, 0.80, 0.10, 0.01)
    DO_thresh = st.slider("DO-Stat Trigger Threshold (%)", 10.0, 95.0, 70.0, 5.0)
    S_feed = st.number_input("Inlet Feed Substrate Conc (g/L)", 0.0, 1000.0, float(preset["S_feed"]), 10.0)

    st.markdown("---")
    st.markdown("#### Initial Reactor States")
    X0 = st.number_input("Initial Biomass X₀ (g/L)", 0.01, 50.0, float(preset["X0"]), 0.05)
    S0 = st.number_input("Initial Substrate S₀ (g/L)", 0.1, 500.0, float(preset["S0"]), 1.0)
    A0 = st.number_input("Initial Byproduct A₀ (g/L)", 0.0, 50.0, 0.0, 0.1)
    kla = st.slider("kL a (1/h)", 5.0, 500.0, float(preset["kla"]))
    batch_time = st.slider("Batch Duration (Hours)", 4.0, 200.0, float(preset["batch_time"]), 1.0)
    alpha, beta = preset["alpha"], preset["beta"]
# ==========================================
# 5. SIMULATION & METRICS EXECUTION
# ==========================================
sim_df, success = run_v4_simulation(
    X0, S0, 0.0, A0, 7.0, 1.0, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, 7.0, 0.15,
    T_curr, T_opt, T_min, T_max, pH_curr, pH_opt, pH_width,
    alpha_A, beta_A, A_crit, feed_policy, F0, mu_set, DO_thresh, S_feed, batch_time
)

if not success or sim_df.empty:
    st.error("⚠️ Differential Equation Convergence Error. Adjust kinetic parameters.")
    st.stop()

final_X = sim_df['Biomass X (g/L)'].iloc[-1]
final_P = sim_df['Product P (g/L)'].iloc[-1]
final_A = sim_df['Byproduct A (g/L)'].iloc[-1]
final_V = sim_df['Reactor Volume V (L)'].iloc[-1]
min_DO = sim_df['Dissolved Oxygen DO (mg/L)'].min()
vol_prod = (final_P * final_V) / batch_time if batch_time > 0 else 0

# ==========================================
# 6. HEADER & KPI DASHBOARD
# ==========================================
st.markdown(f"""
<div class="brand-header">
    <div class="brand-title">
        <span>🧬 BioTwin Pro</span>
        <span class="brand-badge">Enterprise v4.0</span>
    </div>
    <div style="font-size: 0.85rem; opacity: 0.85;">
        User: <b>{st.session_state.get('username', 'Operator')}</b> | Strain: <b>{preset_choice}</b> | Strategy: <b>{feed_policy}</b>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card" style="border-left-color: #0D9488;">
        <div class="kpi-label">Final Biomass (X)</div>
        <div class="kpi-value">{final_X:.2f} <span style="font-size:0.85rem;">g/L</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #0284C7;">
        <div class="kpi-label">Target Product (P)</div>
        <div class="kpi-value" style="color: #0284C7;">{final_P:.2f} <span style="font-size:0.85rem;">g/L</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #E11D48;">
        <div class="kpi-label">Toxic Byproduct (A)</div>
        <div class="kpi-value" style="color: {'#E11D48' if final_A > A_crit*0.7 else '#0D9488'};">{final_A:.2f} <span style="font-size:0.85rem;">g/L</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #8B5CF6;">
        <div class="kpi-label">Volumetric Productivity</div>
        <div class="kpi-value">{vol_prod:.3f} <span style="font-size:0.85rem;">g/h</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. WORKFLOW TABS
# ==========================================
tab_twin, tab_ekf, tab_fitting, tab_report = st.tabs([
    "📈 Dynamic Digital Twin",
    "📡 Telemetry & EKF State Estimator",
    "🔬 Parameter Estimation Engine",
    "📄 Report Generator & Audit"
])

# --- TAB 1: DIGITAL TWIN VISUALIZATION ---
with tab_twin:
    col_chart, col_stats = st.columns([3, 1])
    with col_chart:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Biomass X (g/L)'], name="Biomass (X)", line=dict(color="#0D9488", width=3.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Substrate S (g/L)'], name="Substrate (S)", line=dict(color="#D97706", width=2.5, dash='dash')), secondary_y=False)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Byproduct A (g/L)'], name="Byproduct (Acetate/Lactate)", line=dict(color="#E11D48", width=2.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Product P (g/L)'], name="Product (P)", line=dict(color="#0284C7", width=3.5)), secondary_y=True)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Dissolved Oxygen DO (mg/L)'], name="Dissolved O₂", line=dict(color="#8B5CF6", width=2, dash='dot')), secondary_y=True)

        fig.update_layout(template="none", height=500, hovermode="x unified", legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
        grid_style = dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)")
        fig.update_xaxes(title_text="<b>Batch Time (Hours)</b>", **grid_style)
        fig.update_yaxes(title_text="<b>Biomass, Substrate & Byproduct (g/L)</b>", secondary_y=False, **grid_style)
        fig.update_yaxes(title_text="<b>Product (g/L) / DO (mg/L)</b>", secondary_y=True, **grid_style)

        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("#### Environmental Stress")
        gamma_T_val = temp_factor(T_curr, T_opt, T_min, T_max)
        gamma_pH_val = ph_factor(pH_curr, pH_opt, pH_width)
        st.progress(gamma_T_val, text=f"Temperature Efficiency: {gamma_T_val*100:.1f}%")
        st.progress(gamma_pH_val, text=f"pH Efficiency: {gamma_pH_val*100:.1f}%")
        
        st.markdown("#### Reactor Status")
        st.markdown(f"""
        **Working Volume:** `{final_V:.2f} L`  
        **Inhibition Stress:** `{(1 - final_A/A_crit)*100:.1f}% capacity`  
        **Min DO Warning:** `{'CRITICAL' if min_DO < 1.0 else 'Optimal'}`  
        """)
        st.download_button("📥 Export Simulation Data (CSV)", sim_df.to_csv(index=False), "BioTwin_v4_Run.csv", "text/csv", use_container_width=True)

# --- TAB 2: TELEMETRY & EXTENDED KALMAN FILTER ---
with tab_ekf:
    st.markdown("""
    <div class="section-banner">
        <h4>📡 Live Sensor Telemetry & Extended Kalman Filter (EKF) State Estimator</h4>
        <p>Incorporate live noisy optical density (OD) sensor feeds. The EKF filters signal noise and reconstructs unmeasured hidden variables (Substrate Concentration S).</p>
    </div>
    """, unsafe_allow_html=True)
    
    t_sensor = sim_df['Time (hr)'].values
    true_x = sim_df['Biomass X (g/L)'].values
    
    # Generate synthetic noisy sensor readings
    np.random.seed(42)
    noisy_sensor_x = np.maximum(0, true_x + np.random.normal(0, 0.35, len(true_x)))
    
    # Run Extended Kalman Filter
    est_x, est_s = run_extended_kalman_filter(t_sensor, noisy_sensor_x, noisy_DO=None)
    
    fig_ekf = make_subplots(rows=1, cols=2, subplot_titles=("Biomass State Estimation (Filtering Sensor Noise)", "Reconstructed Substrate Concentration S(t)"))
    
    fig_ekf.add_trace(go.Scatter(x=t_sensor, y=noisy_sensor_x, mode='markers', name='Noisy OD Sensor Stream', marker=dict(color='#E11D48', size=5, opacity=0.6)), row=1, col=1)
    fig_ekf.add_trace(go.Scatter(x=t_sensor, y=est_x, mode='lines', name='EKF Filtered State', line=dict(color='#0D9488', width=3)), row=1, col=1)
    
    fig_ekf.add_trace(go.Scatter(x=t_sensor, y=sim_df['Substrate S (g/L)'], mode='lines', name='True Substrate', line=dict(color='#D97706', dash='dash')), row=1, col=2)
    fig_ekf.add_trace(go.Scatter(x=t_sensor, y=est_s, mode='lines', name='EKF Reconstructed Substrate S(t)', line=dict(color='#0284C7', width=3)), row=1, col=2)
    
    fig_ekf.update_layout(template="none", height=420)
    st.plotly_chart(fig_ekf, use_container_width=True)

# --- TAB 3: PARAMETER ESTIMATION ---
with tab_fitting:
    st.markdown("#### 🔬 Non-Linear Regression & Parameter Fitting Engine")
    input_mode = st.radio("Choose Input Method:", ["📄 Upload CSV File", "✍️ Paste Raw Text Data"], horizontal=True)
    exp_df = None
    
    if input_mode == "📄 Upload CSV File":
        uploaded_file = st.file_uploader("Select Fermentation Run CSV", type=["csv", "txt"])
        if uploaded_file is not None:
            try: exp_df = pd.read_csv(uploaded_file)
            except Exception as e: st.error(f"Error reading file: {e}")
    else:
        raw_text = st.text_area("Paste Tabular Data:", height=120, placeholder="Time,Biomass,Substrate\n0,0.15,30.0\n2,0.32,28.5")
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

    st.dataframe(exp_df, height=130, use_container_width=True)

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
            fig_fit.add_trace(go.Scatter(x=t_data, y=x_data, mode='markers', name='Lab Points', marker=dict(size=8, color='#E11D48')))
            t_smooth = np.linspace(0, max(t_data), 100)
            fig_fit.add_trace(go.Scatter(x=t_smooth, y=fit_growth(t_smooth, *popt), mode='lines', name='Regressed Fit', line=dict(color='#0D9488', dash='dash')))
            fig_fit.update_layout(template="none", height=320)
            st.plotly_chart(fig_fit, use_container_width=True)
        except Exception as e: st.error(f"Regression error: {e}")

# --- TAB 4: REPORT GENERATOR & AUDIT ---
with tab_report:
    st.markdown("#### 📄 Executive Batch Verification Audit Report")
    
    # Capture live dynamic state values
    current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    user_name = st.session_state.get('username', 'Operator')
    facility_name = st.session_state.get('org', 'Default Facility')
    
    # Calculate live process metrics
    final_time = sim_df['Time (hr)'].iloc[-1]
    peak_x = sim_df['Biomass X (g/L)'].max()
    final_s = sim_df['Substrate S (g/L)'].iloc[-1]
    overall_yield_px = (final_P / (final_X - X0)) if (final_X - X0) > 0 else 0.0
    
    report_html = f"""
    <div style="background:#FFFFFF; color:#1E293B; padding:28px; border:1px solid #E2E8F0; border-radius:12px; font-family:'Segoe UI', Roboto, Helvetica, sans-serif; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #0284C7; padding-bottom:12px; margin-bottom:20px;">
            <div>
                <h2 style="color:#0284C7; margin:0; font-size:22px; font-weight:700;">BioTwin Enterprise Audit Certificate</h2>
                <span style="font-size:12px; color:#64748B;">Automated Digital Twin Batch Run Record</span>
            </div>
            <div style="text-align:right;">
                <span style="background:#0284C7; color:#FFF; font-size:11px; font-weight:700; padding:4px 8px; border-radius:4px;">VERIFIED RUN</span>
            </div>
        </div>
        
        <table style="width:100%; font-size:13px; margin-bottom:20px; border-collapse:collapse;">
            <tr>
                <td style="padding:4px 0;"><b>Facility:</b> {facility_name}</td>
                <td style="padding:4px 0;"><b>Operator:</b> {user_name}</td>
            </tr>
            <tr>
                <td style="padding:4px 0;"><b>Active Strain/Profile:</b> {preset_choice}</td>
                <td style="padding:4px 0;"><b>Execution Time:</b> {current_time_str}</td>
            </tr>
        </table>

        <h3 style="font-size:14px; color:#0D9488; border-bottom:1px solid #CBD5E1; padding-bottom:4px; margin-top:16px;">1. Operating Environment & Feeding Strategy</h3>
        <table style="width:100%; font-size:12px; border-collapse:collapse; margin-bottom:16px;">
            <tr style="background:#F8FAFC;">
                <td style="padding:6px;"><b>Temperature (Actual / Target):</b> {T_curr}°C / {T_opt}°C</td>
                <td style="padding:6px;"><b>pH (Actual / Target):</b> {pH_curr} / {pH_opt}</td>
            </tr>
            <tr>
                <td style="padding:6px;"><b>Feeding Strategy:</b> {feed_policy}</td>
                <td style="padding:6px;"><b>Base Feed Rate (F₀):</b> {F0} L/h</td>
            </tr>
            <tr style="background:#F8FAFC;">
                <td style="padding:6px;"><b>Feed Substrate Conc (S_feed):</b> {S_feed} g/L</td>
                <td style="padding:6px;"><b>Volumetric Oxygen Transfer (k_L a):</b> {kla} h⁻¹</td>
            </tr>
        </table>

        <h3 style="font-size:14px; color:#0D9488; border-bottom:1px solid #CBD5E1; padding-bottom:4px; margin-top:16px;">2. Active Kinetic Model Parameters</h3>
        <table style="width:100%; font-size:12px; border-collapse:collapse; margin-bottom:16px;">
            <tr style="background:#F8FAFC;">
                <td style="padding:6px;"><b>Max Growth Rate (μ_max):</b> {mu_max:.4f} h⁻¹</td>
                <td style="padding:6px;"><b>Substrate Affinity (K_s):</b> {Ks:.3f} g/L</td>
            </tr>
            <tr>
                <td style="padding:6px;"><b>Haldane Inhibition (K_i):</b> {Ki:.1f} g/L</td>
                <td style="padding:6px;"><b>Biomass Yield (Y_x/s):</b> {Y_xs:.3f} g/g</td>
            </tr>
        </table>

        <h3 style="font-size:14px; color:#0D9488; border-bottom:1px solid #CBD5E1; padding-bottom:4px; margin-top:16px;">3. Simulated Performance Metrics & Yields</h3>
        <table style="width:100%; font-size:12px; border-collapse:collapse; margin-bottom:16px;">
            <tr style="background:#F8FAFC;">
                <td style="padding:6px;"><b>Batch Duration:</b> {final_time:.1f} Hours</td>
                <td style="padding:6px;"><b>Final Working Volume:</b> {final_V:.2f} L</td>
            </tr>
            <tr>
                <td style="padding:6px;"><b>Peak Biomass (X_max):</b> {peak_x:.2f} g/L</td>
                <td style="padding:6px;"><b>Residual Substrate (S_final):</b> {final_s:.2f} g/L</td>
            </tr>
            <tr style="background:#F8FAFC;">
                <td style="padding:6px;"><b>Target Product Titer (P):</b> <b style="color:#0284C7;">{final_P:.2f} g/L</b></td>
                <td style="padding:6px;"><b>Product / Biomass Yield (Y_p/x):</b> {overall_yield_px:.3f} g/g</td>
            </tr>
            <tr>
                <td style="padding:6px;"><b>Toxic Byproduct (A):</b> <b style="color:{'#E11D48' if final_A >= A_crit*0.8 else '#0D9488'};">{final_A:.2f} g/L</b> (Crit: {A_crit} g/L)</td>
                <td style="padding:6px;"><b>Volumetric Productivity:</b> <b>{vol_prod:.3f} g/h</b></td>
            </tr>
        </table>

        <div style="font-size:10px; color:#94A3B8; margin-top:20px; text-align:center; border-top:1px solid #E2E8F0; padding-top:8px;">
            Generated by BioTwin Pro v4.0 Engine • Confidential Bioprocess Audit Report
        </div>
    </div>
    """
    
    st.components.v1.html(report_html, height=520, scrolling=True)
    st.download_button(
        "📥 Download Verified Audit Certificate (HTML)",
        report_html,
        f"BioTwin_Batch_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        "text/html",
        use_container_width=True
    )
