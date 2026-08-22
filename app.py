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
import hmac
import secrets
import math

# ==========================================
# 1. ENTERPRISE UI/UX DESIGN SYSTEM (CSS)
# ==========================================
st.set_page_config(
    page_title="FermentIQ | Enterprise Bioprocess Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    
    .brand-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.02em;
    }

    .brand-badge {
        background: linear-gradient(135deg, #0284C7, #0D9488);
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 30px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-container {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 1.75rem;
    }

    .kpi-card {
        flex: 1;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0284C7;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.5rem;
    }

    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #0F172A;
        letter-spacing: -0.03em;
    }

    .section-banner {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0D9488;
        padding: 1rem 1.25rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 1.5rem;
    }
    
    .section-banner h4 {
        margin: 0 0 0.25rem 0;
        font-size: 1.05rem;
        color: #0F172A;
        font-weight: 700;
    }

    .section-banner p {
        margin: 0;
        font-size: 0.85rem;
        color: #64748B;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #CBD5E1;
        background-color: #FFFFFF;
        color: #0F172A;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        border-color: #0284C7;
        color: #0284C7;
        box-shadow: 0 4px 6px -1px rgba(0, 132, 199, 0.1);
    }

    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    /* --- DARK MODE ADAPTIVE STYLING --- */
    @media (prefers-color-scheme: dark) {
        .brand-header {
            background: #1E293B !important;
            border-color: #334155 !important;
            color: #F8FAFC !important;
        }
        .brand-title {
            color: #F8FAFC !important;
        }
        .kpi-card {
            background-color: #1E293B !important;
            border-color: #334155 !important;
            color: #F8FAFC !important;
        }
        .kpi-value {
            color: #F8FAFC !important;
        }
        .kpi-label {
            color: #94A3B8 !important;
        }
        .section-banner {
            background: #1E293B !important;
            border-color: #334155 !important;
            color: #F8FAFC !important;
        }
        .section-banner h4 {
            color: #F8FAFC !important;
        }
        .section-banner p {
            color: #94A3B8 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #0B0F19 !important;
            border-right-color: #334155 !important;
        }
        .stButton>button {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border-color: #334155 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AUTHENTICATION & SECURE DATABASE
# ==========================================
def init_user_db():
    conn = sqlite3.connect("fermentiq_users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            facility TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("SELECT id FROM users WHERE username = ?", ("demo",))
    if not cursor.fetchone():
        salt = secrets.token_bytes(16).hex()
        pwd_hash = hashlib.pbkdf2_hmac('sha256', "demo123".encode('utf-8'), bytes.fromhex(salt), 100000).hex()
        cursor.execute("INSERT OR IGNORE INTO users (username, facility, password_hash, salt) VALUES (?, ?, ?, ?)",
                       ("demo", "FermentIQ Enterprise Lab", pwd_hash, salt))
    conn.commit()
    conn.close()

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_bytes(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), 100000).hex()
    return pwd_hash, salt

def register_user(username: str, facility: str, password: str) -> tuple[bool, str]:
    if not username.strip() or not password.strip():
        return False, "Username and password cannot be empty."
    conn = sqlite3.connect("fermentiq_users.db")
    cursor = conn.cursor()
    try:
        pwd_hash, salt = hash_password(password)
        cursor.execute("INSERT INTO users (username, facility, password_hash, salt) VALUES (?, ?, ?, ?)",
                       (username.strip().lower(), facility.strip(), pwd_hash, salt))
        conn.commit()
        conn.close()
        return True, "Account created successfully! Switch to Log In."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists."

def authenticate_user(username: str, password: str) -> tuple[bool, str, str]:
    conn = sqlite3.connect("fermentiq_users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT facility, password_hash, salt FROM users WHERE username = ?", (username.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        facility, stored_hash, salt = row
        pwd_hash, _ = hash_password(password, salt)
        if hmac.compare_digest(pwd_hash, stored_hash):
            return True, facility, "Login successful!"
        return False, "", "Incorrect password."
    return False, "", "Username not found."

init_user_db()

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'onboarded' not in st.session_state:
    st.session_state['onboarded'] = False

if not st.session_state['authenticated']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'><h2>🧬 FermentIQ Enterprise</h2><p style='color: #64748B;'>Advanced Bioprocess Digital Twin & Telemetry Suite</p></div>", unsafe_allow_html=True)
        
        auth_tab1, auth_tab2 = st.tabs(["🔒 Secure Log In", "📝 Create Account"])
        
        with auth_tab1:
            with st.form("login_form"):
                # Clean login inputs without hardcoded credentials pre-displayed
                login_user = st.text_input("Username or Email", value="")
                login_pass = st.text_input("Password", type="password", value="")
                st.caption("💡 Tip: Use username `demo` and password `demo123` or click **Continue as Guest** below.")
                st.markdown("<br>", unsafe_allow_html=True)
                c_login, c_guest = st.columns(2)
                with c_login:
                    login_btn = st.form_submit_button("Access Workspace ➔", use_container_width=True)
                with c_guest:
                    guest_btn = st.form_submit_button("Continue as Guest", use_container_width=True)
                
                if login_btn:
                    success, facility, msg = authenticate_user(login_user, login_pass)
                    if success:
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = login_user.strip().lower()
                        st.session_state['org'] = facility
                        st.rerun()
                    else:
                        st.error(msg)
                elif guest_btn:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = "guest_operator"
                    st.session_state['org'] = "Guest Facility"
                    st.rerun()
                        
        with auth_tab2:
            with st.form("signup_form"):
                new_user = st.text_input("Choose Username / Email")
                new_facility = st.text_input("Facility Name", "BioProcess Corp")
                new_pass = st.text_input("Create Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                st.markdown("<br>", unsafe_allow_html=True)
                signup_btn = st.form_submit_button("Register Enterprise Account", use_container_width=True)
                if signup_btn:
                    if new_pass != confirm_pass:
                        st.error("Passwords do not match!")
                    elif len(new_pass) < 4:
                        st.error("Password must be at least 4 characters.")
                    else:
                        created, msg = register_user(new_user, new_facility, new_pass)
                        if created: st.success(msg)
                        else: st.error(msg)
    st.stop()

if st.session_state['authenticated'] and not st.session_state['onboarded']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 🚀 Setup Your FermentIQ Session")
        with st.form("onboarding_form"):
            user_role = st.selectbox("Role", ["Student", "Academic Researcher", "Industrial Bioprocess Engineer", "R&D Scientist"])
            primary_goal = st.selectbox("Objective", ["Kinetics & Stoichiometry Training", "Fed-Batch Optimization", "Parameter Fitting", "Audit Compliance"])
            experience_level = st.select_slider("Expertise", options=["Beginner", "Intermediate", "Expert"])
            st.markdown("<br>", unsafe_allow_html=True)
            if st.form_submit_button("Launch Dashboard ➔", use_container_width=True):
                st.session_state['user_role'] = user_role
                st.session_state['primary_goal'] = primary_goal
                st.session_state['experience_level'] = experience_level
                st.session_state['onboarded'] = True
                st.rerun()
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

def fermentiq_v4_model(y, t, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2, K_O,
                       T_curr, T_opt, T_min, T_max, pH_curr, pH_opt, pH_width,
                       alpha_A, beta_A, A_crit,
                       feed_policy, F0, mu_set, DO_thresh, S_feed):
    
    X, S, P, A, DO, V = max(0, y[0]), max(0, y[1]), max(0, y[2]), max(0, y[3]), max(0, y[4]), max(1e-3, y[5])
    
    g_T = temp_factor(T_curr, T_opt, T_min, T_max)
    g_pH = ph_factor(pH_curr, pH_opt, pH_width)
    g_DO = DO / (K_O + DO) if K_O > 0 else 1.0
    byproduct_inh = max(0.0, 1.0 - (A / A_crit)) if A_crit > 0 else 1.0
    
    mu_base = mu_max * S / (Ks + S + ((S**2) / Ki)) if S > 0 else 0.0
    mu_eff = mu_base * g_T * g_pH * g_DO * byproduct_inh
    
    if feed_policy == "Exponential":
        F_in = F0 * math.exp(mu_set * t)
    elif feed_policy == "DO-Stat":
        DO_pct = (DO / C_star) * 100.0 if C_star > 0 else 100.0
        F_in = F0 * 3.0 if DO_pct > DO_thresh else F0 * 0.1
    else:
        F_in = F0
        
    D = F_in / V
    
    dXdt = (mu_eff - D) * X
    dSdt = D * (S_feed - S) - ((1.0 / Y_xs) * mu_eff * X) - ((1.0 / Y_ps) * mu_eff * X) if S > 0 else D * (S_feed - S)
    dPdt = -D * P + (Y_ps * mu_eff * X) + (alpha * mu_eff * X) + (beta * X)
    dAdt = -D * A + (alpha_A * mu_eff * X) + (beta_A * X)
    
    OUR = (q_O2 * (mu_eff / mu_max) + 0.05) * X * 1000.0 if mu_max > 0 else 0.05 * X * 1000.0
    OTR = kla * (C_star - DO)
    dDOdt = OTR - OUR - (D * DO)
    dVdt = F_in
    
    return [dXdt, dSdt, dPdt, dAdt, dDOdt, dVdt]

@st.cache_data(ttl=3600)
def run_fermentiq_simulation(X0, S0, P0, A0, DO0, V0, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2, K_O,
                             T_curr, T_opt, T_min, T_max, pH_curr, pH_opt, pH_width,
                             alpha_A, beta_A, A_crit,
                             feed_policy, F0, mu_set, DO_thresh, S_feed, batch_time, n_points=300):
    t = np.linspace(0, batch_time, n_points)
    try:
        sol = odeint(
            fermentiq_v4_model, [X0, S0, P0, A0, DO0, V0], t,
            args=(mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, C_star, q_O2, K_O,
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

def run_extended_kalman_filter(time_pts, noisy_biomass, mu_max, Ks, Y_xs, F0_val, V_init):
    n = len(time_pts)
    x_hat = np.zeros((n, 2))
    x_hat[0] = [noisy_biomass[0], 25.0]
    P_cov = np.eye(2) * 0.1
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.15
    
    for k in range(1, n):
        dt = time_pts[k] - time_pts[k-1]
        if dt <= 0: dt = 1e-4
            
        X_prev, S_prev = x_hat[k-1]
        V_curr = V_init + F0_val * time_pts[k]
        D_curr = F0_val / V_curr
        
        mu_est = mu_max * S_prev / (Ks + S_prev) if S_prev > 0 else 0.0
        X_pred = X_prev + ((mu_est - D_curr) * X_prev) * dt
        S_pred = max(0.0, S_prev + (D_curr * (200.0 - S_prev) - (1.0 / Y_xs) * mu_est * X_prev) * dt)
        
        den = (Ks + S_prev)**2
        dmu_dS = (mu_max * Ks / den) if den > 0 else 0.0
        
        F_mat = np.array([
            [1.0 + (mu_est - D_curr) * dt, X_prev * dmu_dS * dt],
            [-(1.0 / Y_xs) * mu_est * dt, 1.0 - (D_curr + (1.0 / Y_xs) * X_prev * dmu_dS) * dt]
        ])
        
        P_pred = F_mat @ P_cov @ F_mat.T + Q
        z = noisy_biomass[k]
        H = np.array([[1.0, 0.0]])
        y_residual = float(z - (H @ np.array([X_pred, S_pred]))[0])
        
        S_res = (H @ P_pred @ H.T) + R
        K_gain = (P_pred @ H.T) / S_res[0, 0]
        
        updated_state = np.array([X_pred, S_pred]) + (K_gain.flatten() * y_residual)
        x_hat[k] = [max(0.0, float(updated_state[0])), max(0.0, float(updated_state[1]))]
        P_cov = (np.eye(2) - K_gain @ H) @ P_pred
        
    rmse = np.sqrt(np.mean((x_hat[:, 0] - noisy_biomass)**2))
    return x_hat[:, 0], x_hat[:, 1], rmse

# ==========================================
# 4. ORGANISM PRESETS & SIDEBAR CONFIG
# ==========================================
ORGANISM_PRESETS = {
    "E. coli Recombinant Protein": {
        "mu_max": 0.65, "Ks": 0.20, "Ki": 150.0, "Y_xs": 0.50, "Y_ps": 0.18,
        "alpha": 0.12, "beta": 0.02, "X0": 0.15, "S0": 30.0, "batch_time": 18.0,
        "kla": 180.0, "F0": 0.05, "S_feed": 200.0, "T_opt": 37.0, "T_min": 15.0, "T_max": 45.0,
        "pH_opt": 7.0, "pH_width": 1.5, "alpha_A": 0.15, "beta_A": 0.02, "A_crit": 12.0, "q_O2": 0.25, "K_O": 0.1
    },
    "S. cerevisiae (Bioethanol)": {
        "mu_max": 0.42, "Ks": 0.45, "Ki": 80.0, "Y_xs": 0.14, "Y_ps": 0.46,
        "alpha": 0.05, "beta": 0.01, "X0": 0.50, "S0": 110.0, "batch_time": 32.0,
        "kla": 120.0, "F0": 0.0, "S_feed": 0.0, "T_opt": 30.0, "T_min": 10.0, "T_max": 40.0,
        "pH_opt": 5.0, "pH_width": 1.2, "alpha_A": 0.08, "beta_A": 0.01, "A_crit": 85.0, "q_O2": 0.18, "K_O": 0.15
    },
    "CHO Cell Line (mAb Expression)": {
        "mu_max": 0.038, "Ks": 0.12, "Ki": 300.0, "Y_xs": 0.62, "Y_ps": 0.28,
        "alpha": 0.18, "beta": 0.004, "X0": 0.20, "S0": 18.0, "batch_time": 120.0,
        "kla": 25.0, "F0": 0.005, "S_feed": 50.0, "T_opt": 36.5, "T_min": 25.0, "T_max": 41.0,
        "pH_opt": 7.2, "pH_width": 0.8, "alpha_A": 0.22, "beta_A": 0.005, "A_crit": 5.0, "q_O2": 0.08, "K_O": 0.05
    }
}  

with st.sidebar:
    st.markdown("### ⚙️ Control Center")
    st.caption(f"Facility: **{st.session_state.get('org', 'Default Facility')}**")
    
    use_custom_strain = st.checkbox("✍️ Enter Custom Strain", value=False)
    
    if use_custom_strain:
        if 'custom_strain_input' not in st.session_state:
            st.session_state['custom_strain_input'] = "E. coli BL21(DE3)"
            
        st.caption("Quick Suggestions:")
        sug_col1, sug_col2 = st.columns(2)
        with sug_col1:
            if st.button("E. coli BL21", use_container_width=True): st.session_state['custom_strain_input'] = "E. coli BL21(DE3)"
            if st.button("CHO-K1", use_container_width=True): st.session_state['custom_strain_input'] = "CHO-K1 Suspension"
        with sug_col2:
            if st.button("S. cerevisiae", use_container_width=True): st.session_state['custom_strain_input'] = "Saccharomyces cerevisiae BY4741"
            if st.button("P. pastoris", use_container_width=True): st.session_state['custom_strain_input'] = "Pichia pastoris Mut+"

        custom_strain_name = st.text_input("Strain Name", key='custom_strain_input')
        custom_strain_id = st.text_input("Batch ID", "LOT-2026-B01")
        preset_choice = f"{custom_strain_name} ({custom_strain_id})"
        preset = ORGANISM_PRESETS["E. coli Recombinant Protein"]
    else:
        preset_choice = st.selectbox("Active Strain Profile", list(ORGANISM_PRESETS.keys()))
        preset = ORGANISM_PRESETS[preset_choice]

    st.markdown("---")
    st.markdown("#### Kinetics")
    default_mu = st.session_state.get('fitted_mu', float(preset["mu_max"]))
    mu_max = st.slider("μ_max (1/h)", 0.01, 1.50, default_mu, 0.01)
    Ks = st.slider("Ks (g/L)", 0.01, 2.00, float(preset["Ks"]), 0.01)
    Ki = st.number_input("Ki (g/L)", 1.0, 1000.0, float(preset["Ki"]))
    Y_xs = st.slider("Y_x/s (g/g)", 0.05, 0.90, float(preset["Y_xs"]), 0.01)
    Y_ps = st.slider("Y_p/s (g/g)", 0.00, 0.90, float(preset["Y_ps"]), 0.01)

    st.markdown("---")
    st.markdown("#### Environment")
    T_curr = st.slider("Temperature (°C)", 15.0, 45.0, float(preset["T_opt"]), 0.5)
    pH_curr = st.slider("pH", 3.0, 10.0, float(preset["pH_opt"]), 0.1)
    T_opt, T_min, T_max = float(preset["T_opt"]), float(preset["T_min"]), float(preset["T_max"])
    pH_opt, pH_width = float(preset["pH_opt"]), float(preset["pH_width"])

    st.markdown("---")
    st.markdown("#### Byproducts & Feeding")
    alpha_A = st.number_input("α_A", 0.0, 2.0, float(preset["alpha_A"]), 0.01)
    beta_A = st.number_input("β_A", 0.0, 0.5, float(preset["beta_A"]), 0.001)
    A_crit = st.number_input("A_crit (g/L)", 1.0, 200.0, float(preset["A_crit"]), 1.0)
    
    feed_policy = st.selectbox("Feed Policy", ["Constant", "Exponential", "DO-Stat"])
    F0 = st.number_input("F0 (L/h)", 0.0, 5.0, float(preset["F0"]), 0.001)
    mu_set = st.number_input("μ_set (1/h)", 0.01, 0.80, 0.10, 0.01)
    DO_thresh = st.slider("DO Threshold (%)", 10.0, 95.0, 70.0, 5.0)
    S_feed = st.number_input("S_feed (g/L)", 0.0, 1000.0, float(preset["S_feed"]), 10.0)

    st.markdown("---")
    st.markdown("#### Initial States")
    X0 = st.number_input("X₀ (g/L)", 0.01, 50.0, float(preset["X0"]), 0.05)
    S0 = st.number_input("S₀ (g/L)", 0.1, 500.0, float(preset["S0"]), 1.0)
    A0 = st.number_input("A₀ (g/L)", 0.0, 50.0, 0.0, 0.1)
    kla = st.slider("kL a (1/h)", 5.0, 500.0, float(preset["kla"]))
    batch_time = st.slider("Batch Duration (hr)", 4.0, 200.0, float(preset["batch_time"]), 1.0)
    alpha, beta = preset["alpha"], preset["beta"]
    q_O2, K_O = float(preset["q_O2"]), float(preset["K_O"])

# ==========================================
# 5. SIMULATION EXECUTION
# ==========================================
sim_df, success = run_fermentiq_simulation(
    X0, S0, 0.0, A0, 7.0, 1.0, mu_max, Ks, Ki, Y_xs, Y_ps, alpha, beta, kla, 7.0, q_O2, K_O,
    T_curr, T_opt, T_min, T_max, pH_curr, pH_opt, pH_width,
    alpha_A, beta_A, A_crit, feed_policy, F0, mu_set, DO_thresh, S_feed, batch_time
)

if not success or sim_df.empty:
    st.error("⚠️ Model convergence error. Verify kinetic parameters.")
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
        <span>🧬 FermentIQ</span>
        <span class="brand-badge">Enterprise v4.1</span>
    </div>
    <div style="font-size: 0.85rem; color: #64748B;">
        Operator: <b style="color: #0F172A;">{st.session_state.get('username', 'Operator')}</b> | Strain: <b style="color: #0F172A;">{preset_choice}</b> | Mode: <b style="color: #0F172A;">{feed_policy}</b>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card" style="border-left-color: #0D9488;">
        <div class="kpi-label">Final Biomass (X)</div>
        <div class="kpi-value">{final_X:.2f} <span style="font-size:0.85rem; color:#64748B;">g/L</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #0284C7;">
        <div class="kpi-label">Target Product (P)</div>
        <div class="kpi-value" style="color: #0284C7;">{final_P:.2f} <span style="font-size:0.85rem; color:#64748B;">g/L</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #E11D48;">
        <div class="kpi-label">Byproduct (A)</div>
        <div class="kpi-value" style="color: {'#E11D48' if final_A > A_crit*0.7 else '#0F172A'};">{final_A:.2f} <span style="font-size:0.85rem; color:#64748B;">g/L</span></div>
    </div>
    <div class="kpi-card" style="border-left-color: #8B5CF6;">
        <div class="kpi-label">Vol. Productivity</div>
        <div class="kpi-value">{vol_prod:.3f} <span style="font-size:0.85rem; color:#64748B;">g/L·h</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. WORKFLOW TABS
# ==========================================
tab_twin, tab_ekf, tab_fitting, tab_report = st.tabs([
    "📈 Digital Twin",
    "📡 EKF Telemetry",
    "🔬 Parameter Fitting",
    "📄 Batch Compliance & Simulation Record"
])

# --- TAB 1: DIGITAL TWIN VISUALIZATION ---
with tab_twin:
    col_chart, col_stats = st.columns([3, 1])
    with col_chart:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Biomass X (g/L)'], name="Biomass (X)", line=dict(color="#0D9488", width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Substrate S (g/L)'], name="Substrate (S)", line=dict(color="#D97706", width=2, dash='dash')), secondary_y=False)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Byproduct A (g/L)'], name="Byproduct (A)", line=dict(color="#E11D48", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Product P (g/L)'], name="Product (P)", line=dict(color="#0284C7", width=3)), secondary_y=True)
        fig.add_trace(go.Scatter(x=sim_df['Time (hr)'], y=sim_df['Dissolved Oxygen DO (mg/L)'], name="DO", line=dict(color="#8B5CF6", width=2, dash='dot')), secondary_y=True)

        fig.update_layout(
            template="simple_white", height=500, hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=11)),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        grid_style = dict(showgrid=True, gridcolor="#F1F5F9")
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
        
        inhib_cap = max(0.0, (1.0 - (final_A / A_crit)) * 100.0) if A_crit > 0 else 100.0
        st.markdown("#### Reactor Metrics")
        st.markdown(f"""
        **Working Volume:** `{final_V:.2f} L`  
        **Inhibition Capacity:** `{inhib_cap:.1f}%`  
        **DO Status:** `{'CRITICAL' if min_DO < 1.0 else 'Optimal'}`  
        """)
        st.download_button("📥 Export CSV", sim_df.to_csv(index=False), "FermentIQ_Run.csv", "text/csv", use_container_width=True)

# --- TAB 2: TELEMETRY & EKF ---
with tab_ekf:
    st.markdown("""
    <div class="section-banner">
        <h4>📡 Live Sensor Telemetry & Extended Kalman Filter (EKF)</h4>
        <p>Noise-filtered optical density stream estimating unmeasured substrate concentrations in real time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    t_sensor = sim_df['Time (hr)'].values
    true_x = sim_df['Biomass X (g/L)'].values
    
    np.random.seed(42)
    noisy_sensor_x = np.maximum(0, true_x + np.random.normal(0, 0.35, len(true_x)))
    est_x, est_s, ekf_rmse = run_extended_kalman_filter(t_sensor, noisy_sensor_x, mu_max, Ks, Y_xs, F0, 1.0)
    
    st.metric("EKF Estimation RMSE", f"{ekf_rmse:.4f} g/L")
    
    fig_ekf = make_subplots(rows=1, cols=2, subplot_titles=("Biomass Estimation (Sensor Filtering)", "Reconstructed Substrate S(t)"))
    
    fig_ekf.add_trace(go.Scatter(x=t_sensor, y=noisy_sensor_x, mode='markers', name='Noisy OD Sensor', marker=dict(color='#E11D48', size=4, opacity=0.5)), row=1, col=1)
    fig_ekf.add_trace(go.Scatter(x=t_sensor, y=est_x, mode='lines', name='EKF Filtered State', line=dict(color='#0D9488', width=3)), row=1, col=1)
    
    fig_ekf.add_trace(go.Scatter(x=t_sensor, y=sim_df['Substrate S (g/L)'], mode='lines', name='True Substrate', line=dict(color='#D97706', dash='dash')), row=1, col=2)
    fig_ekf.add_trace(go.Scatter(x=t_sensor, y=est_s, mode='lines', name='EKF Reconstructed S(t)', line=dict(color='#0284C7', width=3)), row=1, col=2)
    
    fig_ekf.update_layout(template="simple_white", height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_ekf, use_container_width=True)

# --- TAB 3: PARAMETER FITTING ---
with tab_fitting:
    st.markdown("#### 🔬 Non-Linear Regression & Log-Linear Exponential Phase Detection")
    input_mode = st.radio("Input Source:", ["📄 Upload CSV File", "✍️ Paste Raw Data"], horizontal=True)
    exp_df = None
    
    if input_mode == "📄 Upload CSV File":
        uploaded_file = st.file_uploader("Select CSV", type=["csv", "txt"])
        if uploaded_file is not None:
            try: exp_df = pd.read_csv(uploaded_file)
            except Exception as e: st.error(f"Error: {e}")
    else:
        raw_text = st.text_area("Paste Data:", height=100, placeholder="Time,Biomass\n0,0.15\n2,0.32")
        if raw_text.strip():
            try:
                sep = '\t' if '\t' in raw_text else ','
                exp_df = pd.read_csv(io.StringIO(raw_text.strip()), sep=sep)
            except Exception as e: st.error(f"Error: {e}")

    if exp_df is None:
        exp_df = pd.DataFrame({
            'Time': [0, 2, 4, 6, 8, 12, 16, 20, 24],
            'Biomass': [0.15, 0.32, 0.78, 1.85, 3.90, 7.20, 8.10, 8.35, 8.40],
            'Substrate': [30.0, 28.5, 25.1, 19.8, 12.0, 3.2, 0.5, 0.1, 0.0]
        })

    st.dataframe(exp_df, height=120, use_container_width=True)

    if 'Time' in exp_df.columns and 'Biomass' in exp_df.columns:
        try:
            t_data = exp_df['Time'].astype(float).values
            x_data = exp_df['Biomass'].astype(float).values
            
            best_mu = 0.1
            best_ci = 0.05
            valid_idx = x_data > 0
            if sum(valid_idx) >= 3:
                t_v, x_v = t_data[valid_idx], np.log(x_data[valid_idx])
                slopes = []
                for i in range(len(t_v) - 2):
                    slope, _ = np.polyfit(t_v[i:i+3], x_v[i:i+3], 1)
                    slopes.append(slope)
                if slopes:
                    best_mu = max(0.01, float(np.max(slopes)))
                    best_ci = float(1.96 * (np.std(slopes) / np.sqrt(len(slopes)))) if len(slopes) > 1 else 0.05

            c1, c2 = st.columns(2)
            with c1: st.metric("Regressed Max Growth Rate (μ_max)", f"{best_mu:.4f} h⁻¹", delta=f"± {best_ci:.4f} (95% CI)")
            with c2:
                if st.button("⚡ Sync μ_max to Model", use_container_width=True):
                    st.session_state['fitted_mu'] = float(best_mu)
                    st.success("Synchronized!")
                    st.rerun()

            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(x=t_data, y=x_data, mode='markers', name='Lab Points', marker=dict(size=8, color='#E11D48')))
            t_smooth = np.linspace(0, max(t_data), 100)
            fit_curve = x_data[0] * np.exp(best_mu * t_smooth)
            fig_fit.add_trace(go.Scatter(x=t_smooth, y=fit_curve, mode='lines', name='Exponential Fit', line=dict(color='#0D9488', dash='dash')))
            fig_fit.update_layout(template="simple_white", height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_fit, use_container_width=True)
        except Exception as e: st.error(f"Regression error: {e}")

# --- TAB 4: AUDIT COMPLIANCE & SIMULATION RECORD ---
with tab_report:
    st.markdown("#### 📄 Batch Compliance & Simulation Record")
    
    current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    user_name = st.session_state.get('username', 'Operator')
    facility_name = st.session_state.get('org', 'Default Facility')
    final_time = sim_df['Time (hr)'].iloc[-1]
    
    report_html = f"""
    <div style="background:#FFFFFF; color:#0F172A; padding:32px; border:1px solid #E2E8F0; border-radius:14px; font-family:'Plus Jakarta Sans', sans-serif; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #0284C7; padding-bottom:16px; margin-bottom:24px;">
            <div>
                <h2 style="color:#0284C7; margin:0; font-size:24px; font-weight:700;">FermentIQ Simulation Record</h2>
                <span style="font-size:13px; color:#64748B;">Digital Twin Bioprocess Batch Run</span>
            </div>
            <div>
                <span style="background:#64748B; color:#FFF; font-size:11px; font-weight:700; padding:6px 12px; border-radius:4px; text-transform:uppercase;">SIMULATED BATCH</span>
            </div>
        </div>
        
        <table style="width:100%; font-size:13px; margin-bottom:24px; border-collapse:collapse;">
            <tr>
                <td style="padding:6px 0;"><b>Facility:</b> {facility_name}</td>
                <td style="padding:6px 0;"><b>Operator:</b> {user_name}</td>
            </tr>
            <tr>
                <td style="padding:6px 0;"><b>Strain Profile:</b> {preset_choice}</td>
                <td style="padding:6px 0;"><b>Execution Timestamp:</b> {current_time_str}</td>
            </tr>
        </table>

        <h3 style="font-size:14px; color:#0D9488; border-bottom:1px solid #E2E8F0; padding-bottom:6px; margin-top:20px;">1. Operating Parameters</h3>
        <table style="width:100%; font-size:12px; border-collapse:collapse; margin-bottom:20px;">
            <tr style="background:#F8FAFC;">
                <td style="padding:8px;"><b>Temperature:</b> {T_curr}°C (Opt: {T_opt}°C)</td>
                <td style="padding:8px;"><b>pH:</b> {pH_curr} (Opt: {pH_opt})</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Feeding Strategy:</b> {feed_policy}</td>
                <td style="padding:8px;"><b>Base Feed Rate (F₀):</b> {F0} L/h</td>
            </tr>
        </table>

        <h3 style="font-size:14px; color:#0D9488; border-bottom:1px solid #E2E8F0; padding-bottom:6px; margin-top:20px;">2. Yield & Productivity Metrics</h3>
        <table style="width:100%; font-size:12px; border-collapse:collapse; margin-bottom:24px;">
            <tr style="background:#F8FAFC;">
                <td style="padding:8px;"><b>Batch Duration:</b> {final_time:.1f} Hours</td>
                <td style="padding:8px;"><b>Final Volume:</b> {final_V:.2f} L</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Product Titer (P):</b> <b style="color:#0284C7;">{final_P:.2f} g/L</b></td>
                <td style="padding:8px;"><b>Volumetric Productivity:</b> <b>{vol_prod:.3f} g/L·h</b></td>
            </tr>
        </table>

        <div style="font-size:11px; color:#64748B; background:#F8FAFC; padding:12px; border-radius:8px; margin-bottom:16px; border-left:3px solid #D97706;">
            <b>Regulatory Disclaimer:</b> Non-GMP Digital Twin Simulation Record - For Research & Process Development Use Only. Values derived from mathematical kinetic modeling.
        </div>

        <div style="font-size:11px; color:#94A3B8; text-align:center; border-top:1px solid #E2E8F0; padding-top:12px;">
            FermentIQ v4.1 Engine • Bioprocess Simulation & Analytics Suite
        </div>
    </div>
    """
    st.components.v1.html(report_html, height=580, scrolling=True)
    
    st.download_button(
        label="📥 Download Compliance Report (HTML)",
        data=report_html,
        file_name=f"FermentIQ_Compliance_Report_{datetime.datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
        use_container_width=True
    )
