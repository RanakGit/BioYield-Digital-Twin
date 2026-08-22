"""
BioTwin Pro Enterprise Engine v4.0 - Advanced Microservice
FastAPI + SciPy + EKF State Estimator + Environmental Kinetics
"""

import math
from typing import List, Optional
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import numpy as np
from scipy.integrate import odeint

app = FastAPI(
    title="BioTwin Pro Enterprise Core API",
    version="4.0.0",
    description="High-Performance Bioprocess Digital Twin with EKF & Environmental Kinetics"
)

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if not token or token not in ["biotwin_enterprise_secret_key", "demo", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Bearer Token"
        )
    return token

# --- ADVANCED PARAMETER MODELS ---
class EnvironmentalConfig(BaseModel):
    T_curr: float = Field(37.0, description="Current Operating Temp (°C)")
    T_opt: float = Field(37.0, description="Optimal Temp (°C)")
    T_max: float = Field(45.0, description="Maximum Growth Temp (°C)")
    T_min: float = Field(15.0, description="Minimum Growth Temp (°C)")
    pH_curr: float = Field(7.0, description="Current Operating pH")
    pH_opt: float = Field(7.0, description="Optimal pH")
    pH_width: float = Field(1.5, description="pH Tolerance Width")

class ByproductKinetics(BaseModel):
    alpha_byproduct: float = Field(0.1, ge=0, description="Growth-associated rate constant")
    beta_byproduct: float = Field(0.01, ge=0, description="Non-growth associated rate constant")
    A_crit: float = Field(15.0, gt=0, description="Critical byproduct threshold (g/L)")
    inhibition_power: float = Field(1.0, ge=0.5)

class FeedingPolicyConfig(BaseModel):
    policy: str = Field("Constant", description="'Constant', 'Exponential', or 'DO-Stat'")
    F0: float = Field(0.0, ge=0, description="Base/Initial feed rate (L/h)")
    mu_set: float = Field(0.1, ge=0, description="Target growth rate for exponential feed")
    DO_threshold: float = Field(70.0, ge=0, opacity=100, description="DO % threshold for DO-Stat trigger")
    S_feed: float = Field(200.0, ge=0, description="Feed substrate concentration (g/L)")

class KineticParametersV4(BaseModel):
    mu_max: float = Field(..., gt=0)
    Ks: float = Field(..., gt=0)
    Ki: float = Field(99999.0, gt=0)
    Y_xs: float = Field(..., gt=0)
    Y_ps: float = Field(..., ge=0)
    alpha: float = Field(0.05, ge=0)
    beta: float = Field(0.01, ge=0)
    kla: float = Field(100.0, gt=0)
    C_star: float = Field(7.0, gt=0)
    q_O2: float = Field(0.15, gt=0)
    env: EnvironmentalConfig
    byproduct: ByproductKinetics

class ReactorStateV4(BaseModel):
    X0: float = Field(..., gt=0)
    S0: float = Field(..., gt=0)
    P0: float = Field(0.0, ge=0)
    A0: float = Field(0.0, ge=0, description="Initial Byproduct Conc (g/L)")
    DO0: float = Field(7.0, ge=0)
    V0: float = Field(1.0, gt=0)
    batch_time: float = Field(..., gt=0)
    feeding: FeedingPolicyConfig

# --- HELPER KINETIC FUNCTIONS ---
def calc_temp_factor(T, T_opt, T_min, T_max):
    if T <= T_min or T >= T_max: return 0.0
    num = (T - T_max) * ((T - T_min)**2)
    den = (T_opt - T_min) * ((T_opt - T_min)*(T - T_opt) - (T_opt - T_max)*(T_opt + T_min - 2.0*T))
    return max(0.0, num / den) if den != 0 else 0.0

def calc_ph_factor(pH, pH_opt, width):
    return math.exp(-((pH - pH_opt) / width)**2)

# --- ODE SYSTEM ---
def enterprise_bioprocess_ode(y, t, k: KineticParametersV4, r: ReactorStateV4):
    X, S, P, A, DO, V = max(0, y[0]), max(0, y[1]), max(0, y[2]), max(0, y[3]), max(0, y[4]), max(1e-3, y[5])
    
    # 1. Environmental Stress Corrections
    gamma_T = calc_temp_factor(k.env.T_curr, k.env.T_opt, k.env.T_min, k.env.T_max)
    gamma_pH = calc_ph_factor(k.env.pH_curr, k.env.pH_opt, k.env.pH_width)
    
    # 2. Toxic Byproduct Inhibition
    byproduct_inhibition = max(0.0, 1.0 - (A / k.byproduct.A_crit))**k.byproduct.inhibition_power
    
    # 3. Effective Growth Rate (Haldane + Environmental + Byproduct)
    mu_base = k.mu_max * S / (k.Ks + S + ((S**2) / k.Ki)) if S > 0 else 0.0
    mu_eff = mu_base * gamma_T * gamma_pH * byproduct_inhibition
    
    # 4. Dynamic Feeding Policy Strategy
    feed_policy = r.feeding.policy
    if feed_policy == "Exponential":
        F_in = r.feeding.F0 * math.exp(r.feeding.mu_set * t)
    elif feed_policy == "DO-Stat":
        # DO-Stat: If DO rises above threshold (oxygen accumulation = substrate depletion), feed ramps up
        DO_pct = (DO / k.C_star) * 100.0
        F_in = r.feeding.F0 * 3.5 if DO_pct > r.feeding.DO_threshold else r.feeding.F0 * 0.2
    else:  # Constant
        F_in = r.feeding.F0
        
    D = F_in / V
    
    # 5. Mass Balances
    dXdt = (mu_eff - D) * X
    dSdt = D * (r.feeding.S_feed - S) - ((1.0 / k.Y_xs) * mu_eff * X)
    dPdt = -D * P + (k.Y_ps * mu_eff * X) + (k.alpha * mu_eff * X) + (k.beta * X)
    
    # Luedeking-Piret Byproduct Formation (e.g. Acetate / Lactate)
    dAdt = -D * A + (k.byproduct.alpha_byproduct * mu_eff * X) + (k.byproduct.beta_byproduct * X)
    
    # Oxygen Mass Transfer Dynamics (OTR - OUR)
    OTR = k.kla * (k.C_star - DO)
    OUR = k.q_O2 * X * 1000.0  # mg/L/h
    dDOdt = OTR - OUR - (D * DO)
    
    dVdt = F_in
    
    return [dXdt, dSdt, dPdt, dAdt, dDOdt, dVdt]
