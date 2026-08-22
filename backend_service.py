"""
BioTwin Pro Enterprise Engine v3.0 - Microservice
FastAPI + SciPy + Celery-Ready Kinetic Solver
"""

import math
from typing import List, Optional
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit

app = FastAPI(
    title="BioTwin Pro Numerical Core API",
    version="3.0.0",
    description="High-performance Bioprocess Digital Twin Computation Engine"
)

security = HTTPBearer()

# --- SECURITY & AUTHENTICATION ---
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if not token or token != "biotwin_enterprise_secret_key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# --- DATA MODELS ---
class KineticParameters(BaseModel):
    mu_max: float = Field(..., gt=0, description="Max specific growth rate (1/h)")
    Ks: float = Field(..., gt=0, description="Substrate affinity constant (g/L)")
    Ki: float = Field(default=99999.0, gt=0, description="Haldane inhibition constant (g/L)")
    Y_xs: float = Field(..., gt=0, description="Biomass yield coefficient (g/g)")
    Y_ps: float = Field(..., ge=0, description="Product yield coefficient (g/g)")
    alpha: float = Field(0.05, ge=0)
    beta: float = Field(0.01, ge=0)
    kla: float = Field(100.0, gt=0, description="Volumetric mass transfer coefficient (1/h)")
    C_star: float = Field(7.0, gt=0, description="Saturation dissolved oxygen (mg/L)")
    q_O2: float = Field(0.15, gt=0, description="Specific oxygen consumption rate (g O2/g X/h)")

class ReactorState(BaseModel):
    X0: float = Field(..., gt=0)
    S0: float = Field(..., gt=0)
    P0: float = Field(0.0, ge=0)
    DO0: float = Field(7.0, ge=0)
    V0: float = Field(1.0, gt=0, description="Initial liquid volume (L)")
    F_in: float = Field(0.0, ge=0, description="Feed flow rate (L/h)")
    S_feed: float = Field(0.0, ge=0, description="Substrate concentration in feed (g/L)")
    batch_time: float = Field(..., gt=0)

class SimulationRequest(BaseModel):
    kinetics: KineticParameters
    reactor: ReactorState
    n_points: int = 300

# --- NUMERICAL SOLVER ---
def extended_bioprocess_ode(y, t, k: KineticParameters, r: ReactorState):
    X, S, P, DO, V = max(0, y[0]), max(0, y[1]), max(0, y[2]), max(0, y[3]), max(1e-3, y[4])
    
    # Haldane Substrate Inhibition Kinetics
    mu = k.mu_max * S / (k.Ks + S + (S**2 / k.Ki)) if S > 0 else 0.0
    
    # Dilution rate
    D = r.F_in / V
    
    # Mass Balance ODEs
    dXdt = (mu - D) * X
    dSdt = D * (r.S_feed - S) - (1.0 / k.Y_xs) * mu * X
    dPdt = -D * P + (k.Y_ps * mu * X) + (k.alpha * mu * X) + (k.beta * X)
    
    # Dissolved Oxygen Dynamics (OTR - OUR)
    OTR = k.kla * (k.C_star - DO)
    OUR = k.q_O2 * X * 1000.0  # mg/L scaling
    dDOdt = OTR - OUR - (D * DO)
    
    dVdt = r.F_in
    
    return [dXdt, dSdt, dPdt, dDOdt, dVdt]

@app.post("/api/v3/simulate", dependencies=[Depends(verify_token)])
def simulate_batch(req: SimulationRequest):
    t = np.linspace(0, req.reactor.batch_time, req.n_points)
    y0 = [req.reactor.X0, req.reactor.S0, req.reactor.P0, req.reactor.DO0, req.reactor.V0]
    
    try:
        sol = odeint(extended_bioprocess_ode, y0, t, args=(req.kinetics, req.reactor))
        return {
            "status": "success",
            "time": t.tolist(),
            "biomass": np.clip(sol[:, 0], 0, None).tolist(),
            "substrate": np.clip(sol[:, 1], 0, None).tolist(),
            "product": np.clip(sol[:, 2], 0, None).tolist(),
            "dissolved_oxygen": np.clip(sol[:, 3], 0, None).tolist(),
            "volume": sol[:, 4].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver Convergence Error: {str(e)}")
