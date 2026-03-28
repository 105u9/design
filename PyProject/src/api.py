# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import torch
import time
import logging
from datetime import datetime, timedelta

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import LSTM_ED_Model
from optimization import MOPSO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HVAC Intelligent Control System API",
    description="Backend API for real-time HVAC monitoring and optimization",
    version="2.0.0"
)

# Authentication logic (Simulated for Phase 5)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if token != "fake-super-secret-token":
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return {"username": "admin"}

# Data Models
class SensorData(BaseModel):
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temperature: float
    humidity: float
    co2: float
    person_density: float
    power: float

class ControlAction(BaseModel):
    chilled_water_temp: float
    supply_air_setpoint: float
    mode: str = "AI_AUTO"

class PredictionResponse(BaseModel):
    forecast_horizon: List[str]
    predicted_load: List[float]

# In-memory time-series store (Simulating InfluxDB)
history_data: List[SensorData] = []
MAX_HISTORY = 100

# Global System State
system_state = {
    "ai_mode": True,
    "last_update": datetime.now().isoformat()
}

from preprocessing import load_building_data, clean_and_impute, normalize_data

# Load real data for demo
try:
    df_raw = load_building_data("Panther_office_Karla", "Panther")
    df_real = clean_and_impute(df_raw, method='mean')
    real_samples = df_real.to_dict('records')
except:
    real_samples = []

def initialize_demo_data():
    now = datetime.now()
    if real_samples:
        # Use last 24 records from BDG2 for initial history
        for i in range(min(24, len(real_samples))):
            sample = real_samples[-(24-i)]
            history_data.append(SensorData(
                timestamp=(now - timedelta(hours=24-i)).strftime("%Y-%m-%d %H:%M:%S"),
                temperature=float(sample['airTemperature']),
                humidity=float(sample.get('humidity', 50.0)),
                co2=400.0 + np.random.uniform(0, 100),
                person_density=np.random.uniform(0, 10),
                power=float(sample['power_usage'])
            ))
    else:
        # Fallback to random if loading fails
        for i in range(24):
            t = (now - timedelta(hours=24-i)).strftime("%Y-%m-%d %H:%M:%S")
            history_data.append(SensorData(
                timestamp=t,
                temperature=22.0 + np.random.uniform(-2, 2),
                humidity=50.0 + np.random.uniform(-5, 5),
                co2=400 + np.random.uniform(0, 200),
                person_density=np.random.uniform(0, 10),
                power=100 + np.random.uniform(0, 50)
            ))

initialize_demo_data()

# Endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == "admin" and form_data.password == "admin123":
        return {"access_token": "fake-super-secret-token", "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "version": "2.0.0",
        "endpoints": {
            "monitoring": "/api/v1/monitoring",
            "predict": "/api/v1/predict",
            "optimize": "/api/v1/optimize"
        }
    }

@app.get("/api/v1/monitoring")
def get_monitoring(current_user: dict = Depends(get_current_user)):
    """Get the latest 24 hours of monitoring data and system state"""
    return {
        "history": history_data[-24:],
        "system": system_state
    }

@app.post("/api/v1/toggle_ai")
def toggle_ai(mode: bool, current_user: dict = Depends(get_current_user)):
    """Toggle AI control mode on/off"""
    system_state["ai_mode"] = mode
    system_state["last_update"] = datetime.now().isoformat()
    logger.info(f"AI Mode toggled to: {mode}")
    return {"status": "success", "ai_mode": mode}

@app.post("/api/v1/collect")
def collect_data(data: SensorData, current_user: dict = Depends(get_current_user)):
    """Receive real-time data from sensors (Perception -> Platform)"""
    history_data.append(data)
    if len(history_data) > MAX_HISTORY:
        history_data.pop(0)
    logger.info(f"Received data: {data}")
    return {"status": "success", "received_at": datetime.now().isoformat()}

# Global Model Reference
model_instance = None
try:
    # Load trained model
    input_size = 8 # BDG2 numeric features
    hidden_size = 64
    output_size = 1
    forecast_len = 12
    model_instance = LSTM_ED_Model(input_size, hidden_size, output_size, forecast_len)
    if os.path.exists("src/lstm_model.pth"):
        model_instance.load_state_dict(torch.load("src/lstm_model.pth"))
    model_instance.eval()
except Exception as e:
    logger.warning(f"Could not load LSTM model: {e}")

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_load(current_user: dict = Depends(get_current_user)):
    """Predict load using LSTM model (Model layer)"""
    now = datetime.now()
    forecast_horizon = [(now + timedelta(hours=i)).strftime("%H:00") for i in range(1, 13)]
    
    if model_instance:
        try:
            # Use real recent data from BDG2 if available
            if len(real_samples) >= 24:
                # Get last 24 records, convert to numeric and normalize (simplified)
                recent_data = pd.DataFrame(real_samples[-24:]).select_dtypes(include=[np.number]).values
                input_tensor = torch.FloatTensor(recent_data).unsqueeze(0) # [1, 24, num_features]
            else:
                input_tensor = torch.zeros(1, 24, 8) 
            
            with torch.no_grad():
                pred = model_instance(input_tensor)
            
            # Use the first feature (power_usage) from prediction
            predicted_load_raw = pred[0, :, 0].tolist()
            
            # Rescale back to kW (approximate for demo)
            predicted_load = [float(p * 100 + 150) for p in predicted_load_raw]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            predicted_load = [float(150 + 50 * np.sin(i/3) + np.random.uniform(-5, 5)) for i in range(12)]
    else:
        predicted_load = [float(150 + 50 * np.sin(i/3) + np.random.uniform(-5, 5)) for i in range(12)]
        
    return PredictionResponse(forecast_horizon=forecast_horizon, predicted_load=predicted_load)

@app.post("/api/v1/optimize", response_model=ControlAction)
def optimize_control(current_user: dict = Depends(get_current_user)):
    """Solve for optimal control parameters using MOPSO (Optimization layer)"""
    def fitness_func(x):
        # x[0] = setpoint temp
        energy = x[0]**2 - 40 * x[0] + 500 # Simplified quadratic energy model
        comfort = abs(x[0] - 22.5) # Distance to ideal comfort temp
        return [energy, comfort]

    mopso = MOPSO(fitness_func, bounds=[[18, 26]], num_particles=20, max_iter=10)
    pareto_front = mopso.solve()
    
    # Select the best compromise solution (e.g., closest to ideal temp)
    best_sol = min(pareto_front, key=lambda x: x['fitness'][1])
    
    return ControlAction(
        chilled_water_temp=7.0 + np.random.uniform(-0.5, 0.5),
        supply_air_setpoint=float(best_sol['position'][0])
    )

# Static files for Frontend (Phase 5)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
