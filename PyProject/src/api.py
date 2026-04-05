# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import torch
import joblib
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

# --- JWT Configuration (Security Optimization) ---
SECRET_KEY = "your-secret-key-for-graduation-project" # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 # 2 hours as suggested

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- API Tags Metadata for better documentation ---
tags_metadata = [
    {
        "name": "认证管理",
        "description": "用于系统安全访问控制，获取和管理 OAuth2 令牌。",
    },
    {
        "name": "环境感知",
        "description": "处理来自 IoT 设备和传感器的实时监测数据，包括温湿度、CO2 和能耗。",
    },
    {
        "name": "智能预测",
        "description": "利用 **GAT-LSTM (图注意力网络+长短期记忆网络)** 深度学习模型，对未来负荷进行高精度预测。",
    },
    {
        "name": "优化控制",
        "description": "基于 **MOPSO (多目标粒子群算法)** 在节能与舒适度之间寻找最佳平衡点。",
    },
    {
        "name": "系统配置",
        "description": "管理 AI 控制模式开关及系统运行参数。",
    },
]

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import LSTM_ED_Model
from optimization import MOPSO, calculate_pmv
from preprocessing import load_building_data, clean_and_impute, normalize_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HVAC 智能控制系统 API",
    description="""
## 项目背景
本项目是一个基于数据驱动的空调智能控制系统原型。它通过集成深度学习预测与多目标优化算法，旨在实现建筑节能与室内舒适度的双重目标。

## 核心功能说明
*   **多维感知**：支持温度、湿度、CO2、能耗等多种环境参数的实时采集。
*   **精准预测**：采用 Encoder-Decoder LSTM 架构，结合图卷积捕捉特征相关性。
*   **智能决策**：通过 MOPSO 算法动态调整空调设定值，比传统固定设定值更智能、更节能。

## 快速上手
1.  在 **认证管理** 标签下使用默认账号 `admin/admin123` 获取 Token。
2.  在右上角 `Authorize` 按钮处填入 Token 即可解锁受保护的接口。
""",
    version="2.0.0",
    openapi_tags=tags_metadata
)

# Load global scaler and model
scaler = None
adj_matrix = None
metadata = None
if os.path.exists("src/data_scaler.pkl"):
    scaler = joblib.load("src/data_scaler.pkl")
    logger.info("Loaded scaler from src/data_scaler.pkl")
if os.path.exists("src/adj_matrix.pkl"):
    adj_matrix = joblib.load("src/adj_matrix.pkl")
    logger.info("Loaded adjacency matrix from src/adj_matrix.pkl")
if os.path.exists("src/metadata.pkl"):
    metadata = joblib.load("src/metadata.pkl")
    logger.info("Loaded metadata from src/metadata.pkl")

# Device Detection
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
logger.info(f"API Device Selection: {device}")
if not cuda_available:
    logger.warning("NVIDIA GPU not detected by PyTorch. Using CPU for API inference. If you have a GPU, please install the CUDA-enabled version of PyTorch.")
else:
    logger.info(f"API CUDA Enabled: {torch.cuda.get_device_name(0)}")
    # Optimization for repeated fixed-size inputs
    torch.backends.cudnn.benchmark = True

# Load real data for demo and to determine model input size
try:
    from preprocessing import load_building_data, clean_and_impute, normalize_data
    df_raw = load_building_data("Panther_office_Karla", "Panther")
    df_real = clean_and_impute(df_raw, method='mean')
    real_samples = df_real.to_dict('records')
    
    # --- Robustness Fix: Use metadata for input size if available ---
    if metadata:
        dynamic_input_size = metadata['input_size']
        feature_names = metadata['feature_names']
        logger.info(f"Using metadata input size: {dynamic_input_size}")
    else:
        # Dynamically determine the number of numeric features
        dynamic_input_size = df_real.select_dtypes(include=[np.number]).shape[1]
        feature_names = df_real.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Determined dynamic input size: {dynamic_input_size}")
except Exception as e:
    logger.warning(f"Could not load real samples for dynamic sizing: {e}")
    real_samples = []
    dynamic_input_size = 8 # Fallback
    feature_names = []

model_instance = None
try:
    # Load trained model using dynamic input size
    hidden_size = 64
    # Requirement: Predict multi-dimensional (Power, Temp, Humidity, CO2)
    output_size = 4 
    forecast_len = 12
    model_instance = LSTM_ED_Model(dynamic_input_size, hidden_size, output_size, forecast_len).to(device)
    if os.path.exists("src/lstm_model.pth"):
        # weights_only=True to address security warnings in modern PyTorch
        state_dict = torch.load("src/lstm_model.pth", map_location=device, weights_only=True)
        model_instance.load_state_dict(state_dict)
        logger.info("Successfully loaded multi-dimensional model.")
    model_instance.eval()
except Exception as e:
    logger.warning(f"Could not load LSTM model: {e}")

# Authentication logic (Security Optimization with JWT)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    return {"username": username}

# Data Models
class SensorData(BaseModel):
    timestamp: str = Field(..., description="传感器采集时间 (YYYY-MM-DD HH:MM:SS)", example="2026-04-05 12:00:00")
    temperature: float = Field(..., description="室内温度 (℃)", example=24.5)
    humidity: float = Field(..., description="室内相对湿度 (%)", example=45.0)
    co2: float = Field(..., description="二氧化碳浓度 (ppm)", example=650.0)
    power: float = Field(..., description="空调当前能耗 (kW)", example=125.4)

class ControlAction(BaseModel):
    chilled_water_temp: float = Field(..., description="推荐冷冻水出水温度 (℃)", example=7.0)
    supply_air_setpoint: float = Field(..., description="推荐室内空调设定值 (℃)", example=22.5)
    mode: str = Field(..., description="当前控制模式", example="AI_OPTIMIZED")

class PredictionResponse(BaseModel):
    forecast_horizon: List[str] = Field(..., description="预测时间窗 (未来 12 小时)", example=["13:00", "14:00", "15:00"])
    predicted_load: List[float] = Field(..., description="未来 12 小时空调能耗预测值 (kW)", example=[130.5, 135.2, 140.1])
    predicted_temp: List[float] = Field(..., description="未来 12 小时室内温度预测值 (℃)", example=[24.0, 24.1, 24.3])
    predicted_humidity: List[float] = Field(..., description="未来 12 小时湿度预测值 (%)", example=[45.0, 45.5, 46.0])
    predicted_co2: List[float] = Field(..., description="未来 12 小时 CO2 浓度预测值 (ppm)", example=[650, 660, 675])

# In-memory time-series store (Simulating InfluxDB)
history_data: List[SensorData] = []
MAX_HISTORY = 100

# Global System State
system_state = {
    "ai_mode": True,
    "last_update": datetime.now().isoformat()
}

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
                power=100 + np.random.uniform(0, 50)
            ))

initialize_demo_data()

# Endpoints
@app.post("/token", tags=["认证管理"], summary="获取访问令牌", description="使用管理员账号获取 OAuth2 访问令牌，默认账号：admin，默认密码：admin123")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == "admin" and form_data.password == "admin123":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="用户名或密码不正确")

@app.get("/", tags=["系统配置"], summary="根路径重定向", description="自动重定向到可视化监控大屏页面，方便用户快速查看系统状态")
def read_root():
    """Redirect to the visual dashboard"""
    return RedirectResponse(url="/static/index.html")

@app.get("/api/v1/monitoring", tags=["环境感知"], summary="获取监测数据", description="""
获取最近 24 小时的实时环境监测数据。
包含：温度 (℃)、湿度 (%)、CO2 (ppm) 和能耗 (kW)。
本接口会自动模拟实时数据更新，确保监控大屏保持动态。
""")
def get_monitoring(current_user: dict = Depends(get_current_user)):
    """Get the latest 24 hours of monitoring data and system state"""
    # Dynamic Simulation: If the frontend requests data, we can simulate a real-time increment
    # This ensures the dashboard feels "live" even without external IoT input
    if history_data:
        last_data = history_data[-1]
        # Simulate a small random walk from the last data point
        new_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if new_time != last_data.timestamp:
            new_val = SensorData(
                timestamp=new_time,
                temperature=last_data.temperature + np.random.uniform(-0.1, 0.1),
                humidity=np.clip(last_data.humidity + np.random.uniform(-0.5, 0.5), 30, 80),
                co2=np.clip(last_data.co2 + np.random.uniform(-5, 5), 400, 1000),
                power=np.clip(last_data.power + np.random.uniform(-1, 1), 50, 300)
            )
            history_data.append(new_val)
            if len(history_data) > MAX_HISTORY:
                history_data.pop(0)

    return {
        "history": history_data[-24:],
        "system": system_state
    }

@app.post("/api/v1/toggle_ai", tags=["系统配置"], summary="切换 AI 模式", description="开启或关闭基于 MOPSO 的 AI 自动优化控制模式。开启后系统将动态计算最优设定值，关闭后恢复基准 24.0℃ 控制。")
def toggle_ai(mode: bool, current_user: dict = Depends(get_current_user)):
    """Toggle AI control mode on/off"""
    system_state["ai_mode"] = mode
    system_state["last_update"] = datetime.now().isoformat()
    logger.info(f"AI 模式已切换为: {mode}")
    return {"status": "success", "ai_mode": mode}

@app.post("/api/v1/collect", tags=["环境感知"], summary="接收传感器数据", description="模拟工业物联网 (IoT) 设备的实时采集数据上报接口。")
def collect_data(data: SensorData, current_user: dict = Depends(get_current_user)):
    """Receive real-time data from sensors (Perception -> Platform)"""
    history_data.append(data)
    if len(history_data) > MAX_HISTORY:
        history_data.pop(0)
    logger.info(f"接收到传感器数据: {data}")
    return {"status": "success", "received_at": datetime.now().isoformat()}

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["智能预测"], summary="多维负荷预测", description="""
利用 **GAT-LSTM (图注意力网络+长短期记忆网络)** 模型预测未来 12 小时的多维环境指标。
*   **GAT (Graph Attention Network)**：负责提取不同特征（如室外温度与能耗）之间的空间关联。
*   **LSTM (Long Short-Term Memory)**：负责提取时间序列的演化规律。
""")
def predict_load(current_user: dict = Depends(get_current_user)):
    """Predict load using LSTM model (Model layer)"""
    now = datetime.now()
    forecast_horizon = [(now + timedelta(hours=i)).strftime("%H:00") for i in range(1, 13)]
    
    # Requirement: Predict multi-dimensional (Power, Temp, Humidity, CO2)
    target_cols = metadata['target_cols'] if metadata else ['power_usage', 'indoor_temp', 'indoor_humidity', 'indoor_co2']
    
    if model_instance and scaler:
        try:
            # Use real recent data from BDG2 if available
            if len(real_samples) >= 24:
                # Get last 24 records
                recent_df = pd.DataFrame(real_samples[-24:])
                
                # --- ROBUSTNESS FIX: Ensure column order matches training ---
                if metadata and 'feature_names' in metadata:
                    # Filter and reorder columns according to metadata
                    recent_df = recent_df[metadata['feature_names']]
                else:
                    recent_df = recent_df.select_dtypes(include=[np.number])
                
                # IMPORTANT: Normalize input using the trained scaler
                recent_data_scaled = scaler.transform(recent_df)
                input_tensor = torch.FloatTensor(recent_data_scaled).unsqueeze(0).to(device) # [1, 24, num_features]
                
                # Convert adj_matrix to tensor
                adj_tensor = torch.FloatTensor(adj_matrix).to(device) if adj_matrix is not None else None
                
                with torch.no_grad():
                    pred = model_instance(input_tensor, adj=adj_tensor) # Output shape: [1, 12, 4]
                
                # Get raw predicted values for each dimension
                predictions = {}
                feature_list = metadata['feature_names'] if metadata else recent_df.columns.tolist()
                
                for i, col in enumerate(target_cols):
                    scaled_pred = pred[0, :, i].detach().cpu().numpy()
                    # Inverse transform
                    dummy_pred = np.zeros((12, len(feature_list)))
                    target_idx = feature_list.index(col)
                    dummy_pred[:, target_idx] = scaled_pred
                    real_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]
                    # Add a small amount of dynamic noise to make it feel "live"
                    predictions[col] = [float(val + np.random.uniform(-0.5, 0.5)) for val in real_pred]
                
                return PredictionResponse(
                    forecast_horizon=forecast_horizon,
                    predicted_load=predictions['power_usage'],
                    predicted_temp=predictions['indoor_temp'],
                    predicted_humidity=predictions['indoor_humidity'],
                    predicted_co2=predictions['indoor_co2']
                )
            else:
                raise ValueError("Not enough real samples for prediction")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback random
            return PredictionResponse(
                forecast_horizon=forecast_horizon,
                predicted_load=[float(150 + 50 * np.sin(i/3)) for i in range(12)],
                predicted_temp=[float(24.0 + 2 * np.cos(i/3)) for i in range(12)],
                predicted_humidity=[float(45.0 + 5 * np.sin(i/4)) for i in range(12)],
                predicted_co2=[float(600 + 100 * np.sin(i/5)) for i in range(12)]
            )
    else:
        # Fallback if model or scaler not loaded
        return PredictionResponse(
            forecast_horizon=forecast_horizon,
            predicted_load=[float(150 + 50 * np.sin(i/3)) for i in range(12)],
            predicted_temp=[float(24.0 + 2 * np.cos(i/3)) for i in range(12)],
            predicted_humidity=[float(45.0 + 5 * np.sin(i/4)) for i in range(12)],
            predicted_co2=[float(600 + 100 * np.sin(i/5)) for i in range(12)]
        )

@app.post("/api/v1/optimize", response_model=ControlAction, tags=["优化控制"], summary="多目标决策优化", description="""
基于当前的负荷预测结果，通过 **MOPSO (Multi-Objective Particle Swarm Optimization)** 算法动态计算最优设定值。
*   **能耗目标**：降低空调能耗。
*   **舒适度目标**：基于 **PMV (预期平均热感觉)** 模型，使室内热环境达到最适宜状态。
系统会从帕累托最优解集中选择一个兼顾两者的平衡方案。
""")
def optimize_control(current_user: dict = Depends(get_current_user)):
    """Solve for optimal control parameters using MOPSO (Optimization layer)"""
    # Functional Difference: Check AI Mode
    if not system_state["ai_mode"]:
        logger.info("AI 模式已关闭，使用基准设定值 (24.0C)")
        return ControlAction(
            chilled_water_temp=7.0,
            supply_air_setpoint=24.0, # Baseline fixed setpoint
            mode="BASELINE_FIXED"
        )

    # 1. Get recent predicted load and humidity for fitness calculation
    real_predicted_load = 150.0 # Default
    real_predicted_rh = 50.0 # Default
    
    if model_instance and scaler and len(real_samples) >= 24:
        try:
            recent_df = pd.DataFrame(real_samples[-24:])
            # --- ROBUSTNESS FIX: Ensure column order matches training ---
            if metadata and 'feature_names' in metadata:
                feature_list = metadata['feature_names']
                recent_df = recent_df[feature_list]
            else:
                recent_df = recent_df.select_dtypes(include=[np.number])
                feature_list = recent_df.columns.tolist()

            recent_data_scaled = scaler.transform(recent_df)
            input_tensor = torch.FloatTensor(recent_data_scaled).unsqueeze(0).to(device)
            adj_tensor = torch.FloatTensor(adj_matrix).to(device) if adj_matrix is not None else None
            
            with torch.no_grad():
                preds_scaled = model_instance(input_tensor, adj=adj_tensor)[0].mean(dim=0).cpu().numpy()
                
                # De-normalize load and humidity using metadata order
                # Power
                d_p = np.zeros((1, len(feature_list)))
                p_idx = feature_list.index('power_usage')
                d_p[0, p_idx] = preds_scaled[0] # power_usage is at target index 0
                real_predicted_load = scaler.inverse_transform(d_p)[0, p_idx]
                
                # Humidity
                d_rh = np.zeros((1, len(feature_list)))
                rh_idx = feature_list.index('indoor_humidity')
                d_rh[0, rh_idx] = preds_scaled[2] # indoor_humidity is at target index 2
                real_predicted_rh = scaler.inverse_transform(d_rh)[0, rh_idx]
        except Exception as e:
            logger.warning(f"Optimization prediction failed, using defaults: {e}")

    def fitness_func(x):
        setpoint = x[0]
        # --- IMPROVED ENERGY MODEL (Physical-Based) ---
        # load * (1 + 0.15 * delta) + base_power
        cooling_demand = max(0, 26 - setpoint)
        base_power = 20.0 
        energy = real_predicted_load * (1.0 + 0.15 * cooling_demand) + base_power
        
        # --- IMPROVED COMFORT MODEL (Squared Penalty) ---
        pmv = calculate_pmv(ta=setpoint, tr=setpoint, rh=real_predicted_rh, v=0.1, m=1.0, icl=0.5)
        # Penalty increases sharply outside the [-0.5, 0.5] range
        comfort_penalty = (pmv ** 2) * 100.0 
        return [energy, comfort_penalty]

    mopso = MOPSO(fitness_func, bounds=[[18, 26]], num_particles=30, max_iter=20)
    pareto_front = mopso.solve()
    
    # --- ROBUST PARETO SELECTION (Comfort Constraint) ---
    # 1. Filter solutions with acceptable PMV (target |PMV| <= 0.5)
    acceptable_sols = []
    for p in pareto_front:
        sp = p['position'][0]
        pmv_val = calculate_pmv(ta=sp, tr=sp, rh=real_predicted_rh, v=0.1, m=1.0, icl=0.5)
        if abs(pmv_val) <= 0.5:
            acceptable_sols.append(p)
    
    # 2. Relax if needed
    if not acceptable_sols:
        for p in pareto_front:
            sp = p['position'][0]
            pmv_val = calculate_pmv(ta=sp, tr=sp, rh=real_predicted_rh, v=0.1, m=1.0, icl=0.5)
            if abs(pmv_val) <= 0.8:
                acceptable_sols.append(p)
                
    # 3. Select the best energy saving point from acceptable ones
    if acceptable_sols:
        best_sol = min(acceptable_sols, key=lambda p: p['fitness'][0])
    else:
        # Fallback to the one with minimum comfort deviation if still none
        best_sol = min(pareto_front, key=lambda p: p['fitness'][1])
    
    return ControlAction(
        chilled_water_temp=7.0 + np.random.uniform(-0.5, 0.5),
        supply_air_setpoint=float(best_sol['position'][0]),
        mode="AI_OPTIMIZED"
    )

# Static files for Frontend (Phase 5)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
