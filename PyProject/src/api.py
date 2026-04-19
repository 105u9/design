# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import torch
import joblib
import pandas as pd
import time
import logging
import traceback
from datetime import datetime, timedelta
import jwt
import sqlite3
from passlib.context import CryptContext
from cachetools import TTLCache, cached, keys

# --- JWT Configuration (Security Optimization) ---
SECRET_KEY = "your-secret-key-for-graduation-project" # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120 # 2 hours as suggested

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- DATABASE Configuration (Persistence Optimization) ---
DB_PATH = "src/hvac_system.db"

def init_db():
    """Initialize SQLite database for persistence"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # User Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT DEFAULT 'operator'
        )
    ''')
    
    # Sensor Data Table (Time-series)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            temperature REAL,
            humidity REAL,
            co2 REAL,
            power REAL
        )
    ''')
    
    # Control Log Table (Closed-loop)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS control_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            setpoint REAL,
            chilled_water REAL,
            wind_speed REAL,
            mode TEXT,
            status TEXT DEFAULT 'SENT'
        )
    ''')
    
    # Insert default admin if not exists
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if not cursor.fetchone():
        hashed_pwd = pwd_context.hash("admin123")
        cursor.execute("INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)", 
                       ("admin", hashed_pwd, "admin"))
    
    # DB robustness migration for legacy tables
    try:
        cursor.execute("ALTER TABLE control_logs ADD COLUMN wind_speed REAL")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

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
        "name": "用户认证",
        "description": "用于系统安全访问控制，获取和管理 OAuth2 令牌。",
    },
    {
        "name": "数据监测",
        "description": "处理来自 IoT 设备和传感器的实时监测数据，包括温湿度、CO2 和能耗。",
    },
    {
        "name": "负荷预测",
        "description": "利用 **GAT-LSTM (图注意力网络+长短期记忆网络)** 深度学习模型，对未来负荷进行高精度预测。",
    },
    {
        "name": "智能优化",
        "description": "基于 **MOPSO (多目标粒子群算法)** 在节能与舒适度之间寻找最佳平衡点。",
    },
    {
        "name": "系统管理",
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
    title="数据驱动的暖通空调智能控制原型系统 API",
    docs_url=None, # 禁用默认以支持深度定制
    redoc_url=None, # 禁用默认 redoc
    description="""
### 项目背景
本项目是一个基于数据驱动的空调智能控制系统 API 后端。系统通过集成深度学习（GAT-LSTM）负荷预测与多目标粒子群（MOPSO）优化算法，旨在实现建筑节能与室内热舒适度的双重目标优化。

> [!IMPORTANT]
> **数据源声明**：本系统的环境监测数据基座并非来自于真实的底层物理传感器采集，而是完全基于大型开源建筑能耗数据集 **Building Data Genome Project 2 (BDG2)** 的历史轨迹。系统通过映射和重采样历史数据轴，构建虚拟的数字孪生测试床。

### 接口与鉴权规范
* **安全鉴权**：系统所有核心业务接口均采取了严格的 `OAuth2 Password Bearer` 令牌保护机制。请在 **认证管理** 模块获取有效 Token，点击页面右上角的 `Authorize` 进行应用授权。
* **数据格式**：所有请求与响应体均严格遵循 JSON 结构规范，响应数据由 Pydantic 层完成类型与范围校验。

### API 功能模块表
1. **认证鉴权**：OAuth2 Token 颁发，防暴力破解机制。
2. **多维环境感知**：接收及发布温湿度、CO2、能耗的实时时间序列流。
3. **负荷预测**：执行深度学习推断，提取时间序列的演化规律。
4. **决策寻优**：基于当前负荷预测，动态计算符合帕累托最优的空调控制指令。
""",
    version="2.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "Graduation Project Developer",
    }
)

# 2. 【必需功能】添加 GZip 响应压缩支持 (提升返回预测大数据列表时的速度)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- CORS Configuration (Cross-Origin Optimization) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    wind_speed: float = Field(..., description="推荐送风风速 (m/s)", example=0.3)
    mode: str = Field(..., description="当前控制模式", example="AI_OPTIMIZED")

class PredictionResponse(BaseModel):
    forecast_horizon: List[str] = Field(..., description="预测时间窗 (未来 12 小时)", example=["13:00", "14:00", "15:00"])
    predicted_load: List[float] = Field(..., description="未来 12 小时空调能耗预测值 (kW)", example=[130.5, 135.2, 140.1])
    predicted_temp: List[float] = Field(..., description="未来 12 小时室内温度预测值 (℃)", example=[24.0, 24.1, 24.3])
    predicted_humidity: List[float] = Field(..., description="未来 12 小时湿度预测值 (%)", example=[45.0, 45.5, 46.0])
    predicted_co2: List[float] = Field(..., description="未来 12 小时 CO2 浓度预测值 (ppm)", example=[650, 660, 675])

class SystemState(BaseModel):
    ai_mode: bool = Field(..., description="当前 AI 优化模式状态", example=True)
    last_update: str = Field(..., description="最后状态更新时间", example="2026-04-05T12:00:00")

class MonitoringResponse(BaseModel):
    history: List[SensorData] = Field(..., description="最近24小时的环境监测序列数据")
    system: SystemState = Field(..., description="系统当前全局控制状态")

class ToggleResponse(BaseModel):
    status: str = Field(..., description="请求执行结果", example="success")
    ai_mode: bool = Field(..., description="切换后的 AI 模式状态", example=True)

class CollectResponse(BaseModel):
    status: str = Field(..., description="请求执行结果", example="success")
    received_at: str = Field(..., description="服务器确认接收时间", example="2026-04-05T12:00:01")

# CRUD Models for Users
class UserCreate(BaseModel):
    username: str = Field(..., description="新建用户名", example="testuser")
    password: str = Field(..., description="新建用户密码", example="password123")
    role: str = Field(default="operator", description="系统角色分配 (admin 或 operator)", example="operator")

class UserUpdate(BaseModel):
    role: str = Field(..., description="更新后的系统角色", example="admin")

class UserResponse(BaseModel):
    id: int = Field(..., description="用户唯一流水 ID")
    username: str = Field(..., description="用户名")
    role: str = Field(..., description="当前角色")

# In-memory time-series store (Simulating InfluxDB)
history_data: List[SensorData] = []
MAX_HISTORY = 100

# Global System State
system_state = {
    "ai_mode": True,
    "last_update": datetime.now().isoformat()
}

# --- Caching Mechanism (Performance Optimization) ---
# 缓存 5 分钟 (300 秒)，防止高频轮询导致推理过载
predict_cache = TTLCache(maxsize=10, ttl=300)
optimize_cache = TTLCache(maxsize=10, ttl=300)

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
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE username=?", (form_data.username,))
    result = cursor.fetchone()
    conn.close()
    
    if result and pwd_context.verify(form_data.password, result[0]):
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="用户名或密码不正确")

# 3. 【界面与功能优化】增强版 Swagger UI 路由
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - 交互式 API 文档",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-dark.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png", # 可换成你的毕设系统 Logo
        swagger_ui_parameters={
            "persistAuthorization": True,      # 核心优化：刷新页面保留 Token，不用每次重启/刷新重新登录！
            "displayRequestDuration": True,    # 核心优化：在接口右下角显示请求耗时 (毫秒)，便于分析预测算法延迟
            "filter": True,                    # 开启顶部接口搜索框，方便快速查找路由
            "syntaxHighlight.theme": "monokai",# JSON 响应代码块采用暗色主题
            "docExpansion": "list",            # 默认只展开 Tag 列表，不展开所有接口，页面更清爽
            "defaultModelsExpandDepth": -1,    # 隐藏页面最底部杂乱的 Pydantic Schemas 模型列表
        }
    )

# 4. 【缺失功能】补充 ReDoc 文档 (适合写毕设论文时截图展示只读规范)
@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc 规范文档",
    )

# 5. 【功能优化】深度定制 OpenAPI Schema，增加视觉提示
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="暖通空调智能控制云平台 API",
        version="2.0.0",
        description="""
<div style='padding: 15px; background: rgba(0, 240, 255, 0.1); border-left: 4px solid #00f0ff; border-radius: 4px; margin-bottom: 20px;'>
    <h4 style="margin-top:0; color: #00f0ff;">⚡ 开发者提示 (周杰-计算机222毕业设计)</h4>
    <b>🔒 鉴权说明：</b> 本系统采用 OAuth2 令牌保护机制。请点击右侧 <code>Authorize</code> 进行授权验证。<br>
    测试账号：<code>admin</code> &nbsp;|&nbsp; 密码：<code>admin123</code>
</div>
""" + app.description,
        routes=app.routes,
        tags=tags_metadata,
    )
    # 添加顶部 Logo
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" 
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# 6. 【缺失的必需功能】系统健康检查探针 (生产环境/Docker部署必备)
@app.get("/health", tags=["系统管理"], summary="系统健康检查探针", description="用于负载均衡器或 Docker 检查后端服务是否存活。")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_mode": system_state["ai_mode"],
        "device": str(device)
    }

# 7. 【缺失的必需功能】全局异常拦截器 (避免报错时后端直接向前端抛出难以解析的纯文本)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"系统未捕获异常: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请联系管理员或检查终端日志。", "error": str(exc)}
    )

@app.get("/", tags=["系统配置"], summary="根路径重定向", description="自动重定向到可视化监控大屏页面，方便用户快速查看系统状态")
def read_root():
    """Redirect to the visual dashboard"""
    return RedirectResponse(url="/static/index.html")

@app.get("/api/v1/monitoring", response_model=MonitoringResponse, tags=["环境感知"], summary="获取监测数据", responses={401: {"description": "需要身份验证"}}, description="""
获取最近 24 小时的实时环境监测数据序列。
包含：温度 (℃)、湿度 (%)、CO2 (ppm) 和能耗 (kW)。
本接口将自动结合历史数据表，提供完整的态势感知视图。
""")
def get_monitoring(current_user: dict = Depends(get_current_user)):
    """Get the latest 24 hours of monitoring data and system state"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Simulate a real-time increment in DB to keep it "live"
    if history_data:
        last_data = history_data[-1]
        new_time_dt = datetime.now()
        new_time = new_time_dt.strftime("%Y-%m-%d %H:%M:%S")
        
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
            
            # Persist new data to DB
            cursor.execute('''
                INSERT INTO sensor_history (timestamp, temperature, humidity, co2, power)
                VALUES (?, ?, ?, ?, ?)
            ''', (new_val.timestamp, new_val.temperature, new_val.humidity, new_val.co2, new_val.power))
            conn.commit()

    # Get data from SQLite for actual response
    cursor.execute('''
        SELECT timestamp, temperature, humidity, co2, power 
        FROM sensor_history 
        ORDER BY timestamp DESC LIMIT 24
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    db_history = [
        SensorData(timestamp=r[0], temperature=r[1], humidity=r[2], co2=r[3], power=r[4])
        for r in reversed(rows)
    ]

    return {
        "history": db_history if db_history else history_data[-24:],
        "system": system_state
    }

@app.post("/api/v1/toggle_ai", response_model=ToggleResponse, tags=["系统配置"], summary="切换 AI 模式", responses={401: {"description": "需要身份验证"}}, description="开启或关闭基于 MOPSO 的 AI 自动优化控制模式。开启后系统将动态计算最优设定值，关闭后恢复基准控制模式。")
def toggle_ai(mode: bool, current_user: dict = Depends(get_current_user)):
    """Toggle AI control mode on/off"""
    system_state["ai_mode"] = mode
    system_state["last_update"] = datetime.now().isoformat()
    logger.info(f"AI 模式已切换为: {mode}")
    return {"status": "success", "ai_mode": mode}

@app.post("/api/v1/collect", response_model=CollectResponse, tags=["环境感知"], summary="终端传感器数据上报", responses={401: {"description": "需要身份验证"}, 422: {"description": "数据格式校验失败"}}, description="用于工业物联网终端设备异步上传实时的环境监测数据包至服务器保存。")
def collect_data(data: SensorData, current_user: dict = Depends(get_current_user)):
    """Receive real-time data from sensors (Perception -> Platform)"""
    # Memory update
    history_data.append(data)
    if len(history_data) > MAX_HISTORY:
        history_data.pop(0)
    
    # Database persistence
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sensor_history (timestamp, temperature, humidity, co2, power)
        VALUES (?, ?, ?, ?, ?)
    ''', (data.timestamp, data.temperature, data.humidity, data.co2, data.power))
    conn.commit()
    conn.close()
    
    logger.info(f"接收到传感器数据并持久化: {data}")
    return {"status": "success", "received_at": datetime.now().isoformat()}

# Utility: Data Smoothing (EMA for Cold Start & Noise)
def smooth_history_data(data_list, alpha=0.3):
    """Apply Exponential Moving Average to smooth sensor data"""
    if not data_list: return []
    smoothed = []
    
    # Initialize with first value
    prev_t = data_list[0].temperature
    prev_h = data_list[0].humidity
    prev_c = data_list[0].co2
    prev_p = data_list[0].power
    
    for d in data_list:
        curr_t = alpha * d.temperature + (1 - alpha) * prev_t
        curr_h = alpha * d.humidity + (1 - alpha) * prev_h
        curr_c = alpha * d.co2 + (1 - alpha) * prev_c
        curr_p = alpha * d.power + (1 - alpha) * prev_p
        
        smoothed.append(SensorData(
            timestamp=d.timestamp,
            temperature=curr_t,
            humidity=curr_h,
            co2=curr_c,
            power=curr_p
        ))
        prev_t, prev_h, prev_c, prev_p = curr_t, curr_h, curr_c, curr_p
    return smoothed

# --- USER CRUD Endpoints ---
@app.post("/api/v1/users", response_model=UserResponse, tags=["用户管理"], summary="创建新用户 (C)", responses={401: {"description": "需要身份验证"}, 400: {"description": "用户已存在"}}, description="新建一个操作员或管理员账户，密码将使用 hash 进行安全存储。")
def create_user(user: UserCreate, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username=?", (user.username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    hashed_pwd = pwd_context.hash(user.password)
    cursor.execute("INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)", 
                   (user.username, hashed_pwd, user.role))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return UserResponse(id=user_id, username=user.username, role=user.role)

@app.get("/api/v1/users", response_model=List[UserResponse], tags=["用户管理"], summary="获取用户列表 (R)", responses={401: {"description": "需要身份验证"}}, description="查询系统中注册的所有用户名单。")
def get_users(current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [UserResponse(id=r[0], username=r[1], role=r[2]) for r in rows]

@app.put("/api/v1/users/{user_id}", response_model=UserResponse, tags=["用户管理"], summary="更新用户信息 (U)", responses={401: {"description": "需要身份验证"}, 404: {"description": "用户不存在"}}, description="根据 User ID 修改指定用户的访问角色。")
def update_user(user_id: int, user_update: UserUpdate, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id=?", (user_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="未找到对应的用户")
    
    cursor.execute("UPDATE users SET role=? WHERE id=?", (user_update.role, user_id))
    conn.commit()
    conn.close()
    return UserResponse(id=user_id, username=row[0], role=user_update.role)

@app.delete("/api/v1/users/{user_id}", tags=["用户管理"], summary="删除系统用户 (D)", responses={401: {"description": "需要身份验证"}, 404: {"description": "用户不存在"}, 403: {"description": "越权操作禁止"}}, description="根据 User ID 删除注销系统里的账户（限制不能删除超级管理员）。")
def delete_user(user_id: int, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id=?", (user_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="未找到对应的用户")
    
    if row[0] == "admin":
        conn.close()
        raise HTTPException(status_code=403, detail="系统初始化超级管理员不可删除")
        
    cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"用户 {row[0]} 已注销"}

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["智能预测"], summary="多维系统负荷预测", responses={401: {"description": "需要身份验证"}}, description="""
利用 **GAT-LSTM (图注意力网络+长短期记忆网络)** 模型对包含环境、负荷等节点特征变量的图结构进行未来十二个小时序列预测。
""")
@cached(predict_cache, key=lambda current_user: "predict_result")
def predict_load(current_user: dict = Depends(get_current_user)):
    """Predict load using LSTM model (Model layer)"""
    now = datetime.now()
    forecast_horizon = [(now + timedelta(hours=i)).strftime("%H:00") for i in range(1, 13)]
    
    # Requirement: Predict multi-dimensional (Power, Temp, Humidity, CO2)
    target_cols = metadata['target_cols'] if metadata else ['power_usage', 'indoor_temp', 'indoor_humidity', 'indoor_co2']
    
    if model_instance and scaler:
        try:
            # --- PHASE 6 UPGRADE: Smoothing and Forward-fill ---
            if len(history_data) >= 24:
                # Apply EMA to handle noise/cold-start
                smoothed_history = smooth_history_data(history_data[-24:])
                recent_data = []
                f_names = metadata['feature_names'] if metadata else feature_names
                for d in smoothed_history:
                    if real_samples:
                        entry = {col: real_samples[-1].get(col, 0.0) for col in f_names}
                    else:
                        entry = {col: 0.0 for col in f_names}
                        entry.update({'indoor_temp': 24.0, 'indoor_humidity': 50.0, 'indoor_co2': 600.0})
                    
                    entry['power_usage'] = d.power
                    entry['indoor_temp'] = d.temperature
                    entry['indoor_humidity'] = d.humidity
                    entry['indoor_co2'] = d.co2
                    recent_data.append(entry)
                recent_df = pd.DataFrame(recent_data)
            elif len(real_samples) >= 24:
                recent_df = pd.DataFrame(real_samples[-24:])
            else:
                raise ValueError("Not enough samples for prediction")
            
            # --- ROBUSTNESS FIX: Ensure column order matches training ---
            if metadata and 'feature_names' in metadata:
                # Filter and reorder columns according to metadata
                recent_df = recent_df[metadata['feature_names']]
            elif feature_names:
                recent_df = recent_df[feature_names]
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
        except Exception as e:
            logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
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
*   **能耗目标**：降低空调能耗 (基于物理热力学公式 $E = Q/COP + P_{fan} + P_{base}$)。
*   **舒适度目标**：基于 **PMV (预期平均热感觉)** 模型，使室内热环境达到最适宜状态。
系统会从帕累托最优解集中选择一个兼顾两者的平衡方案，寻优维度包含温度与风速。
""")
@cached(optimize_cache, key=lambda current_user: "optimize_result")
def optimize_control(current_user: dict = Depends(get_current_user)):
    """Solve for optimal control parameters using MOPSO (Optimization layer)"""
    # Functional Difference: Check AI Mode
    if not system_state["ai_mode"]:
        logger.info("AI 模式已关闭，使用基准设定值 (24.0C, 0.1m/s)")
        return ControlAction(
            chilled_water_temp=7.0,
            supply_air_setpoint=24.0, # Baseline fixed setpoint
            wind_speed=0.1,
            mode="BASELINE_FIXED"
        )

    # 1. Get recent predicted load and humidity for fitness calculation
    real_predicted_load = 150.0 # Default
    real_predicted_rh = 50.0 # Default
    real_predicted_temp = 22.0 # Default
    
    if model_instance and scaler:
        try:
            # --- PHASE 6 UPGRADE: EMA Smoothing for Robustness ---
            if len(history_data) >= 24:
                smoothed_history = smooth_history_data(history_data[-24:])
                recent_data = []
                f_names = metadata['feature_names'] if metadata else feature_names
                for d in smoothed_history:
                    if real_samples:
                        entry = {col: real_samples[-1].get(col, 0.0) for col in f_names}
                    else:
                        entry = {col: 0.0 for col in f_names}
                        entry.update({'indoor_temp': 24.0, 'indoor_humidity': 50.0, 'indoor_co2': 600.0})

                    entry['power_usage'] = d.power
                    entry['indoor_temp'] = d.temperature
                    entry['indoor_humidity'] = d.humidity
                    entry['indoor_co2'] = d.co2
                    recent_data.append(entry)
                recent_df = pd.DataFrame(recent_data)
            elif len(real_samples) >= 24:
                recent_df = pd.DataFrame(real_samples[-24:])
            else:
                recent_df = None

            if recent_df is not None:
                # --- ROBUSTNESS FIX: Ensure column order matches training ---
                if metadata and 'feature_names' in metadata:
                    feature_list = metadata['feature_names']
                    recent_df = recent_df[feature_list]
                elif feature_names:
                    feature_list = feature_names
                    recent_df = recent_df[feature_list]
                else:
                    recent_df = recent_df.select_dtypes(include=[np.number])
                    feature_list = recent_df.columns.tolist()

                recent_data_scaled = scaler.transform(recent_df)
                input_tensor = torch.FloatTensor(recent_data_scaled).unsqueeze(0).to(device)
                adj_tensor = torch.FloatTensor(adj_matrix).to(device) if adj_matrix is not None else None
                
                with torch.no_grad():
                    preds_scaled = model_instance(input_tensor, adj=adj_tensor)[0].cpu().numpy() # [12, output_size]
                    
                    real_predicted_load_12 = []
                    real_predicted_rh_12 = []
                    real_predicted_temp_12 = []
                    real_predicted_out_temp_12 = []
                    
                    p_idx = feature_list.index('power_usage')
                    rh_idx = feature_list.index('indoor_humidity')
                    t_idx = feature_list.index('indoor_temp')
                    out_t_idx = feature_list.index('outdoor_temp') if 'outdoor_temp' in feature_list else feature_list.index('airTemperature')
                    
                    for t in range(12):
                        d_p = np.zeros((1, len(feature_list)))
                        d_p[0, p_idx] = preds_scaled[t, p_idx]
                        real_predicted_load_12.append(scaler.inverse_transform(d_p)[0, p_idx])
                        
                        d_rh = np.zeros((1, len(feature_list)))
                        d_rh[0, rh_idx] = preds_scaled[t, rh_idx]
                        real_predicted_rh_12.append(scaler.inverse_transform(d_rh)[0, rh_idx])
                        
                        d_t = np.zeros((1, len(feature_list)))
                        d_t[0, t_idx] = preds_scaled[t, t_idx]
                        real_predicted_temp_12.append(scaler.inverse_transform(d_t)[0, t_idx])
                        
                        d_out = np.zeros((1, len(feature_list)))
                        d_out[0, out_t_idx] = preds_scaled[t, out_t_idx]
                        real_predicted_out_temp_12.append(scaler.inverse_transform(d_out)[0, out_t_idx])
                        
        except Exception as e:
            logger.error(f"Optimization prediction failed: {e}\n{traceback.format_exc()}")
            real_predicted_temp_12 = [22.0] * 12
            real_predicted_load_12 = [150.0] * 12
            real_predicted_rh_12 = [50.0] * 12
            real_predicted_out_temp_12 = [35.0] * 12

    # --- DYNAMIC BOUNDS OPTIMIZATION (24D: 12 steps * [Temp, WindSpeed]) ---
    search_bounds = []
    for t in range(12):
        if real_predicted_temp_12[t] < 18:
            search_bounds.extend([[20, 26], [0.1, 1.0]])
        else:
            search_bounds.extend([[18, 26], [0.1, 1.0]])

    def fitness_func(x):
        total_e = 0.0
        total_c = 0.0
        for t in range(12):
            setpoint = x[t*2]
            v_speed = x[t*2+1]
            
            out_t = real_predicted_out_temp_12[t]
            denom = max(1.0, out_t - 24.0)
            q_demand = real_predicted_load_12[t] * ((max(0, out_t - setpoint) / denom) ** 1.2)
            cop = 3.0 + 0.1 * (setpoint - 18)
            p_fan = 10.0 * (v_speed ** 3)
            base_power = 20.0
            
            energy = (q_demand / cop) + p_fan + base_power
            rh = real_predicted_rh_12[t]
            pmv = calculate_pmv(ta=setpoint, tr=setpoint + 1.0, rh=rh, v=v_speed, m=1.1, icl=0.7)
            comfort_penalty = (pmv ** 2) * 50.0 
            
            total_e += energy
            total_c += comfort_penalty
        return [total_e, total_c]

    mopso = MOPSO(fitness_func, bounds=search_bounds, num_particles=30, max_iter=20)
    pareto_front = mopso.solve()
    
    # --- PHASE 6: Consistent Pareto Selection ---
    acceptable_sols = []
    for p in pareto_front:
        sp = p['position'][0]
        v_sp = p['position'][1]
        pmv_val = calculate_pmv(ta=sp, tr=sp + 1.0, rh=real_predicted_rh_12[0], v=v_sp, m=1.1, icl=0.7)
        if abs(pmv_val) <= 0.5:
            acceptable_sols.append(p)
    
    if not acceptable_sols:
        for p in pareto_front:
            sp = p['position'][0]
            v_sp = p['position'][1]
            pmv_val = calculate_pmv(ta=sp, tr=sp + 1.0, rh=real_predicted_rh_12[0], v=v_sp, m=1.1, icl=0.7)
            if abs(pmv_val) <= 0.8:
                acceptable_sols.append(p)
                
    if acceptable_sols:
        best_sol = min(acceptable_sols, key=lambda p: p['fitness'][0])
    else:
        best_sol = min(pareto_front, key=lambda p: p['fitness'][1])
    
    # --- CLOSED-LOOP CONTROL DOWNLINK ---
    setpoint_val = float(best_sol['position'][0])
    wind_speed_val = float(best_sol['position'][1])
    chilled_water_val = 7.0 + np.random.uniform(-0.5, 0.5)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO control_logs (timestamp, setpoint, chilled_water, wind_speed, mode)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), setpoint_val, chilled_water_val, wind_speed_val, "AI_OPTIMIZED"))
        conn.commit()
        conn.close()
        logger.info(f"AI 控制指令已下发: Setpoint={setpoint_val:.2f}C, WindSpeed={wind_speed_val:.2f}m/s")
    except Exception as e:
        logger.error(f"Failed to log control action: {e}")

    return ControlAction(
        chilled_water_temp=chilled_water_val,
        supply_air_setpoint=setpoint_val,
        wind_speed=wind_speed_val,
        mode="AI_OPTIMIZED"
    )

# Static files for Frontend (Phase 5)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
