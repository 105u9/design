# -*- coding: utf-8 -*-
import os
import argparse
import sys

# Ensure src is in the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocessing import load_data, clean_and_impute, normalize_data, calculate_pcc, load_building_data
from models import LSTM_ED_Model, GraphSageLayer
from optimization import MOPSO
from evaluation import run_backtest
import uvicorn
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

def prepare_sequences(df, target_cols, seq_len=24, forecast_len=12):
    # Exclude timestamp and site_id
    numeric_df = df.select_dtypes(include=[np.number])
    data = numeric_df.values
    target_idxs = [numeric_df.columns.get_loc(col) for col in target_cols]
    
    X, y = [], []
    for i in range(len(data) - seq_len - forecast_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+forecast_len, target_idxs])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

def run_pipeline():
    print("=== Starting Graduation Project Pipeline ===")
    
    # Device Detection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Device Selection: {device}")
    if not cuda_available:
        print("Note: NVIDIA GPU not detected. Using CPU. To enable CUDA, ensure NVIDIA drivers and the correct PyTorch version are installed.")
    else:
        print(f"CUDA Enabled: {torch.cuda.get_device_name(0)}")
    
    # 1. Data Preprocessing (Phase 2)
    print("\n[Step 1/4] Preprocessing building data (Building Genome Project 2)...")
    building_id = "Panther_office_Karla"
    site_id = "Panther"
    
    try:
        df_raw = load_building_data(building_id, site_id)
        df_cleaned = clean_and_impute(df_raw, method='mean')
        df_normalized, scaler = normalize_data(df_cleaned)
        print(f"Data loading complete for building: {building_id}")
        
        # 2. Algorithm & Model (Phase 3)
        # Requirement: Predict multi-dimensional environmental parameters
        target_cols = ['power_usage', 'indoor_temp', 'indoor_humidity', 'indoor_co2']
        print(f"\n[Step 2/4] Training LSTM Prediction Model (Encoder-Decoder) for: {target_cols}")
        X, y = prepare_sequences(df_normalized, target_cols=target_cols)
        
        # Split into train/test (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        input_size = X.shape[2]
        hidden_size = 64
        output_size = len(target_cols) # predicting multi-dimensional
        forecast_len = 12
        
        model = LSTM_ED_Model(input_size, hidden_size, output_size, forecast_len).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Full training on all available data
        epochs = 10 # Increased for real data
        batch_size = 64
        print(f"Starting full training for {epochs} epochs on {len(X_train)} samples...")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(device)
                batch_y = y_train[i:i+batch_size].to(device)
                
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                test_out = model(X_test[:100].to(device)) # Sample test
                test_loss = criterion(test_out, y_test[:100].to(device))
                
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/(len(X_train)/batch_size):.6f}, Val Loss: {test_loss.item():.6f}")
            
        torch.save(model.state_dict(), "src/lstm_model.pth")
        joblib.dump(scaler, "src/data_scaler.pkl")
        print("Model and scaler saved to src/.")

        # 3. Optimization Strategy (Phase 4)
        print("\n[Step 3/4] Running MOPSO optimization (Energy vs Comfort)...")
        # Use the trained model to predict future load for a given setpoint
        model.eval()
        with torch.no_grad():
            # Get real recent data for prediction
            last_seq = X[-1:].to(device)
            # Predict for the power_usage channel (index 0 in output)
            predicted_load_scaled = model(last_seq)[0, :, 0].mean().item()
            
            # De-normalize predicted load properly using scaler
            dummy_pred = np.zeros((1, X.shape[2]))
            target_idxs = [df_normalized.select_dtypes(include=[np.number]).columns.get_loc(col) for col in target_cols]
            power_target_idx = target_idxs[0] # power_usage
            dummy_pred[0, power_target_idx] = predicted_load_scaled
            real_predicted_load = scaler.inverse_transform(dummy_pred)[0, power_target_idx]
            
        def hvac_fitness(x):
            # x[0] is setpoint temperature
            setpoint = x[0]
            # Energy model: Non-linear cooling demand with base cost
            # energy = load * (cooling_delta ^ 1.2) / 10.0 + base_cost
            cooling_demand = max(0, 26 - setpoint)
            energy = real_predicted_load * (cooling_demand ** 1.2) / 10.0 + 5.0
            # Comfort (PMV-like) distance to 22.5C
            comfort = abs(setpoint - 22.5) 
            return [energy, comfort]
        
        mopso = MOPSO(hvac_fitness, [[18, 26]], num_particles=30, max_iter=20) 
        pareto = mopso.solve()
        print(f"MOPSO solved. Found {len(pareto)} Pareto-optimal control points.")
        print(f"Sample recommendation (Best comfort): {min(pareto, key=lambda p: p['fitness'][1])['position'][0]:.2f}C")

        # 4. CSV Backtest Simulation (Evaluation)
        print("\n[Step 4/4] Starting CSV-based Backtest Simulation (AI vs Baseline)...")
        saving_rate = run_backtest(model, X_test, y_test, scaler, target_cols, target_idxs, steps=24)
        print(f"\nFinal Saving Rate: {saving_rate:.2f}%")
        print("Backtest simulation complete. Results saved to 'src/evaluation_report.png'.")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Pipeline Complete! Data-driven analysis finished. ===")

def start_api():
    print("\n=== Starting FastAPI Backend ===")
    print("Local URL: http://localhost:8000")
    print("Web Dashboard: http://localhost:8000/static/index.html")
    print("Interactive Docs: http://localhost:8000/docs")
    print("-" * 50)
    print("Note: Do NOT use http://0.0.0.0:8000 in your browser.")
    print("Please use http://localhost:8000 instead.")
    print("-" * 50)
    # Add a tip about encoding
    print("Tip: If the web dashboard shows garbled characters, please clear browser cache or use Ctrl+F5.")
    from api import app
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

def start_trnsys():
    print("\n=== Starting TRNSYS Bridge (Closed-Loop Control) ===")
    from trnsys_utils import simulate_trnsys_loop
    simulate_trnsys_loop(None, None)

def start_mqtt():
    print("\n=== Starting MQTT IoT Sensor Simulation ===")
    from simulate_iot import simulate_iot_publisher
    simulate_iot_publisher()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVAC Intelligent Control System - Main Program")
    parser.add_argument("--run", action="store_true", help="运行训练与回测全流程 (Run full pipeline)")
    parser.add_argument("--api", action="store_true", help="启动 FastAPI 后端服务 (Start backend API)")
    parser.add_argument("--mqtt", action="store_true", help="启动 MQTT IoT 模拟发布 (Start MQTT IoT Simulation)")
    parser.add_argument("--trnsys", action="store_true", help="启动 TRNSYS 联合仿真接口 (Start TRNSYS Bridge)")
    
    args = parser.parse_args()
    
    if args.run:
        run_pipeline()
    elif args.api:
        start_api()
    elif args.mqtt:
        start_mqtt()
    elif args.trnsys:
        start_trnsys()
    else:
        print("请使用以下参数运行程序 (Please use arguments):")
        print("  python main.py --run      # 运行训练与优化回测流程")
        print("  python main.py --api      # 启动监控后端与可视化大屏")
        print("  python main.py --mqtt     # 启动工业物联网 (MQTT) 模拟数据流")
        print("  python main.py --trnsys   # 启动 TRNSYS 动态闭环仿真接口")
        # Default to run pipeline if no args provided to avoid confusion
        run_pipeline()
