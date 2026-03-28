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

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

def prepare_sequences(df, target_col, seq_len=24, forecast_len=12):
    # Exclude timestamp and site_id
    numeric_df = df.select_dtypes(include=[np.number])
    data = numeric_df.values
    target_idx = numeric_df.columns.get_loc(target_col)
    
    X, y = [], []
    for i in range(len(data) - seq_len - forecast_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+forecast_len, target_idx])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).unsqueeze(-1)

def run_pipeline():
    print("=== Starting Graduation Project Pipeline ===")
    
    # 1. Data Preprocessing (Phase 2)
    print("\n[Step 1/3] Preprocessing building data (Building Genome Project 2)...")
    building_id = "Panther_office_Karla"
    site_id = "Panther"
    
    try:
        df_raw = load_building_data(building_id, site_id)
        df_cleaned = clean_and_impute(df_raw, method='mean')
        df_normalized, scaler = normalize_data(df_cleaned)
        print(f"Data loading complete for building: {building_id}")
        
        # 2. Algorithm & Model (Phase 3)
        print("\n[Step 2/3] Training LSTM Prediction Model (Encoder-Decoder) on Full Dataset...")
        X, y = prepare_sequences(df_normalized, target_col='power_usage')
        
        # Split into train/test (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        input_size = X.shape[2]
        hidden_size = 64
        output_size = 1 # predicting power_usage
        forecast_len = 12
        
        model = LSTM_ED_Model(input_size, hidden_size, output_size, forecast_len)
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
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                test_out = model(X_test[:100]) # Sample test
                test_loss = criterion(test_out, y_test[:100])
                
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/(len(X_train)/batch_size):.6f}, Val Loss: {test_loss.item():.6f}")
            
        torch.save(model.state_dict(), "src/lstm_model.pth")
        print("Model trained on full dataset and saved to src/lstm_model.pth.")

        # 3. Optimization Strategy (Phase 4)
        print("\n[Step 3/3] Running MOPSO optimization (Energy vs Comfort)...")
        # Use the trained model to predict future load for a given setpoint
        # For simplicity, we define a fitness that uses the model's prediction trend
        model.eval()
        with torch.no_grad():
            last_seq = X[-1:].to(torch.float32)
            predicted_future_load = model(last_seq).mean().item()
            
        def hvac_fitness(x):
            # x[0] is setpoint temperature
            setpoint = x[0]
            # Energy cost depends on predicted load and setpoint
            energy = predicted_future_load * (26 - setpoint) / 8.0 
            # Comfort (PMV-like) distance to 22.5C
            comfort = abs(setpoint - 22.5) 
            return [energy, comfort]
        
        mopso = MOPSO(hvac_fitness, [[18, 26]], num_particles=30, max_iter=20) 
        pareto = mopso.solve()
        print(f"MOPSO solved. Found {len(pareto)} Pareto-optimal control points.")
        print(f"Sample recommendation (Best comfort): {min(pareto, key=lambda p: p['fitness'][1])['position'][0]:.2f}C")

        # 4. CSV Backtest Simulation (Evaluation)
        print("\n[Step 4/4] Starting CSV-based Backtest Simulation (AI vs Baseline)...")
        saving_rate = run_backtest(model, X_test, y_test, scaler, steps=24)
        print(f"\nFinal Saving Rate: {saving_rate:.2f}%")
        print("Backtest simulation complete. Results saved to 'src/evaluation_report.png'.")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Pipeline Complete! Data-driven analysis finished. ===")

def start_api():
    print("\n=== Starting FastAPI Backend ===")
    print("API URL: http://localhost:8000")
    print("Interactive Docs: http://localhost:8000/docs")
    print("Web Dashboard: http://localhost:8000/static/index.html")
    # Add a tip about encoding
    print("Tip: If the web dashboard shows garbled characters, please clear browser cache or use Ctrl+F5.")
    from api import app
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVAC Intelligent Control System - Main Program")
    parser.add_argument("--run", action="store_true", help="Run the full pipeline")
    parser.add_argument("--api", action="store_true", help="Start the backend API")
    
    args = parser.parse_args()
    
    if args.run:
        run_pipeline()
    elif args.api:
        start_api()
    else:
        print("Please use arguments:")
        print("  python main.py --run   # Run the training and optimization pipeline")
        print("  python main.py --api   # Start the backend API")
        run_pipeline()
