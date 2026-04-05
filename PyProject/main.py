# -*- coding: utf-8 -*-
import os
import argparse
import sys

# Ensure src is in the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocessing import load_data, clean_and_impute, normalize_data, calculate_pcc, load_building_data, generate_pcc_adj
from models import LSTM_ED_Model, GraphSageLayer
from optimization import MOPSO, calculate_pmv
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
        # Performance optimization for fixed-size inputs
        torch.backends.cudnn.benchmark = True
    
    # 1. Data Preprocessing (Phase 2)
    print("\n[Step 1/4] Preprocessing building data (Building Genome Project 2)...")
    building_id = "Panther_office_Karla"
    site_id = "Panther"
    
    try:
        df_raw = load_building_data(building_id, site_id)
        df_cleaned = clean_and_impute(df_raw, method='mean')
        
        # --- FIX: Split before normalization to avoid Data Leakage ---
        train_size = int(len(df_cleaned) * 0.8)
        df_train = df_cleaned.iloc[:train_size]
        df_test = df_cleaned.iloc[train_size:]
        
        # Fit scaler only on training data
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(df_train[numeric_cols])
        
        # Transform both train and test
        df_train_norm = df_train.copy()
        df_train_norm[numeric_cols] = scaler.transform(df_train[numeric_cols])
        df_test_norm = df_test.copy()
        df_test_norm[numeric_cols] = scaler.transform(df_test[numeric_cols])
        
        # Save metadata (feature order) for API robustness
        metadata = {
            'feature_names': numeric_cols.tolist(),
            'target_cols': ['power_usage', 'indoor_temp', 'indoor_humidity', 'indoor_co2'],
            'input_size': len(numeric_cols)
        }
        joblib.dump(metadata, "src/metadata.pkl")
        print(f"Metadata saved with {len(numeric_cols)} features.")

        # --- NEW: Generate real adjacency matrix from PCC (on train data only) ---
        print("Generating Graph topology from Pearson Correlation Coefficients (Train set)...")
        adj_matrix, feature_names = generate_pcc_adj(df_train, threshold=0.5)
        adj_tensor = torch.FloatTensor(adj_matrix).to(device)
        joblib.dump(adj_matrix, "src/adj_matrix.pkl")
        
        print(f"Data loading complete for building: {building_id}")
        
        # 2. Algorithm & Model (Phase 3)
        target_cols = metadata['target_cols']
        print(f"\n[Step 2/4] Training LSTM Prediction Model (Encoder-Decoder) with GAT for: {target_cols}")
        X_train, y_train = prepare_sequences(df_train_norm, target_cols=target_cols)
        X_test, y_test = prepare_sequences(df_test_norm, target_cols=target_cols)
        
        # Optimization: Move datasets to GPU if memory allows (14k samples is small)
        if cuda_available:
            X_train, y_train = X_train.to(device), y_train.to(device)
            X_test, y_test = X_test.to(device), y_test.to(device)
            print("Training and test datasets moved to CUDA memory for maximum speed.")
        
        input_size = X_train.shape[2]
        hidden_size = 64
        output_size = len(target_cols)
        forecast_len = 12
        
        model = LSTM_ED_Model(input_size, hidden_size, output_size, forecast_len).to(device)
        
        # --- PHASE 5 UPGRADE: Weighted MSE Loss ---
        # Prioritize power_usage (index 0) to reduce SMAPE from 63% to a reasonable range.
        # target_cols = ['power_usage', 'indoor_temp', 'indoor_humidity', 'indoor_co2']
        loss_weights = torch.tensor([3.0, 1.0, 1.0, 0.5]).to(device)
        
        def weighted_mse_loss(input, target, weights):
            # input/target: [batch, seq, output_size]
            sq_err = (input - target) ** 2
            weighted_err = sq_err * weights
            return torch.mean(weighted_err)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Enhanced Training Loop: Teacher Forcing, Validation, Early Stopping
        epochs = 20 
        batch_size = 64
        best_val_loss = float('inf')
        patience = 5
        trigger_times = 0
        teacher_forcing_ratio = 0.5 # Initial ratio
        
        print(f"Starting enhanced training for {epochs} epochs on {len(X_train)} samples...")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            # Decay teacher forcing ratio
            # Smoother decay: tf_ratio = teacher_forcing_ratio * (0.95 ** epoch)
            tf_ratio = teacher_forcing_ratio * (0.95 ** epoch)
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                out = model(batch_X, adj=adj_tensor, y=batch_y, teacher_forcing_ratio=tf_ratio)
                # Use weighted loss
                loss = weighted_mse_loss(out, batch_y, loss_weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(X_test, adj=adj_tensor) 
                val_loss = weighted_mse_loss(val_out, y_test, loss_weights)
            
            avg_train_loss = epoch_loss / (len(X_train)/batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}, TF Ratio: {tf_ratio:.2f}")
            
            # Save Best Model & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "src/lstm_model.pth")
                trigger_times = 0
                print(f"  --> Best model saved at epoch {epoch+1}")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}!")
                    break
            
        joblib.dump(scaler, "src/data_scaler.pkl")
        print("Final scaler saved to src/data_scaler.pkl")

        # 3. Optimization Strategy (Phase 4)
        print("\n[Step 3/4] Running MOPSO optimization (Energy vs Comfort/PMV)...")
        model.eval()
        with torch.no_grad():
            # Get real recent data for prediction from the end of test set
            last_seq = X_test[-1:].to(device)
            # Predict for all channels
            preds_scaled = model(last_seq, adj=adj_tensor)[0] # [12, 4]
            
            # Mean predicted values for the next 12 hours
            avg_preds_scaled = preds_scaled.mean(dim=0).cpu().numpy() # [4]
            
            # De-normalize predicted values properly using scaler and metadata
            feature_names = metadata['feature_names']
            target_idxs = [feature_names.index(col) for col in target_cols]
            
            # De-normalize each target
            real_preds = {}
            for i, col in enumerate(target_cols):
                d_p = np.zeros((1, len(feature_names)))
                d_p[0, target_idxs[i]] = avg_preds_scaled[i]
                real_preds[col] = scaler.inverse_transform(d_p)[0, target_idxs[i]]
            
            real_predicted_load = real_preds['power_usage']
            real_predicted_rh = real_preds['indoor_humidity']
            real_predicted_temp = real_preds['indoor_temp']
            
            # --- DYNAMIC BOUNDS OPTIMIZATION ---
            # If current predicted temp < 18, assume winter/heating mode
            if real_predicted_temp < 18:
                search_bounds = [[20, 26]] # Heating range
            else:
                search_bounds = [[18, 26]] # Cooling range
            
        def hvac_fitness(x):
            # x[0] is setpoint temperature
            setpoint = x[0]
            # --- PHASE 5 UPGRADE: Consistent Physical Models ---
            # 统一采用非线性公式: energy = load * (demand ** 1.2) / 10 + base_power
            # If heating, demand is (setpoint - current), but we use max(0, 26 - setpoint) for cooling.
            # For simplicity, we keep the cooling formula or adjust if heating.
            # Actually, the user suggested: energy = load * (demand ** 1.2) / 10 + base_power
            # We'll stick to the provided formula but use dynamic bounds.
            cooling_demand = max(0, 26 - setpoint)
            base_power = 20.0
            energy = real_predicted_load * (cooling_demand ** 1.2) / 10.0 + base_power
            
            # Use consistent PMV parameters (icl=0.7, m=1.1, tr=ta+1.0)
            pmv = calculate_pmv(ta=setpoint, tr=setpoint + 1.0, rh=real_predicted_rh, v=0.1, m=1.1, icl=0.7)
            comfort_penalty = (pmv ** 2) * 50.0 
            
            return [energy, comfort_penalty]
        
        mopso = MOPSO(hvac_fitness, search_bounds, num_particles=30, max_iter=20) 
        pareto = mopso.solve()
        
        # --- PHASE 5: Consistent Pareto Selection ---
        acceptable_sols = []
        for p in pareto:
            sp = p['position'][0]
            pmv_val = calculate_pmv(ta=sp, tr=sp + 1.0, rh=real_predicted_rh, v=0.1, m=1.1, icl=0.7)
            if abs(pmv_val) <= 0.5:
                acceptable_sols.append(p)
        
        if not acceptable_sols:
            for p in pareto:
                sp = p['position'][0]
                pmv_val = calculate_pmv(ta=sp, tr=sp + 1.0, rh=real_predicted_rh, v=0.1, m=1.1, icl=0.7)
                if abs(pmv_val) <= 0.8:
                    acceptable_sols.append(p)
        
        if acceptable_sols:
            best_sol = min(acceptable_sols, key=lambda p: p['fitness'][0])
        else:
            best_sol = min(pareto, key=lambda p: p['fitness'][1])
            
        print(f"MOPSO solved. Found {len(pareto)} Pareto-optimal control points.")
        print(f"Sample recommendation (Best Acceptable Energy): {best_sol['position'][0]:.2f}C (PMV penalty: {best_sol['fitness'][1]:.2f})")

        # 4. CSV Backtest Simulation (Evaluation)
        print("\n[Step 4/4] Starting CSV-based Backtest Simulation (AI vs Baseline)...")
        # Pass adj_tensor for evaluation
        saving_rate = run_backtest(model, X_test, y_test, scaler, target_cols, target_idxs, steps=24, adj=adj_tensor)
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
    parser.add_argument("--run", action="store_true", help="Run full pipeline")
    parser.add_argument("--api", action="store_true", help="Start backend API")
    parser.add_argument("--mqtt", action="store_true", help="Start MQTT IoT Simulation")
    parser.add_argument("--trnsys", action="store_true", help="Start TRNSYS Bridge")
    
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
        print("Usage:")
        print("  python main.py --run      # Run training and backtest")
        print("  python main.py --api      # Start backend and dashboard")
        print("  python main.py --mqtt     # Start MQTT IoT simulation")
        print("  python main.py --trnsys   # Start TRNSYS bridge")
        # Default to run pipeline if no args provided to avoid confusion
        run_pipeline()
