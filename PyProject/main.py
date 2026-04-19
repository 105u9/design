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
from torch.utils.data import TensorDataset, DataLoader
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
        
        # --- FIX: Split before imputation to avoid Data Leakage ---
        train_size = int(len(df_raw) * 0.8)
        df_train_raw = df_raw.iloc[:train_size]
        df_test_raw = df_raw.iloc[train_size:]
        
        # Clean and impute separately
        df_train = clean_and_impute(df_train_raw, method='mean')
        df_test = clean_and_impute(df_test_raw, method='mean')
        
        # Fit scaler only on training data
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
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
            'target_cols': ['power_usage', 'indoor_temp', 'indoor_humidity', 'indoor_co2', 'outdoor_temp'],
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
        loss_weights = torch.tensor([3.0, 1.0, 1.0, 0.5, 1.0]).to(device)
        
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
        patience = 10
        trigger_times = 0
        teacher_forcing_ratio = 0.5 # Initial ratio
        
        # Prepare DataLoader for efficiency
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(X_train, y_train)
        # FIX: pin_memory=True only works with CPU tensors. 
        # Since we already moved X_train to GPU if available, pin_memory must be False.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=0, pin_memory=False)
        
        print(f"Starting enhanced training for {epochs} epochs on {len(X_train)} samples...")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            # --- IMPROVED TEACHER FORCING DECAY: Linear decay ---
            # From 0.5 to 0.1 over epochs
            tf_ratio = max(0.1, 0.5 - (0.5 - 0.1) * (epoch / epochs))
            
            for batch_X, batch_y in train_loader:
                if cuda_available: 
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                out = model(batch_X, adj=adj_tensor, y=batch_y, teacher_forcing_ratio=tf_ratio)
                # Use weighted loss
                loss = weighted_mse_loss(out, batch_y, loss_weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(X_test, adj=adj_tensor) 
                val_loss = weighted_mse_loss(val_out, y_test, loss_weights)
            
            avg_train_loss = epoch_loss / len(X_train)
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
        print("\n[Step 3/4] Running MOPSO optimization (Energy vs Comfort/PMV) over 12-hour horizon...")
        model.eval()
        with torch.no_grad():
            last_seq = X_test[-1:].to(device)
            preds_scaled = model(last_seq, adj=adj_tensor)[0] # [12, output_size]
            preds_scaled_cpu = preds_scaled.cpu().numpy()
            
            feature_names = metadata['feature_names']
            target_idxs = [feature_names.index(col) for col in target_cols]
            
            real_preds_12 = {col: [] for col in target_cols}
            for t in range(12):
                for i, col in enumerate(target_cols):
                    d_p = np.zeros((1, len(feature_names)))
                    d_p[0, target_idxs[i]] = preds_scaled_cpu[t, i]
                    val = scaler.inverse_transform(d_p)[0, target_idxs[i]]
                    real_preds_12[col].append(val)
            
            real_predicted_load_12 = real_preds_12['power_usage']
            real_predicted_rh_12 = real_preds_12['indoor_humidity']
            real_predicted_temp_12 = real_preds_12['indoor_temp']
            real_predicted_out_temp_12 = real_preds_12['outdoor_temp']
            
            search_bounds = []
            for t in range(12):
                if real_predicted_temp_12[t] < 18:
                    search_bounds.extend([[20, 26], [0.1, 1.0]])
                else:
                    search_bounds.extend([[18, 26], [0.1, 1.0]])
            
        def hvac_fitness(x):
            total_energy = 0.0
            total_comfort_penalty = 0.0
            
            for t in range(12):
                setpoint = x[t * 2]
                v_speed = x[t * 2 + 1]
                
                out_t = real_predicted_out_temp_12[t]
                denom = max(1.0, out_t - 24.0)
                q_demand = real_predicted_load_12[t] * ((max(0, out_t - setpoint) / denom) ** 1.2)
                
                cop = 3.0 + 0.1 * (setpoint - 18)
                p_fan = 10.0 * (v_speed ** 3)
                
                energy = (q_demand / cop) + p_fan + 20.0
                rh = real_predicted_rh_12[t]
                pmv = calculate_pmv(ta=setpoint, tr=setpoint + 1.0, rh=rh, v=v_speed, m=1.1, icl=0.7)
                comfort_penalty = (pmv ** 2) * 50.0 
                
                total_energy += energy
                total_comfort_penalty += comfort_penalty
                
            return [total_energy, total_comfort_penalty]
        
        mopso = MOPSO(hvac_fitness, search_bounds, num_particles=30, max_iter=20) 
        pareto = mopso.solve()
        
        # --- PHASE 6: Consistent Pareto Selection ---
        acceptable_sols = []
        for p in pareto:
            # Check the first step PMV as the primary metric for accepting the full sequence
            sp = p['position'][0]
            v_sp = p['position'][1]
            pmv_val = calculate_pmv(ta=sp, tr=sp + 1.0, rh=real_predicted_rh_12[0], v=v_sp, m=1.1, icl=0.7)
            if abs(pmv_val) <= 0.5:
                acceptable_sols.append(p)
        
        if not acceptable_sols:
            for p in pareto:
                sp = p['position'][0]
                v_sp = p['position'][1]
                pmv_val = calculate_pmv(ta=sp, tr=sp + 1.0, rh=real_predicted_rh_12[0], v=v_sp, m=1.1, icl=0.7)
                if abs(pmv_val) <= 0.8:
                    acceptable_sols.append(p)
        
        if acceptable_sols:
            best_sol = min(acceptable_sols, key=lambda p: p['fitness'][0])
        else:
            best_sol = min(pareto, key=lambda p: p['fitness'][1])
            
        print(f"MOPSO solved. Found {len(pareto)} Pareto-optimal control points.")
        print(f"Sample recommendation (Best Acceptable Energy): {best_sol['position'][0]:.2f}C, {best_sol['position'][1]:.2f}m/s (PMV penalty: {best_sol['fitness'][1]:.2f})")

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
