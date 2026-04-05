# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimization import MOPSO, calculate_pmv

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE and MAPE for model evaluation"""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Calculate MAPE only for non-zero values to avoid astronomical percentages
    # This is common in HVAC where power can be zero at night
    mask = np.abs(y_true) > 0.1 # Threshold for "significant" power
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    return rmse, mape

def run_backtest(model, X_test, y_test, scaler, target_cols, target_idxs, steps=24, adj=None):
    """
    Perform a data-driven backtest on the test set.
    Compares AI control (MOPSO) with Baseline control (Fixed Setpoint).
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Optimization: Move evaluation datasets to GPU once
    if device.type == 'cuda':
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        if adj is not None:
            adj = adj.to(device)
        print("Evaluation datasets moved to CUDA memory for backtest acceleration.")
    
    # 0. Initial model error evaluation
    with torch.no_grad():
        preds_scaled = model(X_test[:steps], adj=adj).cpu().numpy() # [steps, forecast_len, output_size]
        y_true_scaled = y_test[:steps].cpu().numpy() # [steps, forecast_len, output_size]
        
        num_features = X_test.shape[2]
        output_size = len(target_cols)
        
        print(f"\n[Multi-dimensional Prediction] Performance on Test Set (First {steps} windows):")
        for i, col_name in enumerate(target_cols):
            # De-normalize for real error calculation
            dummy_true = np.zeros((steps * 12, num_features))
            dummy_pred = np.zeros((steps * 12, num_features))
            
            t_idx = target_idxs[i]
            dummy_true[:, t_idx] = y_true_scaled[:, :, i].flatten()
            dummy_pred[:, t_idx] = preds_scaled[:, :, i].flatten()
            
            y_real = scaler.inverse_transform(dummy_true)[:, t_idx]
            y_pred = scaler.inverse_transform(dummy_pred)[:, t_idx]
            
            rmse, mape = calculate_metrics(y_real, y_pred)
            unit = "kW" if "power" in col_name else ("C" if "temp" in col_name else ("%" if "humidity" in col_name else "ppm"))
            print(f" - {col_name}: RMSE: {rmse:.4f} {unit}, MAPE: {mape:.2f}%")

    # Storage for results
    ai_setpoints = []
    ai_energy = []
    ai_comfort = []
    
    baseline_setpoints = [24.0] * steps
    baseline_energy = []
    baseline_comfort = []
    
    # Power usage and Humidity target index for optimization
    power_idx_in_y = target_cols.index('power_usage')
    power_target_idx = target_idxs[power_idx_in_y]
    rh_idx_in_y = target_cols.index('indoor_humidity')
    rh_target_idx = target_idxs[rh_idx_in_y]

    print(f"\nStarting CSV-based Backtest for {steps} steps...")
    
    for i in range(steps):
        # 1. AI Control Strategy
        # Predict load for current sequence
        with torch.no_grad():
            current_X = X_test[i:i+1] # Already on device if CUDA
            # Take the mean for the next horizon
            all_preds_scaled = model(current_X, adj=adj)[0].mean(dim=0).cpu().numpy()
            
            # De-normalize predicted values
            # Load
            dummy_p = np.zeros((1, num_features))
            dummy_p[0, power_target_idx] = all_preds_scaled[power_idx_in_y]
            real_predicted_load = scaler.inverse_transform(dummy_p)[0, power_target_idx]
            
            # Humidity
            dummy_rh = np.zeros((1, num_features))
            dummy_rh[0, rh_target_idx] = all_preds_scaled[rh_idx_in_y]
            real_predicted_rh = scaler.inverse_transform(dummy_rh)[0, rh_target_idx]
            
        def hvac_fitness(x):
            setpoint = x[0]
            # Non-linear energy model: load * (delta ^ 1.2) / 10 + base_cost
            cooling_demand = max(0, 26 - setpoint)
            energy = real_predicted_load * (cooling_demand ** 1.2) / 10.0 + 5.0
            
            # PMV-based comfort
            pmv = calculate_pmv(ta=setpoint, tr=setpoint, rh=real_predicted_rh, v=0.1, m=1.0, icl=0.5)
            comfort_penalty = abs(pmv)
            return [energy, comfort_penalty]
            
        mopso = MOPSO(hvac_fitness, [[18, 26]], num_particles=20, max_iter=10)
        pareto = mopso.solve()
        
        # Strategy selection: Utopia point
        energies = [p['fitness'][0] for p in pareto]
        comforts = [p['fitness'][1] for p in pareto]
        min_e, max_e = min(energies), max(energies)
        min_c, max_c = min(comforts), max(comforts)
        de = max_e - min_e if max_e > min_e else 1.0
        dc = max_c - min_c if max_c > min_c else 1.0
        
        def utopia_dist(p):
            norm_e = (p['fitness'][0] - min_e) / de
            norm_c = (p['fitness'][1] - min_c) / dc
            return np.sqrt(norm_e**2 + norm_c**2)
            
        best_sol = min(pareto, key=utopia_dist)
        ai_setpoint = best_sol['position'][0]
        ai_setpoints.append(ai_setpoint)
        ai_energy.append(best_sol['fitness'][0])
        ai_comfort.append(best_sol['fitness'][1])
        
        # 2. Baseline Control Strategy (Fixed 24.0C)
        base_setpoint = 24.0
        cooling_demand_base = max(0, 26 - base_setpoint)
        base_e = real_predicted_load * (cooling_demand_base ** 1.2) / 10.0 + 5.0
        
        pmv_base = calculate_pmv(ta=base_setpoint, tr=base_setpoint, rh=real_predicted_rh, v=0.1, m=1.0, icl=0.5)
        base_c = abs(pmv_base)
        
        baseline_energy.append(base_e)
        baseline_comfort.append(base_c)

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Setpoint Comparison
    axes[0].plot(ai_setpoints, label='AI Control Setpoint', color='blue', marker='o')
    axes[0].plot(baseline_setpoints, label='Baseline (Fixed 24C)', color='red', linestyle='--')
    axes[0].set_title('Setpoint Temperature Comparison')
    axes[0].set_ylabel('Temperature (C)')
    axes[0].legend()
    
    # Energy Consumption
    axes[1].bar(np.arange(steps) - 0.2, ai_energy, width=0.4, label='AI Energy', color='blue', alpha=0.7)
    axes[1].bar(np.arange(steps) + 0.2, baseline_energy, width=0.4, label='Baseline Energy', color='red', alpha=0.7)
    axes[1].set_title('Energy Consumption per Step')
    axes[1].set_ylabel('Energy (kW)')
    axes[1].legend()
    
    # Cumulative Energy
    axes[2].plot(np.cumsum(ai_energy), label='AI Cumulative Energy', color='blue', linewidth=2)
    axes[2].plot(np.cumsum(baseline_energy), label='Baseline Cumulative Energy', color='red', linewidth=2)
    axes[2].set_title('Cumulative Energy Saving Effect')
    axes[2].set_ylabel('Total Energy (kW)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('src/evaluation_report.png')
    print("Evaluation report saved to 'src/evaluation_report.png'.")
    
    # Statistics
    total_ai_e = sum(ai_energy)
    total_base_e = sum(baseline_energy)
    saving_rate = (total_base_e - total_ai_e) / total_base_e * 100
    avg_ai_c = sum(ai_comfort) / steps
    avg_base_c = sum(baseline_comfort) / steps
    
    print(f"\nBacktest Results Summary ({steps} hours):")
    print(f"- Total Energy (AI): {total_ai_e:.2f} kW")
    print(f"- Total Energy (Baseline): {total_base_e:.2f} kW")
    print(f"- Energy Saving Rate: {saving_rate:.2f}%")
    print(f"- Avg Comfort Deviation (AI): {avg_ai_c:.2f}")
    print(f"- Avg Comfort Deviation (Baseline): {avg_base_c:.2f}")
    
    return saving_rate
