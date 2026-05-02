# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimization import MOPSO, calculate_pmv

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE and SMAPE for model evaluation"""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Use SMAPE (Symmetric Mean Absolute Percentage Error) to avoid division by zero
    # and handle low values better than standard MAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero for both zero values
    mask = denominator > 0.01 
    if np.any(mask):
        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    else:
        smape = 0.0
    return rmse, smape

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
            
            rmse, smape = calculate_metrics(y_real, y_pred)
            unit = "kW" if "power" in col_name else ("C" if "temp" in col_name else ("%" if "humidity" in col_name else "ppm"))
            print(f" - {col_name}: RMSE: {rmse:.4f} {unit}, SMAPE: {smape:.2f}%")

    # Storage for results
    ai_setpoints = []
    ai_energy = []
    ai_comfort = []
    
    baseline_setpoints = [24.0] * steps
    baseline_energy = []
    baseline_comfort = []
    
    power_idx_in_y = target_cols.index('power_usage')
    power_target_idx = target_idxs[power_idx_in_y]
    rh_idx_in_y = target_cols.index('indoor_humidity')
    rh_target_idx = target_idxs[rh_idx_in_y]
    out_temp_idx_in_y = target_cols.index('outdoor_temp')
    out_temp_target_idx = target_idxs[out_temp_idx_in_y]

    print(f"\nStarting CSV-based Backtest for {steps} steps...")
    
    for i in range(steps):
        # 1. AI Control Strategy
        with torch.no_grad():
            current_X = X_test[i:i+1] # Already on device if CUDA
            preds_scaled_cpu = model(current_X, adj=adj)[0].cpu().numpy()
            
            real_predicted_load_12 = []
            real_predicted_rh_12 = []
            real_predicted_out_temp_12 = []
            
            for t in range(12):
                dummy_p = np.zeros((1, num_features))
                dummy_p[0, power_target_idx] = preds_scaled_cpu[t, power_idx_in_y]
                real_predicted_load_12.append(scaler.inverse_transform(dummy_p)[0, power_target_idx])
                
                dummy_rh = np.zeros((1, num_features))
                dummy_rh[0, rh_target_idx] = preds_scaled_cpu[t, rh_idx_in_y]
                real_predicted_rh_12.append(scaler.inverse_transform(dummy_rh)[0, rh_target_idx])
                
                dummy_out = np.zeros((1, num_features))
                dummy_out[0, out_temp_target_idx] = preds_scaled_cpu[t, out_temp_idx_in_y]
                real_predicted_out_temp_12.append(scaler.inverse_transform(dummy_out)[0, out_temp_target_idx])
            
        def hvac_fitness(x):
            total_e = 0.0
            total_c = 0.0
            
            for t in range(12):
                setpoint = x[t * 2]
                v_speed = x[t * 2 + 1]
                
                out_t = real_predicted_out_temp_12[t]
                denom = max(1.0, out_t - 24.0)
                q_demand = real_predicted_load_12[t] * ((max(0, out_t - setpoint) / denom) ** 1.2)
                cop = 3.0 + 0.1 * (setpoint - 18)
                p_fan = 10.0 * (v_speed ** 3)
                base_power = 20.0
                
                e = (q_demand / cop) + p_fan + base_power
                rh = real_predicted_rh_12[t]
                pmv = calculate_pmv(ta=setpoint, tr=setpoint + 1.0, rh=rh, v=v_speed, m=1.1, icl=0.7)
                c = (pmv ** 2) * 50.0 
                
                total_e += e
                total_c += c
            return [total_e, total_c]
            
        search_bounds = []
        for t in range(12):
            search_bounds.extend([[18, 26], [0.1, 1.0]])
            
        mopso = MOPSO(hvac_fitness, search_bounds, num_particles=30, max_iter=20)
        pareto = mopso.solve()
        
        # --- PHASE 6: Consistent Pareto Selection ---
        acceptable_sols = []
        for p in pareto:
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
            
        ai_setpoint = best_sol['position'][0]
        ai_wind_speed = best_sol['position'][1]
        
        # Only execute step 1 and record single-step true energy
        out_t = real_predicted_out_temp_12[0]
        denom = max(1.0, out_t - 24.0)
        q_demand_ai = real_predicted_load_12[0] * ((max(0, out_t - ai_setpoint) / denom) ** 1.2)
        cop_ai = 3.0 + 0.1 * (ai_setpoint - 18)
        p_fan_ai = 10.0 * (ai_wind_speed ** 3)
        actual_ai_energy = (q_demand_ai / cop_ai) + p_fan_ai + 20.0
        
        ai_setpoints.append(ai_setpoint)
        ai_energy.append(actual_ai_energy)
        
        # Record ABSOLUTE PMV for evaluation
        final_pmv = calculate_pmv(ta=ai_setpoint, tr=ai_setpoint + 1.0, rh=real_predicted_rh_12[0], v=ai_wind_speed, m=1.1, icl=0.7)
        ai_comfort.append(abs(final_pmv))
        
        # 2. Baseline Control Strategy (Fixed 24.0C, 0.1m/s)
        base_setpoint = 24.0
        base_v = 0.1
        
        q_demand_base = real_predicted_load_12[0] * ((max(0, out_t - base_setpoint) / denom) ** 1.2)
        cop_base = 3.0 + 0.1 * (base_setpoint - 18)
        p_fan_base = 10.0 * (base_v ** 3)
        base_e = (q_demand_base / cop_base) + p_fan_base + 20.0
        
        pmv_base = calculate_pmv(ta=base_setpoint, tr=base_setpoint + 1.0, rh=real_predicted_rh_12[0], v=base_v, m=1.1, icl=0.7)
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
    
    print("\n" + "="*50)
    print("Calculation Formula: SavingRate = (E_baseline - E_ai) / E_baseline * 100%")
    print("="*50)
    
    print(f"\nBacktest Results Summary ({steps} hours):")
    print(f"- Total Energy (AI):         {total_ai_e:.2f} kW")
    print(f"- Total Energy (Baseline Fixed T=24C, v=0.1m/s): {total_base_e:.2f} kW")
    print(f"- Energy Saving Rate:        {saving_rate:.2f}%")
    print(f"- Avg Comfort Deviation (AI): {avg_ai_c:.2f}")
    print(f"- Avg Comfort Deviation (Base): {avg_base_c:.2f}")
    
    return saving_rate
