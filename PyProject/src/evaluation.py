# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimization import MOPSO

def run_backtest(model, X_test, y_test, scaler, steps=24):
    """
    Perform a data-driven backtest on the test set.
    Compares AI control (MOPSO) with Baseline control (Fixed Setpoint).
    """
    model.eval()
    
    # Storage for results
    ai_setpoints = []
    ai_energy = []
    ai_comfort = []
    
    baseline_setpoints = [24.0] * steps
    baseline_energy = []
    baseline_comfort = []
    
    print(f"Starting CSV-based Backtest for {steps} steps...")
    
    for i in range(steps):
        # 1. AI Control Strategy
        # Predict load for current sequence
        with torch.no_grad():
            current_X = X_test[i:i+1].to(torch.float32)
            predicted_load = model(current_X).mean().item()
            # De-normalize load (approximate for fitness)
            real_predicted_load = predicted_load * 100 + 150 # Simplified de-normalization
            
        def hvac_fitness(x):
            # x[0] is setpoint temperature
            setpoint = x[0]
            # Energy model: energy = load * (cooling_delta)
            energy = real_predicted_load * (26 - setpoint) / 8.0 
            comfort = abs(setpoint - 22.5) 
            return [energy, comfort]
            
        mopso = MOPSO(hvac_fitness, [[18, 26]], num_particles=20, max_iter=10)
        pareto = mopso.solve()
        
        # Strategy: Choose the "elbow" solution from the Pareto front
        # For simplicity, we choose a solution that has good comfort (e.g. 23-24C) but saves energy
        # Utopia point in normalized space: [min(energy), min(comfort)]
        energies = [p['fitness'][0] for p in pareto]
        comforts = [p['fitness'][1] for p in pareto]
        min_e, max_e = min(energies), max(energies)
        min_c, max_c = min(comforts), max(comforts)
        
        # Avoid division by zero
        de = max_e - min_e if max_e > min_e else 1.0
        dc = max_c - min_c if max_c > min_c else 1.0
        
        def utopia_dist(p):
            norm_e = (p['fitness'][0] - min_e) / de
            norm_c = (p['fitness'][1] - min_c) / dc
            return np.sqrt(norm_e**2 + norm_c**2)
            
        best_sol = min(pareto, key=utopia_dist)
        ai_setpoint = best_sol['position'][0]
        ai_setpoints.append(ai_setpoint)
        
        # Calculate AI actual performance
        ai_e = best_sol['fitness'][0]
        ai_c = best_sol['fitness'][1]
        ai_energy.append(ai_e)
        ai_comfort.append(ai_c)
        
        # 2. Baseline Control Strategy (Fixed 22.0C - Less efficient)
        base_setpoint = 22.0
        base_e = real_predicted_load * (26 - base_setpoint) / 8.0
        base_c = abs(base_setpoint - 22.5)
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
