# -*- coding: utf-8 -*-
import numpy as np

# Phase 4:# Phase 4: PMV (Predicted Mean Vote) Model for Comfort Calculation
def calculate_pmv(ta, tr, rh, v, m, icl):
    """
    Standard PMV calculation based on ISO 7730.
    ta: Air Temperature (C)
    tr: Mean Radiant Temperature (C) - often same as Ta
    rh: Relative Humidity (%)
    v: Air Velocity (m/s)
    m: Metabolic Rate (met)
    icl: Clothing Insulation (clo)
    """
    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (ta + 235)) # Water vapor pressure
    
    icl_m2k_w = 0.155 * icl
    m_w_m2 = m * 58.15
    fcl = 1.0 + 0.2 * icl if icl <= 0.5 else 1.05 + 0.1 * icl
    
    hcf = 12.1 * np.sqrt(v)
    taa = ta + 273.15
    tra = tr + 273.15
    
    # Surface temperature of clothing (Iterative, but we use simplified approximation)
    tcl = ta + (35.5 - ta) / (3.5 * icl_m2k_w * 1 + 1) 
    
    # Heat loss components
    hl1 = 3.05e-3 * (5733 - 6.99 * (m_w_m2 - 58.15) - pa)
    hl2 = 0.42 * (m_w_m2 - 58.15 - 58.15) if m_w_m2 > 58.15 else 0
    hl3 = 1.7e-5 * m_w_m2 * (5867 - pa)
    hl4 = 0.0014 * m_w_m2 * (34 - ta)
    hl5 = 3.96e-8 * fcl * (pow(tcl + 273.15, 4) - pow(tra, 4))
    hl6 = fcl * hcf * (tcl - ta)
    
    ts = 0.303 * np.exp(-0.036 * m_w_m2) + 0.028
    l = m_w_m2 - hl1 - hl2 - hl3 - hl4 - hl5 - hl6 # Thermal load
    
    pmv = ts * l
    return pmv

# Multi-Objective Particle Swarm Optimization (MOPSO)
class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_fitness = [float('inf'), float('inf')] # [Energy, Discomfort]

class MOPSO:
    def __init__(self, fitness_func, bounds, num_particles=30, max_iter=50, max_archive_size=50):
        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.max_archive_size = max_archive_size
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.archive = [] # Pareto front
        
        # Velocity clamping limit (e.g., 20% of range)
        self.v_max = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.2
        
    def is_dominated(self, fit1, fit2):
        # fit1 dominates fit2 if all objectives are better or equal and at least one is strictly better
        return all(f1 <= f2 for f1, f2 in zip(fit1, fit2)) and any(f1 < f2 for f1, f2 in zip(fit1, fit2))

    def calculate_crowding_distance(self, archive):
        if len(archive) <= 2:
            for p in archive: p['crowding_distance'] = float('inf')
            return archive
        
        num_objs = len(archive[0]['fitness'])
        for p in archive: p['crowding_distance'] = 0
        
        for m in range(num_objs):
            archive.sort(key=lambda x: x['fitness'][m])
            archive[0]['crowding_distance'] = float('inf')
            archive[-1]['crowding_distance'] = float('inf')
            
            f_min = archive[0]['fitness'][m]
            f_max = archive[-1]['fitness'][m]
            
            if f_max == f_min: continue
            
            for i in range(1, len(archive) - 1):
                archive[i]['crowding_distance'] += (archive[i+1]['fitness'][m] - archive[i-1]['fitness'][m]) / (f_max - f_min)
        return archive

    def update_archive(self, particle):
        fitness = self.fitness_func(particle.position)
        is_p_dominated = False
        new_archive = []
        for arch_p in self.archive:
            if self.is_dominated(arch_p['fitness'], fitness):
                is_p_dominated = True
                new_archive.append(arch_p)
            elif self.is_dominated(fitness, arch_p['fitness']):
                pass # remove arch_p
            else:
                new_archive.append(arch_p)
        
        if not is_p_dominated:
            new_archive.append({'position': np.copy(particle.position), 'fitness': fitness})
            
        # Pruning the archive based on Crowding Distance if it exceeds max size
        if len(new_archive) > self.max_archive_size:
            new_archive = self.calculate_crowding_distance(new_archive)
            new_archive.sort(key=lambda x: x['crowding_distance'], reverse=True)
            new_archive = new_archive[:self.max_archive_size]
            
        self.archive = new_archive

    def solve(self):
        for _ in range(self.max_iter):
            for p in self.particles:
                # Update velocity and position
                w, c1, c2 = 0.5, 1.5, 1.5
                r1, r2 = np.random.rand(), np.random.rand()
                
                # Select a random leader from archive if available
                if self.archive:
                    # Probabilistically select from the top 10% least crowded particles
                    self.calculate_crowding_distance(self.archive)
                    self.archive.sort(key=lambda x: x['crowding_distance'], reverse=True)
                    top_n = max(1, int(len(self.archive) * 0.1))
                    leader = self.archive[np.random.randint(top_n)]['position']
                else:
                    leader = p.best_position
                    
                # Update velocity with clamping
                new_v = w * p.velocity + c1 * r1 * (p.best_position - p.position) + c2 * r2 * (leader - p.position)
                p.velocity = np.clip(new_v, -self.v_max, self.v_max)
                
                # Update position
                p.position = np.clip(p.position + p.velocity, self.bounds[:, 0], self.bounds[:, 1])
                
                # Update best position
                current_fitness = self.fitness_func(p.position)
                if self.is_dominated(current_fitness, p.best_fitness):
                    p.best_position = np.copy(p.position)
                    p.best_fitness = current_fitness
                
                self.update_archive(p)
        return self.archive

# Phase 4: Cosine Similarity for Recommendation
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend_config(current_state, history_states, history_configs):
    similarities = [cosine_similarity(current_state, h_state) for h_state in history_states]
    best_idx = np.argmax(similarities)
    return history_configs[best_idx], similarities[best_idx]

if __name__ == "__main__":
    # Example values for demonstration
    real_predicted_load = 50.0
    real_predicted_rh = 50.0

    def hvac_fitness(x):
        setpoint = x[0]
        # --- PHASE 5 UPGRADE: Consistent Physical Models ---
        # ?????¡Â???????: energy = load * (demand ** 1.2) / 10 + base_power
        cooling_demand = max(0, 26 - setpoint)
        # 20.0kW base operating power
        base_power = 20.0
        energy = real_predicted_load * (cooling_demand ** 1.2) / 10.0 + base_power
        
        # --- PMV Parameters Alignment (ISO 7730) ---
        # Consistent parameters: icl=0.7 (clothing), m=1.1 (metabolic), tr=ta+1.0 (radiant)
        pmv = calculate_pmv(ta=setpoint, tr=setpoint + 1.0, rh=real_predicted_rh, v=0.1, m=1.1, icl=0.7)
        # Penalty increases sharply outside the [-0.5, 0.5] range
        comfort_penalty = (pmv ** 2) * 50.0 
        
        return [energy, comfort_penalty]
        
    mopso = MOPSO(hvac_fitness, [[18, 26]], num_particles=30, max_iter=20)
    pareto = mopso.solve()
    
    # --- ROBUST PARETO SELECTION (Comfort Constraint) ---
    # 1. Filter solutions with acceptable PMV (target |PMV| <= 0.5)
    acceptable_sols = []
    for p in pareto:
        sp = p['position'][0]
        pmv_val = calculate_pmv(ta=sp, tr=sp + 1.0, rh=real_predicted_rh, v=0.1, m=1.1, icl=0.7)
        if abs(pmv_val) <= 0.5:
            acceptable_sols.append(p)
    
    # 2. Relax if needed
    if not acceptable_sols:
        for p in pareto:
            sp = p['position'][0]
            pmv_val = calculate_pmv(ta=sp, tr=sp + 1.0, rh=real_predicted_rh, v=0.1, m=1.1, icl=0.7)
            if abs(pmv_val) <= 0.8:
                acceptable_sols.append(p)
                
    # 3. Select the best energy saving point from acceptable ones
    if acceptable_sols:
        best_sol = min(acceptable_sols, key=lambda p: p['fitness'][0])
    else:
        # Fallback to the one with minimum comfort deviation if still none
        best_sol = min(pareto, key=lambda p: p['fitness'][1])
        
    ai_setpoint = best_sol['position'][0]
    print(f"Optimized HVAC Setpoint: {ai_setpoint:.2f}C")
