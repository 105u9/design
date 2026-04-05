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
    def __init__(self, fitness_func, bounds, num_particles=30, max_iter=50):
        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.archive = [] # Pareto front
        
    def is_dominated(self, fit1, fit2):
        # fit1 dominates fit2 if all objectives are better or equal and at least one is strictly better
        return all(f1 <= f2 for f1, f2 in zip(fit1, fit2)) and any(f1 < f2 for f1, f2 in zip(fit1, fit2))

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
        self.archive = new_archive

    def solve(self):
        for _ in range(self.max_iter):
            for p in self.particles:
                # Update velocity and position
                w, c1, c2 = 0.5, 1.5, 1.5
                r1, r2 = np.random.rand(), np.random.rand()
                
                # Select a random leader from archive if available
                if self.archive:
                    leader = self.archive[np.random.randint(len(self.archive))]['position']
                else:
                    leader = p.best_position
                    
                p.velocity = w * p.velocity + c1 * r1 * (p.best_position - p.position) + c2 * r2 * (leader - p.position)
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
    # Example fitness function: f1 = x^2, f2 = (x-2)^2
    def example_fitness(x):
        return [x[0]**2, (x[0]-2)**2]
    
    mopso = MOPSO(example_fitness, [[-10, 10]])
    pareto_front = mopso.solve()
    print(f"Found {len(pareto_front)} points on Pareto front.")
    for p in pareto_front[:5]:
        print(f"Position: {p['position']}, Fitness: {p['fitness']}")
