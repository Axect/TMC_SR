import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import i0, iv
from scipy.integrate import dblquad

# --- 1. High-Precision Simulation (Metropolis) ---

def metropolis_tmc_refined(n_events=10000, n_particles=20, T=0.25, 
                           n_burn_in=100, n_steps_sampling=50):
    """
    Metropolis algorithm with Burn-in for higher accuracy.
    - n_burn_in: Steps to discard (thermalization).
    - n_steps_sampling: Steps to run after burn-in.
    """
    # 1. Initialization (Using Gamma for roughly correct starting point)
    p_mag = np.random.gamma(shape=2, scale=T, size=(n_events, n_particles))
    phi = np.random.uniform(0, 2*np.pi, size=(n_events, n_particles))
    px = p_mag * np.cos(phi)
    py = p_mag * np.sin(phi)
    
    # Enforce sum=0 initially (Start from a valid state)
    px -= np.mean(px, axis=1, keepdims=True)
    py -= np.mean(py, axis=1, keepdims=True)

    # 2. Total steps = Burn-in + Sampling
    # One "step" here means attempting updates for n_particles * 2 times
    total_cycles = n_burn_in + n_steps_sampling
    total_updates = total_cycles * n_particles 
    
    for _ in range(total_updates):
        # A. Pick random pairs (Vectorized across events)
        i = np.random.randint(0, n_particles)
        j = np.random.randint(0, n_particles)
        while i == j:
            j = np.random.randint(0, n_particles)
            
        # B. Propose Kick
        dx = np.random.normal(scale=0.5*T, size=n_events) # Smaller kick size for higher acceptance
        dy = np.random.normal(scale=0.5*T, size=n_events)
        
        # Current State
        pix, piy = px[:, i], py[:, i]
        pjx, pjy = px[:, j], py[:, j]
        E_old = np.sqrt(pix**2 + piy**2) + np.sqrt(pjx**2 + pjy**2)
        
        # Proposed State (Conserves Momentum)
        pix_new, piy_new = pix + dx, piy + dy
        pjx_new, pjy_new = pjx - dx, pjy - dy
        E_new = np.sqrt(pix_new**2 + piy_new**2) + np.sqrt(pjx_new**2 + pjy_new**2)
        
        # C. Acceptance (Metropolis Criterion)
        dE = E_new - E_old
        prob = np.exp(-dE / T)
        accept = np.random.rand(n_events) < prob
        
        # Update
        px[accept, i] = pix_new[accept]
        py[accept, i] = piy_new[accept]
        px[accept, j] = pjx_new[accept]
        py[accept, j] = pjy_new[accept]

    return px, py

# --- 2. High-Precision Theory (Exact Integral) ---

def calculate_exact_c2_robust(N, T=0.25):
    """
    Calculates exact c2{2} with tighter numerical tolerances.
    """
    mean_p2_F = 6 * T**2
    coeff = 2 / ((N - 2) * mean_p2_F)

    def pdf(p):
        return (p / T**2) * np.exp(-p / T)

    def numerator(p2, p1):
        x = coeff * p1 * p2
        return iv(2, x) * pdf(p1) * pdf(p2)

    def denominator(p2, p1):
        x = coeff * p1 * p2
        return i0(x) * pdf(p1) * pdf(p2)

    # Limit set to 20*T (Prob < 1e-9) for stability
    limit = 20 * T
    
    # Increase integration precision
    opts = {'epsabs': 1.49e-8, 'epsrel': 1.49e-8}
    num_val, _ = dblquad(numerator, 0, limit, lambda x: 0, lambda x: limit, **opts)
    den_val, _ = dblquad(denominator, 0, limit, lambda x: 0, lambda x: limit, **opts)
    
    return num_val / den_val

# --- 3. Execution ---
# Parameters for higher precision
N_list = [10, 20, 30, 40, 50, 60, 80, 100]
Events = 20000  # Increased event count for better Precision (Statistical Error)
T_val = 0.25

sim_c2_sqrt = []
sim_c2_err = []
exact_theory = []

print(f"Running high-precision analysis (Events={Events})...")

for n in N_list:
    # Simulation with sufficient burn-in
    # burn_in=200 ensures we forget the initial 'shifted' state
    px_sim, py_sim = metropolis_tmc_refined(n_events=Events, n_particles=n, T=T_val, 
                                            n_burn_in=200, n_steps_sampling=50)
    
    phi_sim = np.arctan2(py_sim, px_sim)
    Q2 = np.sum(np.exp(1j * 2 * phi_sim), axis=1)
    
    # c2 calculation
    c2_evts = (np.abs(Q2)**2 - n) / (n * (n - 1))
    mean_val = np.mean(c2_evts)
    err_val = np.std(c2_evts) / np.sqrt(Events) # Standard Error
    
    # Store results
    sim_c2_sqrt.append(np.sqrt(np.abs(mean_val)))
    sim_c2_err.append(err_val / (2 * np.sqrt(np.abs(mean_val))))

    # Exact Theory
    val_exact = calculate_exact_c2_robust(n, T=T_val)
    exact_theory.append(np.sqrt(val_exact))
    
    print(f"N={n:3d} | Sim: {sim_c2_sqrt[-1]:.5f} +/- {sim_c2_err[-1]:.5f} | Theory: {exact_theory[-1]:.5f}")

# --- 4. Plotting ---
try:
    plt.style.use(['science', 'nature'])
except:
    plt.style.use('default')

fig, ax = plt.subplots(figsize=(6, 4))

# Comparison
ax.plot(N_list, exact_theory, 'r-', linewidth=2, label='Exact Theory')
ax.errorbar(N_list, sim_c2_sqrt, yerr=sim_c2_err, fmt='ko', markersize=4, capsize=3, 
            label=f'Metropolis Sim (Evts={Events})')

ax.set_xlabel(r'$N$ (Multiplicity)')
ax.set_ylabel(r'$\sqrt{c_2\{2\}}$')
ax.set_title(r'Precision Check: TMC Elliptic Flow')
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('tmc_high_precision_check.png', dpi=300)
print("Saved high-precision plot.")
