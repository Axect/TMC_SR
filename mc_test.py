import numpy as np
import matplotlib.pyplot as plt
import scienceplots

def generate_and_shift(n_events=10000, n_particles=20, T=0.25):
    """
    Generate TMC (Total Momentum Conservation) data using the Generate and Shift method.
    """
    # 1. Generate initial momentum (Exponential distribution)
    # Randomly generate magnitude p and angle phi
    p_mag_raw = np.random.exponential(scale=T, size=(n_events, n_particles))
    phi_raw = np.random.uniform(0, 2*np.pi, size=(n_events, n_particles))

    # 2. Convert to Cartesian coordinates (px, py)
    px = p_mag_raw * np.cos(phi_raw)
    py = p_mag_raw * np.sin(phi_raw)

    # 3. Apply Shift (core step)
    # Subtract mean momentum per event -> sum becomes 0
    px_shifted = px - np.mean(px, axis=1, keepdims=True)
    py_shifted = py - np.mean(py, axis=1, keepdims=True)

    # 4. Calculate final p, phi
    p_mag_final = np.sqrt(px_shifted**2 + py_shifted**2)
    phi_final = np.arctan2(py_shifted, px_shifted)

    return px_shifted, py_shifted, p_mag_final, phi_final

# --- Execution parameters ---
N = 20
Events = 50000
T_val = 0.25

print(f"Simulation started: N={N}, Events={Events}...")
px, py, p_mag, phi = generate_and_shift(n_events=Events, n_particles=N, T=T_val)

# --- Verification 1: Check momentum conservation ---
sum_px = np.sum(px, axis=1)
sum_py = np.sum(py, axis=1)
print(f"Mean of momentum sum (X): {np.mean(sum_px):.2e} (should be nearly 0)")

# --- Verification 2: Calculate c2{2} value (integrated flow) ---
# Using Q-vector method: Q2 = sum(e^(i*2*phi))
Q2 = np.sum(np.exp(1j * 2 * phi), axis=1)
# 2-particle correlations: <|Q2|^2 - N> / (N(N-1))
c2_2_evts = (np.abs(Q2)**2 - N) / (N * (N - 1))
c2_2_mean = np.mean(c2_2_evts)

print(f"Calculated c2{{2}} (Integrated): {c2_2_mean:.5f}")

# --- Visualize results (p_t distribution) ---
with plt.style.context(['science', 'nature']):
    fig, ax = plt.subplots()
    ax.hist(p_mag.flatten(), bins=100, density=True, alpha=0.6, label='Generated (Shifted)')
    x = np.linspace(0, 2, 100)
    ax.plot(x, (1/T_val)*np.exp(-x/T_val), 'r--', label='Original Input PDF')
    ax.set_title(f"Transverse Momentum Distribution (N={N})")
    ax.set_xlabel(r"$p$ [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig("pt_distribution_shifted.png", dpi=300)
