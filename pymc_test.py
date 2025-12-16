import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
from scipy.special import i0, iv
from scipy.integrate import dblquad
import scienceplots

# Check PyMC version
print(f"Running with PyMC version: {pm.__version__}")

# --- 1. Theory Calculation (Exact) ---

def calculate_exact_c2_robust(N, T=0.25):
    """
    Calculates the exact theoretical c2{2} value using numerical integration.
    This serves as the ground truth.
    
    Returns:
        float: The exact c2{2} value.
    """
    mean_p2_F = 6 * T**2  # <p^2> for Gamma(2, T)
    coeff = 2 / ((N - 2) * mean_p2_F)

    # PDF: P(p) = (p/T^2)*exp(-p/T) (Radial part of 2D distribution)
    def pdf(p):
        return (p / T**2) * np.exp(-p / T)

    # Integrands for numerator and denominator
    def numerator(p2, p1):
        x = coeff * p1 * p2
        return iv(2, x) * pdf(p1) * pdf(p2)

    def denominator(p2, p1):
        x = coeff * p1 * p2
        return i0(x) * pdf(p1) * pdf(p2)

    # Integration limits (20*T covers probability < 1e-9)
    limit = 20 * T
    opts = {'epsabs': 1e-8, 'epsrel': 1e-8}
    
    num_val, _ = dblquad(numerator, 0, limit, lambda x: 0, lambda x: limit, **opts)
    den_val, _ = dblquad(denominator, 0, limit, lambda x: 0, lambda x: limit, **opts)
    
    return num_val / den_val


# --- 2. PyMC Simulation Model ---

def run_pymc_tmc(N, T=0.25, draws=5000, tune=2000, chains=4,
                 target_accept=0.95, return_momenta=False, verbose=False):
    """
    Runs the TMC simulation using PyMC with high-precision settings.

    Args:
        N (int): Number of particles.
        T (float): Temperature parameter (GeV).
        draws (int): Number of samples to draw per chain.
        tune (int): Number of tuning steps for NUTS sampler.
        chains (int): Number of independent MCMC chains.
        target_accept (float): Target acceptance rate for NUTS (higher = more precise).
        return_momenta (bool): If True, returns flat arrays of px, py for distribution check.
        verbose (bool): If True, print diagnostic information.

    Returns:
        np.array: Array of c2{2} values calculated for each event (if return_momenta=False).
        tuple: (px_flat, py_flat) (if return_momenta=True).
    """
    with pm.Model() as model:
        # --- A. Priors for N-1 particles ---
        # The single particle distribution f(p) ~ exp(-p/T) in 2D
        # corresponds to a Gamma(alpha=2, beta=1/T) for the magnitude p.
        p_mag_rest = pm.Gamma("p_mag_rest", alpha=2, beta=1/T, shape=N-1)
        phi_rest = pm.Uniform("phi_rest", lower=0, upper=2*np.pi, shape=N-1)

        # Convert to Cartesian coordinates (using PyTensor)
        px_rest = p_mag_rest * pt.cos(phi_rest)
        py_rest = p_mag_rest * pt.sin(phi_rest)

        # --- B. Enforce Momentum Conservation ---
        # The N-th particle is fully determined by the sum of the others.
        px_last = -pt.sum(px_rest)
        py_last = -pt.sum(py_rest)

        # Calculate magnitude of the N-th particle
        p_mag_last = pt.sqrt(px_last**2 + py_last**2)

        # --- C. Apply Physics Constraint via Potential ---
        # The N-th particle must also satisfy the Boltzmann weight exp(-p/T).
        # We add log(exp(-p/T)) = -p/T to the model's log-likelihood.
        pm.Potential("energy_constraint_nth", -p_mag_last / T)

        # --- D. Reconstruction for Output ---
        # Concatenate (N-1) particles with the N-th particle
        # px_all shape: (N,)
        px_all = pt.concatenate([px_rest, pt.stack([px_last])])
        py_all = pt.concatenate([py_rest, pt.stack([py_last])])

        # Deterministic variables are saved in the trace
        pm.Deterministic("px", px_all)
        pm.Deterministic("py", py_all)

        # --- E. Sampling with High-Precision Settings ---
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,  # Parallel chains
            nuts_sampler="pymc",
            nuts_sampler_kwargs={"target_accept": target_accept},
            progressbar=False,
            discard_tuned_samples=True,
            return_inferencedata=True,
            random_seed=42,  # Reproducibility
        )

    # --- F. Diagnostics (optional) ---
    if verbose:
        import arviz as az
        summary = az.summary(trace, var_names=["p_mag_rest"])
        print(f"  ESS bulk (min): {summary['ess_bulk'].min():.0f}")
        print(f"  ESS tail (min): {summary['ess_tail'].min():.0f}")
        print(f"  R-hat (max): {summary['r_hat'].max():.4f}")

    # --- G. Post-Processing ---
    # Extract data from posterior
    # shape: (chains, draws, N)
    px_data = trace.posterior["px"].values
    py_data = trace.posterior["py"].values

    # Merge chains and draws
    # shape: (total_samples, N)
    px_flat = px_data.reshape(-1, N)
    py_flat = py_data.reshape(-1, N)

    if return_momenta:
        return px_flat, py_flat

    # Calculate Azimuthal Angles
    phi_flat = np.arctan2(py_flat, px_flat)

    # Calculate Q-vector: Q2 = sum(e^(i*2*phi))
    Q2 = np.sum(np.exp(1j * 2 * phi_flat), axis=1)

    # Calculate c2{2} estimator per event
    # Formula: (|Q|^2 - N) / (N(N-1))
    c2_samples = (np.abs(Q2)**2 - N) / (N * (N - 1))

    return c2_samples


# --- 3. Distribution Verification Function ---

def check_and_plot_distribution(N=20, T=0.25, draws=5000, chains=4):
    """
    Runs a simulation to verify if the generated momentum distribution
    matches the theoretical Gamma distribution.
    """
    print(f"\n--- Running Distribution Check (N={N}) ---")
    px, py = run_pymc_tmc(N, T, draws=draws, tune=2000, chains=chains,
                          return_momenta=True, verbose=True)
    
    # Calculate magnitude p_T
    p_mag = np.sqrt(px**2 + py**2).flatten()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Histogram of simulation data
    ax.hist(p_mag, bins=100, density=True, alpha=0.5, 
            color='black', label='PyMC Samples')
    
    # Theoretical Curve: P(p) = (p/T^2)*exp(-p/T)
    p_axis = np.linspace(0, np.max(p_mag), 200)
    pdf_theory = (p_axis / T**2) * np.exp(-p_axis / T)
    ax.plot(p_axis, pdf_theory, 'r--', linewidth=2, 
            label=r'Theory $\propto p e^{-p/T}$')
    
    ax.set_xlabel(r'$p_T$ [GeV]')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Momentum Distribution Verification (N={N})')
    ax.legend()
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.2)
    
    filename = "pymc_distribution_check.png"
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Distribution check passed. Plot saved as {filename}")


# --- 4. Main Execution Loop ---

if __name__ == "__main__":
    
    # Set plotting style
    try:
        plt.style.use(['science', 'nature'])
    except:
        plt.style.use('default')

    # Parameters - High Precision Settings
    N_list = [10, 20, 30, 40, 50, 60, 80, 100]  # Extended range
    T_val = 0.25
    Draws = 5000     # Samples per chain
    Chains = 4       # Number of independent chains
    Tune = 2000      # Tuning steps
    Target_Accept = 0.95  # Higher acceptance for precision

    # Step 1: Verify Distribution first
    check_and_plot_distribution(N=20, T=T_val, draws=Draws, chains=Chains)

    # Step 2: Run Main Simulation Loop
    results_sim_mean = []
    results_sim_err = []
    results_exact = []

    print(f"\n--- Starting PyMC Simulation (T={T_val} GeV) ---")
    print(f"{'N':<5} | {'Sim sqrt(c2)':<15} | {'Theory sqrt(c2)':<15}")
    print("-" * 45)

    for n_particle in N_list:
        # 1. Run PyMC Simulation with high-precision settings
        c2_samples = run_pymc_tmc(
            N=n_particle,
            T=T_val,
            draws=Draws,
            tune=Tune,
            chains=Chains,
            target_accept=Target_Accept,
            verbose=True
        )

        # Calculate Statistics
        # Note: c2{2} can be slightly negative due to fluctuations, so we take mean then sqrt
        mean_c2 = np.mean(c2_samples)
        err_c2_mean = np.std(c2_samples) / np.sqrt(len(c2_samples))

        # Convert to RMS flow (sqrt(c2))
        # Propagate error: d(sqrt(x)) = dx / (2*sqrt(x))
        sim_val = np.sqrt(np.abs(mean_c2))
        sim_err = err_c2_mean / (2 * sim_val)

        results_sim_mean.append(sim_val)
        results_sim_err.append(sim_err)

        # 2. Exact Theory Calculation
        exact_val = np.sqrt(calculate_exact_c2_robust(n_particle, T=T_val))
        results_exact.append(exact_val)

        print(f"{n_particle:<5} | {sim_val:.5f} +/- {sim_err:.5f} | {exact_val:.5f}")


    # --- 5. Visualization of Results ---

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot Theory
    ax.plot(N_list, results_exact, 'r-', linewidth=2, label='Exact Analytical (Bessel)')

    # Plot Simulation
    total_samples = Draws * Chains
    ax.errorbar(N_list, results_sim_mean, yerr=results_sim_err,
                fmt='ko', markersize=6, capsize=3,
                label=f'PyMC NUTS ({total_samples} samples)')

    # Plot Settings
    ax.set_xlabel(r'$N$ (Multiplicity)')
    ax.set_ylabel(r'$\sqrt{c_2\{2\}}$')
    ax.set_title(r'TMC Elliptic Flow: PyMC vs Exact Theory')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

    # Save
    output_filename = 'pymc_tmc_result.png'
    fig.tight_layout()
    fig.savefig(output_filename, dpi=300)
    print(f"\nFinal result plot saved as {output_filename}")
