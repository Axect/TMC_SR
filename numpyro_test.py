import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.special import i0, iv
from scipy.integrate import dblquad
import scienceplots

# Check versions
print(f"Running with NumPyro version: {numpyro.__version__}")
print(f"Running with JAX version: {jax.__version__}")

# Set random seed for full reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Force CPU backend and set up parallel chains
jax.config.update('jax_platform_name', 'cpu')
numpyro.set_host_device_count(16)  # Enable up to 16 parallel chains on CPU


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


# --- 2. NumPyro Model Definition ---

def tmc_model(N, T):
    """
    NumPyro model for TMC-constrained momentum sampling.

    Args:
        N (int): Number of particles.
        T (float): Temperature parameter (GeV).
    """
    # --- A. Priors for N-1 particles ---
    # The single particle distribution f(p) ~ exp(-p/T) in 2D
    # corresponds to a Gamma(alpha=2, beta=1/T) for the magnitude p.
    # Note: NumPyro uses rate parametrization (beta = 1/scale)
    p_mag_rest = numpyro.sample(
        "p_mag_rest",
        dist.Gamma(concentration=2.0, rate=1.0/T).expand([N-1])
    )
    phi_rest = numpyro.sample(
        "phi_rest",
        dist.Uniform(0, 2*jnp.pi).expand([N-1])
    )

    # Convert to Cartesian coordinates
    px_rest = p_mag_rest * jnp.cos(phi_rest)
    py_rest = p_mag_rest * jnp.sin(phi_rest)

    # --- B. Enforce Momentum Conservation ---
    # The N-th particle is fully determined by the sum of the others.
    px_last = -jnp.sum(px_rest)
    py_last = -jnp.sum(py_rest)

    # Calculate magnitude of the N-th particle
    p_mag_last = jnp.sqrt(px_last**2 + py_last**2)

    # --- C. Apply Physics Constraint via Factor ---
    # The N-th particle must also satisfy the Boltzmann weight exp(-p/T).
    # We add log(exp(-p/T)) = -p/T to the model's log-likelihood.
    numpyro.factor("energy_constraint_nth", -p_mag_last / T)

    # --- D. Reconstruction for Output ---
    # Concatenate (N-1) particles with the N-th particle
    px_all = jnp.concatenate([px_rest, jnp.array([px_last])])
    py_all = jnp.concatenate([py_rest, jnp.array([py_last])])

    # Deterministic variables are saved in the trace
    numpyro.deterministic("px", px_all)
    numpyro.deterministic("py", py_all)


# --- 3. NumPyro Simulation Runner ---

def run_numpyro_tmc(N, T=0.25, draws=5000, warmup=2000, chains=4,
                    target_accept=0.95, return_momenta=False, verbose=False):
    """
    Runs the TMC simulation using NumPyro with high-precision settings.

    Args:
        N (int): Number of particles.
        T (float): Temperature parameter (GeV).
        draws (int): Number of samples to draw per chain.
        warmup (int): Number of warmup steps for NUTS sampler.
        chains (int): Number of independent MCMC chains.
        target_accept (float): Target acceptance rate for NUTS (higher = more precise).
        return_momenta (bool): If True, returns flat arrays of px, py for distribution check.
        verbose (bool): If True, print diagnostic information.

    Returns:
        np.array: Array of c2{2} values calculated for each event (if return_momenta=False).
        tuple: (px_flat, py_flat) (if return_momenta=True).
    """
    # Initialize NUTS sampler
    nuts_kernel = NUTS(
        tmc_model,
        target_accept_prob=target_accept,
        init_strategy=numpyro.infer.init_to_median()
    )

    # Create MCMC object
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        progress_bar=True,
        chain_method='parallel'  # Run chains in parallel on CPU
    )

    # Run sampling with fixed random key
    rng_key = jax.random.PRNGKey(RANDOM_SEED)
    mcmc.run(rng_key, N=N, T=T)

    # --- Diagnostics (optional) ---
    if verbose:
        mcmc.print_summary(exclude_deterministic=True)

    # --- Post-Processing ---
    # Get samples from posterior
    samples = mcmc.get_samples()

    # Extract px and py: shape (total_samples, N)
    px_data = np.array(samples["px"])
    py_data = np.array(samples["py"])

    if return_momenta:
        return px_data, py_data

    # Calculate Azimuthal Angles
    phi_data = np.arctan2(py_data, px_data)

    # Calculate Q-vector: Q2 = sum(e^(i*2*phi))
    Q2 = np.sum(np.exp(1j * 2 * phi_data), axis=1)

    # Calculate c2{2} estimator per event
    # Formula: (|Q|^2 - N) / (N(N-1))
    c2_samples = (np.abs(Q2)**2 - N) / (N * (N - 1))

    return c2_samples


# --- 4. Distribution Verification Function ---

def check_and_plot_distribution(N=20, T=0.25, draws=5000, chains=4):
    """
    Runs a simulation to verify if the generated momentum distribution
    matches the theoretical Gamma distribution.
    """
    print(f"\n--- Running Distribution Check (N={N}) ---")
    px, py = run_numpyro_tmc(N, T, draws=draws, warmup=2000, chains=chains,
                             return_momenta=True, verbose=True)

    # Calculate magnitude p_T
    p_mag = np.sqrt(px**2 + py**2).flatten()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    # Histogram of simulation data
    ax.hist(p_mag, bins=100, density=True, alpha=0.5,
            color='black', label='NumPyro Samples')

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

    filename = "numpyro_distribution_check.png"
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Distribution check passed. Plot saved as {filename}")


# --- 5. Main Execution Loop ---

if __name__ == "__main__":

    # Set plotting style
    try:
        plt.style.use(['science', 'nature'])
    except:
        plt.style.use('default')

    # Parameters - High Precision Settings
    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Extended range
    T_val = 0.25
    Draws = 5000    # Samples per chain
    Chains = 16     # Number of independent chains
    Warmup = 2500   # Warmup steps (equivalent to PyMC's tune)
    Target_Accept = 0.95  # Higher acceptance for precision

    # Step 1: Verify Distribution first
    check_and_plot_distribution(N=20, T=T_val, draws=Draws, chains=Chains)

    # Step 2: Run Main Simulation Loop
    results_sim_mean = []
    results_sim_err = []
    results_exact = []

    print(f"\n--- Starting NumPyro Simulation (T={T_val} GeV) ---")
    print(f"{'N':<5} | {'Sim sqrt(c2)':<15} | {'Theory sqrt(c2)':<15}")
    print("-" * 45)

    for n_particle in N_list:
        # 1. Run NumPyro Simulation with high-precision settings
        c2_samples = run_numpyro_tmc(
            N=n_particle,
            T=T_val,
            draws=Draws,
            warmup=Warmup,
            chains=Chains,
            target_accept=Target_Accept,
            verbose=False
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


    # --- 6. Visualization of Results ---

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot Theory
    ax.plot(N_list, results_exact, 'r-', linewidth=2, label='Exact Analytical (Bessel)')

    # Plot Simulation
    total_samples = Draws * Chains
    ax.errorbar(N_list, results_sim_mean, yerr=results_sim_err,
                fmt='ko', markersize=6, capsize=3,
                label=f'NumPyro NUTS ({total_samples} samples)')

    # Plot Settings
    ax.set_xlabel(r'$N$ (Multiplicity)')
    ax.set_ylabel(r'$\sqrt{c_2\{2\}}$')
    ax.set_title(r'TMC Elliptic Flow: NumPyro vs Exact Theory')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

    # Save
    output_filename = 'numpyro_tmc_result.png'
    fig.tight_layout()
    fig.savefig(output_filename, dpi=300)
    print(f"\nFinal result plot saved as {output_filename}")
