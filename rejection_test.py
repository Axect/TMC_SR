"""
TMC Sampling Comparison: Constrained vs Rejection Sampling

This script compares two approaches for sampling momentum distributions
under the Total Momentum Conservation (TMC) constraint:

1. Constrained Sampling (NumPyro NUTS): Sample N-1 particles freely,
   determine N-th particle by constraint, apply Boltzmann factor.

2. Rejection Sampling: Sample all N particles freely, accept only
   those with |Σp| < tolerance.

The comparison analyzes:
- Accuracy (c2{2} vs exact theory)
- Acceptance rate (rejection sampling only)
- Systematic bias from finite tolerance
- Computational efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import gamma as gamma_dist
import scienceplots

# Import from numpyro_test.py
from numpyro_test import calculate_exact_c2_robust, run_numpyro_tmc

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# --- Helper Functions ---

def calculate_c2_from_momenta(px, py):
    """
    Calculate c2{2} from momentum arrays.

    Args:
        px, py: shape (n_events, N) momentum components

    Returns:
        c2_samples: shape (n_events,) c2{2} values per event
    """
    N = px.shape[1]
    phi = np.arctan2(py, px)
    Q2 = np.sum(np.exp(1j * 2 * phi), axis=1)
    c2_samples = (np.abs(Q2)**2 - N) / (N * (N - 1))
    return c2_samples


def theoretical_acceptance_rate(N, T, tolerance):
    """
    Calculate theoretical acceptance rate for rejection sampling.

    Based on CLT: Σp_x and Σp_y are approximately Gaussian with
    variance σ² = N × Var(p_x).

    For Boltzmann distribution in 2D: p ~ Gamma(2, T), φ ~ Uniform(0, 2π)
    - p_x = p × cos(φ)
    - E[p²] = 6T² (for Gamma(2, T))
    - E[cos²(φ)] = 1/2
    - Var(p_x) = E[p²]×E[cos²(φ)] = 3T²

    For 2D Gaussian (Σp_x, Σp_y), P(|Σp| < ε) = 1 - exp(-ε²/(2σ²))

    Args:
        N: number of particles
        T: temperature (GeV)
        tolerance: acceptance threshold for |Σp|

    Returns:
        Theoretical acceptance probability
    """
    sigma_sq = 3 * N * T**2  # Variance of each component sum: N × 3T²
    return 1 - np.exp(-tolerance**2 / (2 * sigma_sq))


# --- Rejection Sampling Implementation ---

def rejection_sampling_tmc(N, T=0.25, n_target=5000, tolerance=0.1,
                           batch_size=100000, max_attempts=100000000,
                           verbose=False):
    """
    Generate TMC-constrained samples using rejection sampling.

    Samples all N particles from Boltzmann distribution, then accepts
    only events where √(Σpx² + Σpy²) < tolerance.

    Args:
        N: Number of particles per event
        T: Temperature parameter (GeV)
        n_target: Target number of accepted samples
        tolerance: Acceptance threshold for total momentum magnitude
        batch_size: Number of events to generate per batch
        max_attempts: Maximum total attempts before giving up
        verbose: Print progress information

    Returns:
        dict with keys:
            'px': (n_accepted, N) array of px values
            'py': (n_accepted, N) array of py values
            'n_accepted': actual number of accepted samples
            'n_total': total number of attempts
            'acceptance_rate': n_accepted / n_total
            'elapsed_time': time in seconds
    """
    start_time = time.time()

    accepted_px = []
    accepted_py = []
    n_total = 0
    n_accepted = 0

    while n_accepted < n_target and n_total < max_attempts:
        # Generate batch of events
        # Magnitude from Gamma(2, T): p ~ p*exp(-p/T)
        p_mag = np.random.gamma(shape=2.0, scale=T, size=(batch_size, N))

        # Angle from Uniform(0, 2π)
        phi = np.random.uniform(0, 2*np.pi, size=(batch_size, N))

        # Convert to Cartesian
        px = p_mag * np.cos(phi)
        py = p_mag * np.sin(phi)

        # Calculate total momentum magnitude
        sum_px = np.sum(px, axis=1)
        sum_py = np.sum(py, axis=1)
        total_p_mag = np.sqrt(sum_px**2 + sum_py**2)

        # Apply acceptance criterion
        mask = total_p_mag < tolerance

        # Collect accepted events
        if np.any(mask):
            accepted_px.append(px[mask])
            accepted_py.append(py[mask])
            n_accepted += np.sum(mask)

        n_total += batch_size

        if verbose and n_total % (batch_size * 10) == 0:
            current_rate = n_accepted / n_total if n_total > 0 else 0
            print(f"  Progress: {n_accepted}/{n_target} accepted, "
                  f"rate={current_rate:.6f}, attempts={n_total:,}")

    elapsed = time.time() - start_time

    # Concatenate results
    if accepted_px:
        all_accepted_px = np.vstack(accepted_px)
        all_accepted_py = np.vstack(accepted_py)
        total_accepted = len(all_accepted_px)
        # Truncate to n_target for return
        px_result = all_accepted_px[:n_target]
        py_result = all_accepted_py[:n_target]
    else:
        px_result = np.array([]).reshape(0, N)
        py_result = np.array([]).reshape(0, N)
        total_accepted = 0

    return {
        'px': px_result,
        'py': py_result,
        'n_accepted': len(px_result),
        'n_total': n_total,
        'total_accepted': total_accepted,  # before truncation
        'acceptance_rate': total_accepted / n_total if n_total > 0 else 0,
        'elapsed_time': elapsed
    }


# --- Comparison Experiment ---

def run_comparison_experiment(N_list, tolerance_list, T=0.25, n_samples=5000,
                              run_constrained=True, verbose=True):
    """
    Run full comparison experiment between constrained and rejection sampling.

    Args:
        N_list: List of particle multiplicities to test
        tolerance_list: List of tolerance values for rejection sampling
        T: Temperature (GeV)
        n_samples: Target number of samples
        run_constrained: Whether to run NumPyro constrained sampling
        verbose: Print progress

    Returns:
        dict with experiment results
    """
    results = {
        'N_list': N_list,
        'tolerance_list': tolerance_list,
        'T': T,
        'n_samples': n_samples,
        'exact_theory': {},       # N -> exact c2{2}
        'constrained': {},        # N -> {'c2_mean', 'c2_err', 'time'}
        'rejection': {},          # (N, tol) -> {'c2_mean', 'c2_err', 'acc_rate', 'time'}
        'theoretical_acc_rate': {}  # (N, tol) -> theoretical acceptance rate
    }

    # Calculate exact theory for all N
    if verbose:
        print("\n=== Calculating Exact Theory ===")
    for N in N_list:
        exact_c2 = calculate_exact_c2_robust(N, T)
        results['exact_theory'][N] = exact_c2
        if verbose:
            print(f"N={N}: exact sqrt(c2) = {np.sqrt(exact_c2):.6f}")

    # Run constrained sampling (NumPyro)
    if run_constrained:
        if verbose:
            print("\n=== Running Constrained Sampling (NumPyro NUTS) ===")
        for N in N_list:
            if verbose:
                print(f"\nN={N}:")
            start = time.time()
            c2_samples = run_numpyro_tmc(N, T, draws=n_samples//4, warmup=1000,
                                         chains=4, target_accept=0.9, verbose=False)
            elapsed = time.time() - start

            mean_c2 = np.mean(c2_samples)
            err_c2 = np.std(c2_samples) / np.sqrt(len(c2_samples))
            sqrt_c2 = np.sqrt(np.abs(mean_c2))
            sqrt_err = err_c2 / (2 * sqrt_c2) if sqrt_c2 > 0 else 0

            results['constrained'][N] = {
                'c2_mean': mean_c2,
                'c2_err': err_c2,
                'sqrt_c2': sqrt_c2,
                'sqrt_err': sqrt_err,
                'time': elapsed,
                'n_samples': len(c2_samples)
            }
            if verbose:
                print(f"  sqrt(c2) = {sqrt_c2:.6f} +/- {sqrt_err:.6f}, time={elapsed:.1f}s")

    # Run rejection sampling for each (N, tolerance) combination
    if verbose:
        print("\n=== Running Rejection Sampling ===")
    for tol in tolerance_list:
        if verbose:
            print(f"\n--- Tolerance = {tol} ---")
        for N in N_list:
            if verbose:
                print(f"\nN={N}, tol={tol}:")

            # Calculate theoretical acceptance rate
            theo_rate = theoretical_acceptance_rate(N, T, tol)
            results['theoretical_acc_rate'][(N, tol)] = theo_rate

            if verbose:
                print(f"  Theoretical acceptance rate: {theo_rate:.6f}")
                expected_attempts = n_samples / theo_rate if theo_rate > 0 else float('inf')
                print(f"  Expected attempts for {n_samples} samples: {expected_attempts:,.0f}")

            # Run rejection sampling
            rej_result = rejection_sampling_tmc(
                N, T, n_target=n_samples, tolerance=tol,
                batch_size=100000, max_attempts=50000000,
                verbose=verbose
            )

            if rej_result['n_accepted'] > 0:
                c2_samples = calculate_c2_from_momenta(rej_result['px'], rej_result['py'])
                mean_c2 = np.mean(c2_samples)
                err_c2 = np.std(c2_samples) / np.sqrt(len(c2_samples))
                sqrt_c2 = np.sqrt(np.abs(mean_c2))
                sqrt_err = err_c2 / (2 * sqrt_c2) if sqrt_c2 > 0 else 0
            else:
                mean_c2 = np.nan
                err_c2 = np.nan
                sqrt_c2 = np.nan
                sqrt_err = np.nan

            results['rejection'][(N, tol)] = {
                'c2_mean': mean_c2,
                'c2_err': err_c2,
                'sqrt_c2': sqrt_c2,
                'sqrt_err': sqrt_err,
                'acceptance_rate': rej_result['acceptance_rate'],
                'time': rej_result['elapsed_time'],
                'n_accepted': rej_result['n_accepted'],
                'n_total': rej_result['n_total']
            }

            if verbose:
                print(f"  Actual acceptance rate: {rej_result['acceptance_rate']:.6f}")
                print(f"  Accepted: {rej_result['n_accepted']}, Attempts: {rej_result['n_total']:,}")
                print(f"  sqrt(c2) = {sqrt_c2:.6f} +/- {sqrt_err:.6f}")
                print(f"  Time: {rej_result['elapsed_time']:.2f}s")

    return results


# --- Visualization Functions ---

def plot_comparison_results(results, save_prefix='rejection_comparison'):
    """
    Create 4 comparison plots:
    1. c2{2} vs N for all methods
    2. Acceptance rate vs N
    3. Relative bias vs tolerance
    4. Computation time comparison
    """
    try:
        plt.style.use(['science', 'nature'])
    except:
        plt.style.use('default')

    N_list = results['N_list']
    tolerance_list = results['tolerance_list']
    T = results['T']

    # Color map for different tolerances
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(tolerance_list)))

    # --- Plot 1: c2{2} vs N ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    # Exact theory
    exact_vals = [np.sqrt(results['exact_theory'][N]) for N in N_list]
    ax1.plot(N_list, exact_vals, 'k-', linewidth=2, label='Exact Theory', zorder=10)

    # Constrained sampling
    if results['constrained']:
        const_vals = [results['constrained'][N]['sqrt_c2'] for N in N_list]
        const_errs = [results['constrained'][N]['sqrt_err'] for N in N_list]
        ax1.errorbar(N_list, const_vals, yerr=const_errs,
                     fmt='s', color='red', markersize=8, capsize=3,
                     label='Constrained (NumPyro)', zorder=9)

    # Rejection sampling for each tolerance
    for i, tol in enumerate(tolerance_list):
        rej_vals = []
        rej_errs = []
        valid_N = []
        for N in N_list:
            data = results['rejection'].get((N, tol))
            if data and not np.isnan(data['sqrt_c2']):
                rej_vals.append(data['sqrt_c2'])
                rej_errs.append(data['sqrt_err'])
                valid_N.append(N)
        if valid_N:
            ax1.errorbar(valid_N, rej_vals, yerr=rej_errs,
                         fmt='o', color=colors[i], markersize=6, capsize=2,
                         label=f'Rejection (tol={tol})', alpha=0.8)

    ax1.set_xlabel(r'$N$ (Multiplicity)')
    ax1.set_ylabel(r'$\sqrt{c_2\{2\}}$')
    ax1.set_title('TMC Flow: Constrained vs Rejection Sampling')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f'{save_prefix}_c2_vs_N.png', dpi=300)
    print(f"Saved: {save_prefix}_c2_vs_N.png")

    # --- Plot 2: Acceptance Rate vs N ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    for i, tol in enumerate(tolerance_list):
        # Theoretical
        theo_rates = [theoretical_acceptance_rate(N, T, tol) for N in N_list]
        ax2.plot(N_list, theo_rates, '--', color=colors[i], linewidth=1.5, alpha=0.7)

        # Measured
        meas_rates = []
        valid_N = []
        for N in N_list:
            data = results['rejection'].get((N, tol))
            if data and data['acceptance_rate'] > 0:
                meas_rates.append(data['acceptance_rate'])
                valid_N.append(N)
        if valid_N:
            ax2.scatter(valid_N, meas_rates, color=colors[i], s=50,
                        label=f'tol={tol}', marker='o', zorder=5)

    ax2.set_xlabel(r'$N$ (Multiplicity)')
    ax2.set_ylabel('Acceptance Rate')
    ax2.set_title('Rejection Sampling Acceptance Rate')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')
    fig2.tight_layout()
    fig2.savefig(f'{save_prefix}_acceptance_rate.png', dpi=300)
    print(f"Saved: {save_prefix}_acceptance_rate.png")

    # --- Plot 3: Relative Bias vs Tolerance ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))

    markers = ['o', 's', '^', 'D', 'v']
    for j, N in enumerate(N_list):
        exact_sqrt = np.sqrt(results['exact_theory'][N])
        biases = []
        valid_tols = []
        for tol in tolerance_list:
            data = results['rejection'].get((N, tol))
            if data and not np.isnan(data['sqrt_c2']):
                rel_bias = (data['sqrt_c2'] / exact_sqrt - 1) * 100  # percent
                biases.append(rel_bias)
                valid_tols.append(tol)
        if valid_tols:
            ax3.plot(valid_tols, biases, marker=markers[j % len(markers)],
                     label=f'N={N}', linewidth=1.5, markersize=6)

    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax3.set_xlabel('Tolerance')
    ax3.set_ylabel('Relative Bias (%)')
    ax3.set_title(r'Bias in $\sqrt{c_2\{2\}}$ vs Tolerance')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(f'{save_prefix}_bias_vs_tolerance.png', dpi=300)
    print(f"Saved: {save_prefix}_bias_vs_tolerance.png")

    # --- Plot 4: Computation Time ---
    fig4, ax4 = plt.subplots(figsize=(7, 5))

    # Constrained times
    if results['constrained']:
        const_times = [results['constrained'][N]['time'] for N in N_list]
        ax4.plot(N_list, const_times, 's-', color='red', markersize=8,
                 linewidth=2, label='Constrained (NumPyro)')

    # Rejection times for each tolerance
    for i, tol in enumerate(tolerance_list):
        rej_times = []
        valid_N = []
        for N in N_list:
            data = results['rejection'].get((N, tol))
            if data and data['time'] > 0:
                rej_times.append(data['time'])
                valid_N.append(N)
        if valid_N:
            ax4.plot(valid_N, rej_times, 'o-', color=colors[i],
                     markersize=6, linewidth=1.5, label=f'Rejection (tol={tol})')

    ax4.set_xlabel(r'$N$ (Multiplicity)')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Computation Time Comparison')
    ax4.set_yscale('log')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, which='both')
    fig4.tight_layout()
    fig4.savefig(f'{save_prefix}_computation_time.png', dpi=300)
    print(f"Saved: {save_prefix}_computation_time.png")

    plt.close('all')


def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    N_list = results['N_list']
    tolerance_list = results['tolerance_list']

    # Header
    header = f"{'N':>5} | {'Exact':>10} | {'Constr.':>10}"
    for tol in tolerance_list:
        header += f" | {f'tol={tol}':>10}"
    print(header)
    print("-" * len(header))

    # Data rows
    for N in N_list:
        exact = np.sqrt(results['exact_theory'][N])
        const = results['constrained'].get(N, {}).get('sqrt_c2', np.nan)

        row = f"{N:>5} | {exact:>10.5f} | {const:>10.5f}"
        for tol in tolerance_list:
            rej = results['rejection'].get((N, tol), {}).get('sqrt_c2', np.nan)
            row += f" | {rej:>10.5f}"
        print(row)

    print("\n" + "-"*80)
    print("ACCEPTANCE RATES (Rejection Sampling)")
    print("-"*80)

    header = f"{'N':>5}"
    for tol in tolerance_list:
        header += f" | {f'tol={tol}':>12}"
    print(header)
    print("-" * len(header))

    for N in N_list:
        row = f"{N:>5}"
        for tol in tolerance_list:
            rate = results['rejection'].get((N, tol), {}).get('acceptance_rate', np.nan)
            row += f" | {rate:>12.6f}"
        print(row)


# --- Main Execution ---

if __name__ == "__main__":
    print("="*60)
    print("TMC Sampling Comparison: Constrained vs Rejection")
    print("="*60)

    # Experiment parameters
    N_list = [10, 20, 50, 100]
    tolerance_list = [0.05, 0.1, 0.2, 0.5, 1.0]
    T_val = 0.25
    n_samples = 5000

    # Run experiment
    results = run_comparison_experiment(
        N_list=N_list,
        tolerance_list=tolerance_list,
        T=T_val,
        n_samples=n_samples,
        run_constrained=True,
        verbose=True
    )

    # Print summary
    print_summary_table(results)

    # Create plots
    print("\n=== Generating Plots ===")
    plot_comparison_results(results, save_prefix='rejection_comparison')

    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
