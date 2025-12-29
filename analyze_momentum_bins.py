#!/usr/bin/env python3
"""
Momentum-bin based c2{2} analysis for TMC simulations.
Calculates differential flow correlations c2{2}(pT1, pT2) for cross-bin pairs.

Usage:
    python analyze_momentum_bins.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import simulation functions from numpyro_test
from numpyro_test import run_numpyro_tmc

# =============================================================================
# Configuration
# =============================================================================

T_VAL = 0.25                    # Temperature (GeV)
N_RANGE = range(20, 102, 10)    # Coarse grid for initial analysis: [20, 20, ..., 100]

# pT bin boundaries (GeV) - designed for T=0.25
# Mean pT = 2*T = 0.5 GeV
# Experimentally, pT < 0.3 GeV is typically excluded (low tracking efficiency)
# New bins: 0.3-0.5 (low), 0.5-0.8 (mid), 0.8+ (high)
PT_MIN_CUT = 0.3  # Minimum pT cut (GeV)
PT_BINS = [0.3, 0.5, 0.8, np.inf]
BIN_LABELS = ['low', 'mid', 'high']

# MCMC settings (increased for better statistics)
DRAWS = 5000
WARMUP = 2000
CHAINS = 16
TARGET_ACCEPT = 0.95

# Output paths
OUTPUT_DIR = Path("data")
FIG_DIR = Path("figs")


# =============================================================================
# Core Functions
# =============================================================================

def assign_bins(pT: np.ndarray, bin_edges: list) -> np.ndarray:
    """
    Assign particles to pT bins.

    Particles below the minimum pT cut (first bin edge) are assigned bin index -1
    and should be excluded from analysis.

    Args:
        pT: Array of shape (n_samples, N) - transverse momentum magnitudes
        bin_edges: List of bin boundaries [0.3, 0.5, 0.8, inf]

    Returns:
        bin_indices: Array of shape (n_samples, N) with bin index (0, 1, 2)
                     or -1 for particles below minimum pT cut
    """
    # np.digitize with bin_edges[1:-1] gives us [0.5, 0.8], so:
    #   pT < 0.5 -> 0 (low)
    #   0.5 <= pT < 0.8 -> 1 (mid)
    #   pT >= 0.8 -> 2 (high)
    bin_idx = np.digitize(pT, bin_edges[1:-1])

    # Mark particles below minimum pT cut as -1 (excluded)
    # bin_edges[0] is the minimum pT cut (e.g., 0.3 GeV)
    bin_idx = np.where(pT < bin_edges[0], -1, bin_idx)

    return bin_idx


def calculate_binned_c2_vectorized(phi: np.ndarray, pT: np.ndarray,
                                    bin_edges: list, bin_labels: list) -> dict:
    """
    Calculate c2{2} for all bin combinations using vectorized pair counting.

    Physics formula:
        c2{2}(bin_i, bin_j) = <cos(2*(phi_1 - phi_2))>
        where particle 1 is in bin_i, particle 2 is in bin_j

    Using trig identity for vectorization:
        cos(2*delta_phi) = cos(2*phi_1)*cos(2*phi_2) + sin(2*phi_1)*sin(2*phi_2)

    Args:
        phi: Array of shape (n_samples, N) - azimuthal angles
        pT: Array of shape (n_samples, N) - transverse momenta
        bin_edges: Bin boundary list
        bin_labels: List of bin names ['low', 'mid', 'high']

    Returns:
        Dictionary with keys like ('low', 'mid') containing:
            - 'c2_mean': Mean c2{2} across all events
            - 'c2_err': Standard error
            - 'n_pairs_mean': Average number of pairs per event
            - 'n_valid_events': Number of events with valid pairs
    """
    n_samples, N = phi.shape
    n_bins = len(bin_labels)

    # Pre-compute trig functions
    cos2phi = np.cos(2 * phi)  # (n_samples, N)
    sin2phi = np.sin(2 * phi)  # (n_samples, N)

    # Assign bins
    bin_idx = assign_bins(pT, bin_edges)  # (n_samples, N)

    results = {}

    for i in range(n_bins):
        for j in range(n_bins):
            # Create masks for each bin
            mask_i = (bin_idx == i)  # (n_samples, N)
            mask_j = (bin_idx == j)  # (n_samples, N)

            # Count particles in each bin per event
            n_i = mask_i.sum(axis=1)  # (n_samples,)
            n_j = mask_j.sum(axis=1)  # (n_samples,)

            # Sum of cos and sin within each bin (Q-vector components)
            # Qc = sum(cos(2*phi)) for particles in bin
            # Qs = sum(sin(2*phi)) for particles in bin
            Qc_i = np.where(mask_i, cos2phi, 0).sum(axis=1)
            Qs_i = np.where(mask_i, sin2phi, 0).sum(axis=1)
            Qc_j = np.where(mask_j, cos2phi, 0).sum(axis=1)
            Qs_j = np.where(mask_j, sin2phi, 0).sum(axis=1)

            # Sum of cos(2*delta_phi) over all pairs = Qc_i*Qc_j + Qs_i*Qs_j
            sum_cos = Qc_i * Qc_j + Qs_i * Qs_j

            if i == j:
                # Same bin: subtract self-pairs
                # Self contribution: sum_k cos(2*(phi_k - phi_k)) = n_i (each cos(0) = 1)
                sum_cos = sum_cos - n_i
                n_pairs = n_i * (n_i - 1)  # Ordered pairs excluding self
            else:
                # Different bins: all pairs are valid
                n_pairs = n_i * n_j

            # Calculate c2 per event (avoid division by zero)
            valid = n_pairs > 0
            c2_per_event = np.full(n_samples, np.nan)
            c2_per_event[valid] = sum_cos[valid] / n_pairs[valid]

            # Statistics over valid events
            c2_valid = c2_per_event[~np.isnan(c2_per_event)]

            if len(c2_valid) > 0:
                results[(bin_labels[i], bin_labels[j])] = {
                    'c2_mean': np.mean(c2_valid),
                    'c2_err': np.std(c2_valid) / np.sqrt(len(c2_valid)),
                    'n_pairs_mean': np.mean(n_pairs[valid]),
                    'n_valid_events': len(c2_valid)
                }
            else:
                results[(bin_labels[i], bin_labels[j])] = {
                    'c2_mean': np.nan,
                    'c2_err': np.nan,
                    'n_pairs_mean': 0,
                    'n_valid_events': 0
                }

    return results


def calculate_bin_statistics(pT: np.ndarray, bin_edges: list, bin_labels: list) -> dict:
    """
    Calculate statistics about particle distribution in bins.

    Args:
        pT: Array of shape (n_samples, N) - transverse momenta
        bin_edges: Bin boundary list
        bin_labels: List of bin names

    Returns:
        Dictionary with bin population statistics
    """
    bin_idx = assign_bins(pT, bin_edges)
    n_samples, N = pT.shape

    stats = {}

    # First, track excluded particles (pT < minimum cut)
    excluded_mask = (bin_idx == -1)
    excluded_counts = excluded_mask.sum(axis=1)
    stats['_excluded'] = {
        'mean_count': np.mean(excluded_counts),
        'std_count': np.std(excluded_counts),
        'fraction': np.mean(excluded_counts) / N,
    }

    for i, label in enumerate(bin_labels):
        mask = (bin_idx == i)
        counts = mask.sum(axis=1)  # Per-event counts

        # Calculate <p^2> for particles in this bin (TMC ensemble average)
        pT_sq = pT**2
        # For each event, compute mean(p^2) over particles in this bin
        sum_pT_sq = np.where(mask, pT_sq, 0).sum(axis=1)
        # Avoid division by zero for events with empty bins
        valid = counts > 0
        mean_p_sq_per_event = np.zeros(n_samples)
        mean_p_sq_per_event[valid] = sum_pT_sq[valid] / counts[valid]
        # Overall mean <p^2>_Omega for this bin
        mean_p_sq = np.mean(mean_p_sq_per_event[valid]) if valid.sum() > 0 else np.nan

        stats[label] = {
            'mean_count': np.mean(counts),
            'std_count': np.std(counts),
            'fraction': np.mean(counts) / N,
            'empty_events': np.sum(counts == 0),
            'mean_p_sq': mean_p_sq  # <p^2>_Omega for this bin
        }

    return stats


def calculate_approximation_c2_binned(mean_p1_sq: float, mean_p2_sq: float,
                                       N: int, T: float) -> float:
    """
    Calculate the approximation c2{2} for a bin combination.

    Formula: c2{2} = <p1^2>_Omega * <p2^2>_Omega / (2 * (N-2)^2 * <p^2>_F^2)

    Args:
        mean_p1_sq: <p^2>_Omega for bin 1
        mean_p2_sq: <p^2>_Omega for bin 2
        N: Number of particles
        T: Temperature (GeV)

    Returns:
        Approximate c2{2} value
    """
    if np.isnan(mean_p1_sq) or np.isnan(mean_p2_sq):
        return np.nan

    # <p^2>_F = 6*T^2 for Gamma(2, T) distribution
    mean_p2_F = 6 * T**2

    # Approximation formula
    c2_approx = (mean_p1_sq * mean_p2_sq) / (2.0 * (N - 2)**2 * mean_p2_F**2)

    return c2_approx


def calculate_exact_c2_binned(N: int, T: float,
                               p1_min: float, p1_max: float,
                               p2_min: float, p2_max: float) -> float:
    """
    Calculate exact c2{2} for specific momentum bin ranges using numerical integration.

    The exact formula integrates the Bessel function ratio I_2/I_0 over the
    specified momentum ranges.

    Args:
        N: Number of particles
        T: Temperature (GeV)
        p1_min, p1_max: Momentum range for particle 1 (bin_i)
        p2_min, p2_max: Momentum range for particle 2 (bin_j)

    Returns:
        Exact c2{2} value for this bin combination
    """
    from scipy.special import i0, iv
    from scipy.integrate import dblquad

    mean_p2_F = 6 * T**2  # <p^2> for Gamma(2, T)
    coeff = 2 / ((N - 2) * mean_p2_F)

    # PDF: P(p) = (p/T^2)*exp(-p/T) (Radial part of 2D Boltzmann distribution)
    def pdf(p):
        return (p / T**2) * np.exp(-p / T)

    # Integrands for numerator and denominator
    def numerator(p2, p1):
        x = coeff * p1 * p2
        return iv(2, x) * pdf(p1) * pdf(p2)

    def denominator(p2, p1):
        x = coeff * p1 * p2
        return i0(x) * pdf(p1) * pdf(p2)

    # Handle infinity for upper limits
    p1_upper = min(p1_max, 20 * T)  # 20*T covers probability < 1e-9
    p2_upper = min(p2_max, 20 * T)

    opts = {'epsabs': 1e-6, 'epsrel': 1e-6}

    try:
        num_val, _ = dblquad(numerator, p1_min, p1_upper,
                              lambda x: p2_min, lambda x: p2_upper, **opts)
        den_val, _ = dblquad(denominator, p1_min, p1_upper,
                              lambda x: p2_min, lambda x: p2_upper, **opts)

        if den_val > 0:
            return num_val / den_val
        else:
            return np.nan
    except Exception:
        return np.nan


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_binned_analysis(N_range, T: float, pt_bins: list, bin_labels: list,
                        draws: int, warmup: int, chains: int) -> pd.DataFrame:
    """
    Run binned c2{2} analysis for all multiplicities.

    Args:
        N_range: Range of particle multiplicities
        T: Temperature (GeV)
        pt_bins: Bin boundary list
        bin_labels: Bin label list
        draws, warmup, chains: MCMC parameters

    Returns:
        DataFrame with columns: N, bin_i, bin_j, c2_mean, c2_err, n_pairs_mean, etc.
    """
    results = []

    for idx, N in enumerate(N_range):
        progress = f"[{idx+1}/{len(N_range)}]"
        print(f"{progress} Processing N = {N}...", end=" ", flush=True)

        # Run MCMC
        px, py = run_numpyro_tmc(
            N=N, T=T, draws=draws, warmup=warmup, chains=chains,
            target_accept=TARGET_ACCEPT, return_momenta=True, verbose=False
        )

        # Calculate phi and pT
        phi = np.arctan2(py, px)
        pT = np.sqrt(px**2 + py**2)

        # Calculate binned c2
        c2_results = calculate_binned_c2_vectorized(phi, pT, pt_bins, bin_labels)

        # Calculate bin statistics (includes <p^2> for each bin)
        bin_stats = calculate_bin_statistics(pT, pt_bins, bin_labels)

        # Create bin edge lookup
        bin_edges_dict = {
            bin_labels[i]: (pt_bins[i], pt_bins[i+1])
            for i in range(len(bin_labels))
        }

        # Store results
        for (bin_i, bin_j), data in c2_results.items():
            # Get <p^2> for each bin
            mean_p1_sq = bin_stats[bin_i]['mean_p_sq']
            mean_p2_sq = bin_stats[bin_j]['mean_p_sq']

            # Get bin edges
            p1_min, p1_max = bin_edges_dict[bin_i]
            p2_min, p2_max = bin_edges_dict[bin_j]

            # Calculate approximation for this bin combination
            c2_approx = calculate_approximation_c2_binned(mean_p1_sq, mean_p2_sq, N, T)

            # Calculate exact for this bin combination
            c2_exact = calculate_exact_c2_binned(N, T, p1_min, p1_max, p2_min, p2_max)

            results.append({
                'N': N,
                'bin_i': bin_i,
                'bin_j': bin_j,
                'c2_mean': data['c2_mean'],
                'c2_err': data['c2_err'],
                'n_pairs_mean': data['n_pairs_mean'],
                'n_valid_events': data['n_valid_events'],
                'c2_approx': c2_approx,
                'c2_exact': c2_exact,
                # Bin-specific <p^2> values
                'mean_p1_sq': mean_p1_sq,
                'mean_p2_sq': mean_p2_sq,
                # Bin population info
                'bin_i_mean_count': bin_stats[bin_i]['mean_count'],
                'bin_j_mean_count': bin_stats[bin_j]['mean_count'],
            })

        # Print progress
        print("done")

    return pd.DataFrame(results)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_c2_heatmap(df: pd.DataFrame, N: int, output_dir: Path):
    """
    Create 2D heatmap of c2{2} matrix for bin combinations at fixed N.

    Args:
        df: DataFrame with binned results
        N: Multiplicity to plot
        output_dir: Directory to save figure
    """
    # Filter for specific N
    df_n = df[df['N'] == N]

    if df_n.empty:
        print(f"Warning: No data for N={N}")
        return

    n_bins = len(BIN_LABELS)

    # Build matrix
    c2_matrix = np.zeros((n_bins, n_bins))
    err_matrix = np.zeros((n_bins, n_bins))

    for _, row in df_n.iterrows():
        i = BIN_LABELS.index(row['bin_i'])
        j = BIN_LABELS.index(row['bin_j'])
        c2_matrix[i, j] = row['c2_mean']
        err_matrix[i, j] = row['c2_err']

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # Use diverging colormap centered at 0
    vmax = np.nanmax(np.abs(c2_matrix))
    im = ax.imshow(c2_matrix, cmap='RdBu_r', origin='lower',
                   vmin=-vmax, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=r'$c_2\{2\}$')

    # Add text annotations
    for i in range(n_bins):
        for j in range(n_bins):
            val = c2_matrix[i, j]
            err = err_matrix[i, j]
            if np.isnan(val):
                text = 'N/A'
            else:
                text = f'{val:.4f}\n$\\pm${err:.1e}'
            color = 'white' if abs(val) > 0.5 * vmax else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    # Labels
    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    ax.set_xticklabels([f'{b}\n$p_T$' for b in BIN_LABELS])
    ax.set_yticklabels([f'{b}\n$p_T$' for b in BIN_LABELS])
    ax.set_xlabel(r'Particle 2 $p_T$ bin')
    ax.set_ylabel(r'Particle 1 $p_T$ bin')
    ax.set_title(f'$c_2{{2}}(p_{{T1}}, p_{{T2}})$ at N={N}')

    fig.tight_layout()
    fig.savefig(output_dir / f'c2_heatmap_N{N}.png', dpi=300)
    plt.close(fig)
    print(f"  Saved heatmap for N={N}")


def plot_c2_vs_N(df: pd.DataFrame, output_dir: Path):
    """
    Create 1D projections: c2{2}(N) for each bin combination.

    Creates a 3x3 subplot grid showing c2 vs N for all 9 bin combinations,
    comparing simulation with exact theory and approximation formula.
    """
    n_bins = len(BIN_LABELS)

    fig, axes = plt.subplots(n_bins, n_bins, figsize=(12, 10), sharex=True)

    for i, bin_i in enumerate(BIN_LABELS):
        for j, bin_j in enumerate(BIN_LABELS):
            ax = axes[i, j]

            # Filter data
            mask = (df['bin_i'] == bin_i) & (df['bin_j'] == bin_j)
            df_subset = df[mask].sort_values('N')

            N = df_subset['N'].values
            c2 = df_subset['c2_mean'].values
            err = df_subset['c2_err'].values
            c2_approx = df_subset['c2_approx'].values
            c2_exact = df_subset['c2_exact'].values

            # Plot simulation c2
            ax.errorbar(N, c2, yerr=err, fmt='ko', markersize=4, capsize=2,
                        label='Sim')

            # Plot exact theory
            ax.plot(N, c2_exact, 'b-', linewidth=1.5, label='Exact')

            # Plot approximation formula
            ax.plot(N, c2_approx, 'r--', linewidth=1.5, label='Approx')

            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

            # Labels
            ax.set_title(f'{bin_i}-{bin_j}', fontsize=10, fontweight='bold')
            if i == n_bins - 1:
                ax.set_xlabel('N')
            if j == 0:
                ax.set_ylabel(r'$c_2\{2\}$')

            ax.grid(True, alpha=0.3)

            if i == 0 and j == n_bins - 1:
                ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(r'Binned $c_2\{2\}$: Simulation vs Exact vs Approximation',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'c2_vs_N_binned.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved c2 vs N projection plot")


def plot_c2_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Compare all bin combinations: Simulation vs Exact vs Approximation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, 9))

    # Left plot: Simulation results
    ax = axes[0]
    color_idx = 0
    for bin_i in BIN_LABELS:
        for bin_j in BIN_LABELS:
            mask = (df['bin_i'] == bin_i) & (df['bin_j'] == bin_j)
            df_subset = df[mask].sort_values('N')

            N = df_subset['N'].values
            c2 = df_subset['c2_mean'].values
            err = df_subset['c2_err'].values

            label = f'{bin_i}-{bin_j}'
            ax.errorbar(N, c2, yerr=err, fmt='o-', markersize=3, capsize=2,
                        label=label, color=colors[color_idx], alpha=0.8)
            color_idx += 1

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('N (Multiplicity)')
    ax.set_ylabel(r'$c_2\{2\}$')
    ax.set_title('Simulation')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # Middle plot: Exact theory
    ax = axes[1]
    color_idx = 0
    for bin_i in BIN_LABELS:
        for bin_j in BIN_LABELS:
            mask = (df['bin_i'] == bin_i) & (df['bin_j'] == bin_j)
            df_subset = df[mask].sort_values('N')

            N = df_subset['N'].values
            c2_exact = df_subset['c2_exact'].values

            label = f'{bin_i}-{bin_j}'
            ax.plot(N, c2_exact, 'o-', markersize=3,
                    label=label, color=colors[color_idx], alpha=0.8)
            color_idx += 1

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('N (Multiplicity)')
    ax.set_ylabel(r'$c_2\{2\}$ (Exact)')
    ax.set_title(r'Exact: $\int I_2/I_0$ over bin')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # Right plot: Approximation formula
    ax = axes[2]
    color_idx = 0
    for bin_i in BIN_LABELS:
        for bin_j in BIN_LABELS:
            mask = (df['bin_i'] == bin_i) & (df['bin_j'] == bin_j)
            df_subset = df[mask].sort_values('N')

            N = df_subset['N'].values
            c2_approx = df_subset['c2_approx'].values

            label = f'{bin_i}-{bin_j}'
            ax.plot(N, c2_approx, 'o-', markersize=3,
                    label=label, color=colors[color_idx], alpha=0.8)
            color_idx += 1

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('N (Multiplicity)')
    ax.set_ylabel(r'$c_2\{2\}$ (Approx)')
    ax.set_title(r'Approx: $\frac{\langle p_1^2 \rangle \langle p_2^2 \rangle}{2(N-2)^2 \langle p^2 \rangle_F^2}$')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'c2_comparison_all_bins.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved comparison plot")


def plot_pT_distribution(px: np.ndarray, py: np.ndarray,
                          T: float, bin_edges: list, output_dir: Path):
    """
    Plot momentum distribution histogram with bin boundaries.

    Args:
        px, py: Momentum arrays from single simulation
        T: Temperature
        bin_edges: Bin boundaries [0.3, 0.5, 0.8, inf]
        output_dir: Output directory
    """
    pT = np.sqrt(px**2 + py**2).flatten()

    fig, ax = plt.subplots(figsize=(7, 4))

    # Histogram
    counts, bins, patches = ax.hist(pT, bins=100, density=True, alpha=0.6,
                                     color='steelblue', edgecolor='none',
                                     label='Simulation')

    # Color bins differently based on new bin boundaries
    # bin_edges = [0.3, 0.5, 0.8, inf], so we need to handle excluded region too
    bin_colors = ['#cccccc', '#ff6b6b', '#4ecdc4', '#45b7d1']  # gray for excluded
    for patch, b in zip(patches, bins[:-1]):
        if b < bin_edges[0]:  # Excluded region (pT < 0.3)
            patch.set_facecolor(bin_colors[0])
            patch.set_alpha(0.3)
        elif b < bin_edges[1]:  # Low bin
            patch.set_facecolor(bin_colors[1])
            patch.set_alpha(0.6)
        elif b < bin_edges[2]:  # Mid bin
            patch.set_facecolor(bin_colors[2])
            patch.set_alpha(0.6)
        else:  # High bin
            patch.set_facecolor(bin_colors[3])
            patch.set_alpha(0.6)

    # Theory curve
    p_axis = np.linspace(0, 3, 200)
    pdf_theory = (p_axis / T**2) * np.exp(-p_axis / T)
    ax.plot(p_axis, pdf_theory, 'k-', linewidth=2, label=r'Theory $\propto p e^{-p/T}$')

    # Minimum pT cut (excluded region boundary)
    ax.axvline(bin_edges[0], color='red', linestyle='-', linewidth=2)
    ax.text(bin_edges[0] - 0.02, ax.get_ylim()[1] * 0.95,
            f'cut={bin_edges[0]} GeV', rotation=90, va='top', ha='right',
            fontsize=9, color='red')

    # Bin boundaries with labels
    for edge in bin_edges[1:-1]:
        ax.axvline(edge, color='black', linestyle='--', linewidth=2)
        ax.text(edge + 0.02, ax.get_ylim()[1] * 0.95, f'{edge} GeV',
                rotation=90, va='top', fontsize=9)

    # Add region labels
    ax.text(bin_edges[0] / 2, ax.get_ylim()[1] * 0.7, 'EXCL',
            ha='center', fontsize=10, fontweight='bold', color='#888888')
    bin_centers = [(bin_edges[i] + min(bin_edges[i+1], 2.0)) / 2
                   for i in range(len(BIN_LABELS))]
    for center, label, color in zip(bin_centers, BIN_LABELS, bin_colors[1:]):
        ax.text(center, ax.get_ylim()[1] * 0.7, label.upper(),
                ha='center', fontsize=11, fontweight='bold', color=color)

    # Labels
    ax.set_xlabel(r'$p_T$ [GeV]')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Momentum Distribution with Bin Boundaries (T={T} GeV)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 2.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'pT_distribution_binned.png', dpi=300)
    plt.close(fig)
    print("  Saved pT distribution plot")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main entry point for binned c2{2} analysis."""
    print("=" * 70)
    print("Momentum-Bin Based c2{2} Analysis")
    print("=" * 70)
    print(f"Temperature: T = {T_VAL} GeV")
    print(f"N range: {list(N_RANGE)}")
    print(f"Minimum pT cut: {PT_MIN_CUT} GeV (particles below excluded)")
    print(f"pT bins: {PT_BINS} -> labels: {BIN_LABELS}")
    print(f"  - {BIN_LABELS[0]}: {PT_BINS[0]:.1f} - {PT_BINS[1]:.1f} GeV")
    print(f"  - {BIN_LABELS[1]}: {PT_BINS[1]:.1f} - {PT_BINS[2]:.1f} GeV")
    print(f"  - {BIN_LABELS[2]}: {PT_BINS[2]:.1f}+ GeV")
    print(f"Samples per N: {DRAWS} draws x {CHAINS} chains = {DRAWS * CHAINS}")
    print("-" * 70)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Run analysis
    print("\n[Phase 1] Running simulations and calculating binned c2{2}...")
    df = run_binned_analysis(
        N_range=N_RANGE,
        T=T_VAL,
        pt_bins=PT_BINS,
        bin_labels=BIN_LABELS,
        draws=DRAWS,
        warmup=WARMUP,
        chains=CHAINS
    )

    # Add derived columns
    df['sqrt_c2'] = np.sqrt(np.abs(df['c2_mean']))
    df['bin_pair'] = df['bin_i'] + '_' + df['bin_j']

    # Save data
    output_path = OUTPUT_DIR / 'binned_c2_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    # Generate plots
    print("\n[Phase 2] Generating visualizations...")

    # Set plotting style
    try:
        plt.style.use(['science', 'nature'])
    except:
        plt.style.use('default')
        print("  (Using default plot style - scienceplots not available)")

    # Heatmaps for selected N values
    for N in [20, 50, 100]:
        if N in df['N'].values:
            plot_c2_heatmap(df, N, FIG_DIR)

    # 1D projections
    plot_c2_vs_N(df, FIG_DIR)

    # Comparison plot
    plot_c2_comparison(df, FIG_DIR)

    # pT distribution (run one quick simulation for this)
    print("  Generating pT distribution plot...")
    px, py = run_numpyro_tmc(N=50, T=T_VAL, draws=1000, warmup=500,
                             chains=4, return_momenta=True, verbose=False)
    plot_pT_distribution(px, py, T_VAL, PT_BINS, FIG_DIR)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Show sample of results
    print("\nSample c2{2} values at N=50 (Simulation vs Exact vs Approximation):")
    df_50 = df[df['N'] == 50][['bin_i', 'bin_j', 'c2_mean', 'c2_exact', 'c2_approx']]
    print(df_50.to_string(index=False))

    print(f"\nOutput files:")
    print(f"  - Data: {output_path}")
    print(f"  - Figures: {FIG_DIR}/")
    for fig_file in FIG_DIR.glob('*.png'):
        print(f"      {fig_file.name}")

    print("\nDone!")
    return df


if __name__ == "__main__":
    main()
