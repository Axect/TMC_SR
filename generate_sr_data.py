#!/usr/bin/env python3
"""
Generate training data for Symbolic Regression on TMC c2{2} correlation.
Runs NumPyro simulations for N in [10, 100] and extracts physics observables
for each momentum bin combination.

Data structure:
    Each row = (N, bin_i, bin_j) combination
    p1 quantities = averages over particles in bin_i
    p2 quantities = averages over particles in bin_j
    c2 = two-particle correlation between bin_i and bin_j

Usage:
    python generate_sr_data.py           # Default: N=20-100, step=2
    python generate_sr_data.py --high    # High-N: N=50-100, step=5
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Import simulation functions from numpyro_test
from numpyro_test import run_numpyro_tmc, calculate_exact_c2_robust

# Import binned analysis functions
from analyze_momentum_bins import (
    assign_bins, calculate_exact_c2_binned,
    PT_BINS, PT_MIN_CUT, BIN_LABELS
)

# =============================================================================
# Configuration
# =============================================================================

T_VAL = 0.25                    # Temperature (GeV)
DRAWS = 5000                    # Samples per chain
WARMUP = 2500                   # Warmup steps
CHAINS = 16                     # Parallel chains
TARGET_ACCEPT = 0.95            # NUTS acceptance rate

OUTPUT_DIR = Path("data")

# Preset configurations
PRESETS = {
    'default': {
        'n_range': range(20, 102, 2),   # N=20-100, step=2
        'suffix': '',
    },
    'high': {
        'n_range': range(50, 101, 1),   # N=50-100, step=1
        'suffix': '_high',
    },
}


# =============================================================================
# Feature Extraction - Binned
# =============================================================================

def extract_binned_features(px: np.ndarray, py: np.ndarray, N: int, T: float) -> list:
    """
    Extract physics-motivated features for each bin combination.

    For each (bin_i, bin_j) pair:
        - p1 quantities: averages over particles in bin_i
        - p2 quantities: averages over particles in bin_j
        - c2{2}: correlation between particles in bin_i and bin_j

    Args:
        px: Array of shape (n_samples, N) - x-component of momenta
        py: Array of shape (n_samples, N) - y-component of momenta
        N: Number of particles
        T: Temperature parameter

    Returns:
        List of dictionaries, one per bin combination
    """
    n_samples = px.shape[0]

    # Calculate pT and phi
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)

    # Assign bins (returns -1 for excluded particles)
    bin_idx = assign_bins(pT, PT_BINS)

    # Pre-compute trig functions
    cos2phi = np.cos(2 * phi)
    sin2phi = np.sin(2 * phi)

    # Theoretical reference
    mean_p2_F = 6 * T**2  # <p^2>_F for Gamma(2, T)

    # Bin edge lookup for exact theory calculation
    bin_edges_dict = {
        BIN_LABELS[k]: (PT_BINS[k], PT_BINS[k+1])
        for k in range(len(BIN_LABELS))
    }

    # Pre-compute bin statistics for each bin
    bin_stats = {}
    for k, label in enumerate(BIN_LABELS):
        mask_k = (bin_idx == k)  # (n_samples, N)
        n_k = mask_k.sum(axis=1)  # particles in this bin per event

        # <p>_bin = average |p| for particles in this bin
        sum_p = np.where(mask_k, pT, 0).sum(axis=1)
        valid_k = n_k > 0
        mean_p_per_event = np.zeros(n_samples)
        mean_p_per_event[valid_k] = sum_p[valid_k] / n_k[valid_k]
        mean_p = np.mean(mean_p_per_event[valid_k]) if valid_k.sum() > 0 else np.nan

        # <p^2>_bin = average p^2 for particles in this bin
        sum_p_sq = np.where(mask_k, pT**2, 0).sum(axis=1)
        mean_p_sq_per_event = np.zeros(n_samples)
        mean_p_sq_per_event[valid_k] = sum_p_sq[valid_k] / n_k[valid_k]
        mean_p_sq = np.mean(mean_p_sq_per_event[valid_k]) if valid_k.sum() > 0 else np.nan

        # <p^3>_bin for higher moments
        sum_p_cu = np.where(mask_k, pT**3, 0).sum(axis=1)
        mean_p_cu_per_event = np.zeros(n_samples)
        mean_p_cu_per_event[valid_k] = sum_p_cu[valid_k] / n_k[valid_k]
        mean_p_cu = np.mean(mean_p_cu_per_event[valid_k]) if valid_k.sum() > 0 else np.nan

        # <p^4>_bin
        sum_p_qu = np.where(mask_k, pT**4, 0).sum(axis=1)
        mean_p_qu_per_event = np.zeros(n_samples)
        mean_p_qu_per_event[valid_k] = sum_p_qu[valid_k] / n_k[valid_k]
        mean_p_qu = np.mean(mean_p_qu_per_event[valid_k]) if valid_k.sum() > 0 else np.nan

        # Mean particle count
        mean_n = np.mean(n_k)

        bin_stats[label] = {
            'mean_p': mean_p,
            'mean_p_sq': mean_p_sq,
            'mean_p_cu': mean_p_cu,
            'mean_p_qu': mean_p_qu,
            'mean_n': mean_n,
            'mask': mask_k,
            'n': n_k,
        }

    results = []

    for i, bin_i in enumerate(BIN_LABELS):
        for j, bin_j in enumerate(BIN_LABELS):
            # Get pre-computed masks and counts
            mask_i = bin_stats[bin_i]['mask']
            mask_j = bin_stats[bin_j]['mask']
            n_i = bin_stats[bin_i]['n']
            n_j = bin_stats[bin_j]['n']

            # === A. Compute c2{2} for this bin combination ===
            # Q-vector components for each bin
            Qc_i = np.where(mask_i, cos2phi, 0).sum(axis=1)
            Qs_i = np.where(mask_i, sin2phi, 0).sum(axis=1)
            Qc_j = np.where(mask_j, cos2phi, 0).sum(axis=1)
            Qs_j = np.where(mask_j, sin2phi, 0).sum(axis=1)

            # Sum of cos(2*delta_phi) over all pairs
            sum_cos = Qc_i * Qc_j + Qs_i * Qs_j

            if i == j:
                # Same bin: subtract self-pairs
                sum_cos = sum_cos - n_i
                n_pairs = n_i * (n_i - 1)
            else:
                # Different bins
                n_pairs = n_i * n_j

            # c2 per event
            valid = n_pairs > 0
            c2_per_event = np.full(n_samples, np.nan)
            c2_per_event[valid] = sum_cos[valid] / n_pairs[valid]

            c2_valid = c2_per_event[~np.isnan(c2_per_event)]

            if len(c2_valid) < 10:
                # Not enough valid events for this bin combination
                continue

            c2_mean = np.mean(c2_valid)
            c2_std = np.std(c2_valid) / np.sqrt(len(c2_valid))

            # === B. Get bin-averaged p1, p2 quantities ===
            # p1 = particles in bin_i, p2 = particles in bin_j
            mean_p1 = bin_stats[bin_i]['mean_p']
            mean_p1_sq = bin_stats[bin_i]['mean_p_sq']
            mean_p1_cu = bin_stats[bin_i]['mean_p_cu']
            mean_p1_qu = bin_stats[bin_i]['mean_p_qu']
            mean_n1 = bin_stats[bin_i]['mean_n']

            mean_p2 = bin_stats[bin_j]['mean_p']
            mean_p2_sq = bin_stats[bin_j]['mean_p_sq']
            mean_p2_cu = bin_stats[bin_j]['mean_p_cu']
            mean_p2_qu = bin_stats[bin_j]['mean_p_qu']
            mean_n2 = bin_stats[bin_j]['mean_n']

            # === C. Derived quantities for SR ===
            # Product of squared momenta (key term in approximation)
            p1_sq_p2_sq = mean_p1_sq * mean_p2_sq

            # Cross products
            mean_p1_p2 = mean_p1 * mean_p2  # <|p1|> * <|p2|>

            # Number of accepted particles (pT >= cut)
            n_accepted = (bin_idx >= 0).sum(axis=1)
            mean_n_accepted = np.mean(n_accepted)

            # === D. Exact theory ===
            p1_min, p1_max = bin_edges_dict[bin_i]
            p2_min, p2_max = bin_edges_dict[bin_j]
            c2_exact = calculate_exact_c2_binned(N, T, p1_min, p1_max, p2_min, p2_max)

            results.append({
                # Identifiers
                'N': N,
                'bin_i': bin_i,
                'bin_j': bin_j,
                'bin_i_idx': i,
                'bin_j_idx': j,

                # Target: c2{2}
                'c2_mean': c2_mean,
                'c2_std': c2_std,
                'c2_exact': c2_exact,

                # Bin_i (p1) averaged quantities
                'mean_p1': mean_p1,           # <|p|>_bin_i
                'mean_p1_sq': mean_p1_sq,     # <p^2>_bin_i
                'mean_p1_cu': mean_p1_cu,     # <p^3>_bin_i
                'mean_p1_qu': mean_p1_qu,     # <p^4>_bin_i
                'mean_n1': mean_n1,           # <n>_bin_i

                # Bin_j (p2) averaged quantities
                'mean_p2': mean_p2,           # <|p|>_bin_j
                'mean_p2_sq': mean_p2_sq,     # <p^2>_bin_j
                'mean_p2_cu': mean_p2_cu,     # <p^3>_bin_j
                'mean_p2_qu': mean_p2_qu,     # <p^4>_bin_j
                'mean_n2': mean_n2,           # <n>_bin_j

                # Cross terms and derived
                'mean_p1_p2': mean_p1_p2,             # <|p1|> * <|p2|>
                'p1_sq_p2_sq': p1_sq_p2_sq,           # <p1^2> * <p2^2>
                'mean_p2_F': mean_p2_F,               # 6*T^2 (theoretical)

                # Global quantities
                'inv_N': 1.0 / N,
                'mean_n_accepted': mean_n_accepted,

                # Statistics
                'n_valid_events': len(c2_valid),
            })

    return results


def extract_inclusive_features(px: np.ndarray, py: np.ndarray, N: int, T: float) -> dict:
    """
    Extract features for inclusive (all particles with pT >= cut) analysis.

    Args:
        px, py: Momentum arrays
        N: Number of particles
        T: Temperature

    Returns:
        Dictionary of features
    """
    n_samples = px.shape[0]

    # Calculate pT and phi
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)

    # Apply pT cut
    mask_accepted = (pT >= PT_MIN_CUT)

    # Pre-compute trig functions
    cos2phi = np.cos(2 * phi)
    sin2phi = np.sin(2 * phi)

    # Number of accepted particles per event
    n_acc = mask_accepted.sum(axis=1)

    # Q-vector for accepted particles
    Qc = np.where(mask_accepted, cos2phi, 0).sum(axis=1)
    Qs = np.where(mask_accepted, sin2phi, 0).sum(axis=1)

    # |Q2|^2 = Qc^2 + Qs^2
    Q2_sq = Qc**2 + Qs**2

    # c2{2} = (|Q2|^2 - n_acc) / (n_acc * (n_acc - 1))
    n_pairs = n_acc * (n_acc - 1)
    valid = n_pairs > 0

    c2_per_event = np.full(n_samples, np.nan)
    c2_per_event[valid] = (Q2_sq[valid] - n_acc[valid]) / n_pairs[valid]

    c2_valid = c2_per_event[~np.isnan(c2_per_event)]
    c2_mean = np.mean(c2_valid)
    c2_std = np.std(c2_valid) / np.sqrt(len(c2_valid))

    # Momentum averages for accepted particles
    # <p>
    sum_p = np.where(mask_accepted, pT, 0).sum(axis=1)
    mean_p_per_event = np.zeros(n_samples)
    mean_p_per_event[valid] = sum_p[valid] / n_acc[valid]
    mean_p = np.mean(mean_p_per_event[valid])

    # <p^2>
    sum_p_sq = np.where(mask_accepted, pT**2, 0).sum(axis=1)
    mean_p_sq_per_event = np.zeros(n_samples)
    mean_p_sq_per_event[valid] = sum_p_sq[valid] / n_acc[valid]
    mean_p_sq = np.mean(mean_p_sq_per_event[valid])

    # Theoretical reference
    mean_p2_F = 6 * T**2

    # Exact theory (with pT cut)
    c2_exact = calculate_exact_c2_binned(N, T, PT_MIN_CUT, np.inf, PT_MIN_CUT, np.inf)

    return {
        'N': N,
        'c2_mean': c2_mean,
        'c2_std': c2_std,
        'c2_exact': c2_exact,
        'mean_p': mean_p,
        'mean_p_sq': mean_p_sq,
        'mean_p2_F': mean_p2_F,
        'inv_N': 1.0 / N,
        'mean_n_accepted': np.mean(n_acc),
        'acceptance_fraction': np.mean(n_acc) / N,
    }


# =============================================================================
# Main Execution
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate SR training data for TMC c2{2}'
    )
    parser.add_argument(
        '--high', action='store_true',
        help='Use high-N preset: N=50-100, step=1'
    )
    return parser.parse_args()


def main():
    """Generate SR training data for all multiplicities and bin combinations."""
    args = parse_args()

    # Select preset
    preset_name = 'high' if args.high else 'default'
    preset = PRESETS[preset_name]
    n_range = preset['n_range']
    suffix = preset['suffix']

    # Output paths
    output_path_binned = OUTPUT_DIR / f"sr_training_data_binned{suffix}.csv"
    output_path_inclusive = OUTPUT_DIR / f"sr_training_data_inclusive{suffix}.csv"

    print("=" * 70)
    print("Symbolic Regression Data Generation for TMC c2{2}")
    print("=" * 70)
    print(f"Preset: {preset_name}")
    print(f"N range: [{min(n_range)}, {max(n_range)}], step {n_range.step}")
    print(f"Temperature: T = {T_VAL} GeV")
    print(f"Minimum pT cut: {PT_MIN_CUT} GeV")
    print(f"pT bins: {PT_BINS} -> {BIN_LABELS}")
    print(f"Samples per N: {DRAWS} draws x {CHAINS} chains = {DRAWS * CHAINS}")
    print(f"Output (binned): {output_path_binned}")
    print(f"Output (inclusive): {output_path_inclusive}")
    print("-" * 70)

    results_binned = []
    results_inclusive = []

    for idx, N in enumerate(n_range):
        progress = f"[{idx+1}/{len(n_range)}]"
        print(f"{progress} Processing N = {N}...", end=" ", flush=True)

        # Run NumPyro simulation
        px, py = run_numpyro_tmc(
            N=N, T=T_VAL, draws=DRAWS, warmup=WARMUP,
            chains=CHAINS, target_accept=TARGET_ACCEPT,
            return_momenta=True, verbose=False
        )

        # Extract binned features
        binned_features = extract_binned_features(px, py, N, T_VAL)
        results_binned.extend(binned_features)

        # Extract inclusive features
        inclusive_features = extract_inclusive_features(px, py, N, T_VAL)
        results_inclusive.append(inclusive_features)

        # Print progress
        c2_inc = inclusive_features['c2_mean']
        n_combos = len(binned_features)
        print(f"c2(incl) = {c2_inc:.6f}, {n_combos} bin combos")

    # Create DataFrames and save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_binned = pd.DataFrame(results_binned)
    df_binned.to_csv(output_path_binned, index=False)

    df_inclusive = pd.DataFrame(results_inclusive)
    df_inclusive.to_csv(output_path_inclusive, index=False)

    print("-" * 70)
    print(f"Saved {len(results_binned)} binned data points to {output_path_binned}")
    print(f"Saved {len(results_inclusive)} inclusive data points to {output_path_inclusive}")

    # Summary
    print("\n" + "=" * 70)
    print("BINNED DATA SUMMARY")
    print("=" * 70)
    print(f"\nFeature columns ({len(df_binned.columns)}):")
    for col in df_binned.columns:
        print(f"  - {col}")

    print("\nBin combination statistics:")
    for bin_i in BIN_LABELS:
        for bin_j in BIN_LABELS:
            mask = (df_binned['bin_i'] == bin_i) & (df_binned['bin_j'] == bin_j)
            count = mask.sum()
            if count > 0:
                c2_range = df_binned.loc[mask, 'c2_mean']
                print(f"  {bin_i}-{bin_j}: {count} points, "
                      f"c2 range [{c2_range.min():.6f}, {c2_range.max():.6f}]")

    print("\n" + "=" * 70)
    print("INCLUSIVE DATA SUMMARY")
    print("=" * 70)
    print(df_inclusive.describe().to_string())

    return df_binned, df_inclusive


if __name__ == "__main__":
    main()
