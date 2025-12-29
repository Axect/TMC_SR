#!/usr/bin/env python3
"""
Generate training data for Symbolic Regression on TMC c2{2} correlation.
Runs NumPyro simulations for multiple T and N values, extracts physics observables
for each momentum bin combination.

Data structure:
    Each row = (T, N, bin_i, bin_j) combination
    p1 quantities = averages over particles in bin_i
    p2 quantities = averages over particles in bin_j
    c2 = two-particle correlation between bin_i and bin_j

Usage:
    python generate_sr_data.py
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path

import jax

# Import simulation functions from numpyro_test
from numpyro_test import run_numpyro_tmc

# Import binned analysis functions
from analyze_momentum_bins import (
    assign_bins, calculate_exact_c2_binned,
    PT_BINS, PT_MIN_CUT, BIN_LABELS
)

# =============================================================================
# Configuration
# =============================================================================

# Temperature values to scan (GeV)
T_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35]

# Multiplicity range
N_RANGE = range(20, 102, 2)  # N=20-100, step=2

# MCMC settings
DRAWS = 6000                    # Samples per chain
WARMUP = 3000                   # Warmup steps
CHAINS = 16                     # Parallel chains
TARGET_ACCEPT = 0.95            # NUTS acceptance rate
SAVE_EVERY = 5                  # Save checkpoint every N runs

# Output
OUTPUT_DIR = Path("data")
OUTPUT_PATH = OUTPUT_DIR / "sr_training_data.csv"


# =============================================================================
# Feature Extraction
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
                # Identifiers (T added)
                'T': T,
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


# =============================================================================
# Main Execution
# =============================================================================

def clear_memory():
    """Clear JAX caches and run garbage collection."""
    jax.clear_caches()
    gc.collect()


def main():
    """Generate SR training data for all T and N combinations."""
    print("=" * 70)
    print("Symbolic Regression Data Generation for TMC c2{2}")
    print("=" * 70)
    print(f"T values: {T_VALUES} GeV")
    print(f"N range: [{min(N_RANGE)}, {max(N_RANGE)}], step {N_RANGE.step}")
    print(f"Minimum pT cut: {PT_MIN_CUT} GeV")
    print(f"pT bins: {PT_BINS} -> {BIN_LABELS}")
    print(f"Samples per (T,N): {DRAWS} draws x {CHAINS} chains = {DRAWS * CHAINS}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Checkpoint every: {SAVE_EVERY} runs")
    print("-" * 70)

    # Checkpoint file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "sr_checkpoint.csv"

    # Check for existing checkpoint (resume capability)
    completed_keys = set()
    results = []
    if checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        df_checkpoint = pd.read_csv(checkpoint_path)
        results = df_checkpoint.to_dict('records')
        # Track completed (T, N) pairs
        for row in results:
            completed_keys.add((row['T'], row['N']))
        print(f"  Loaded {len(results)} existing data points")
        print(f"  Completed (T, N) pairs: {len(completed_keys)}")

    total_runs = len(T_VALUES) * len(N_RANGE)
    run_idx = 0
    runs_since_save = 0

    for T in T_VALUES:
        print(f"\n[T = {T:.2f} GeV]")

        for N in N_RANGE:
            run_idx += 1
            progress = f"[{run_idx}/{total_runs}]"

            # Skip if already completed
            if (T, N) in completed_keys:
                print(f"  {progress} N = {N}... (skipped, already done)")
                continue

            print(f"  {progress} N = {N}...", end=" ", flush=True)

            # Run NumPyro simulation
            px, py = run_numpyro_tmc(
                N=N, T=T, draws=DRAWS, warmup=WARMUP,
                chains=CHAINS, target_accept=TARGET_ACCEPT,
                return_momenta=True, verbose=False
            )

            # Extract binned features
            binned_features = extract_binned_features(px, py, N, T)
            results.extend(binned_features)
            runs_since_save += 1

            n_combos = len(binned_features)
            print(f"{n_combos} bin combos")

            # Clear memory after each run
            del px, py
            clear_memory()

            # Periodic checkpoint save
            if runs_since_save >= SAVE_EVERY:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(checkpoint_path, index=False)
                print(f"    [Checkpoint saved: {len(results)} points]")
                runs_since_save = 0

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)

    # Remove checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("\n[Checkpoint removed after successful completion]")

    print("\n" + "-" * 70)
    print(f"Saved {len(results)} data points to {OUTPUT_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"\nFeature columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")

    print("\nData by T value:")
    for T in T_VALUES:
        mask = df['T'] == T
        print(f"  T={T:.2f}: {mask.sum()} rows")

    print("\nBin combination statistics:")
    for bin_i in BIN_LABELS:
        for bin_j in BIN_LABELS:
            mask = (df['bin_i'] == bin_i) & (df['bin_j'] == bin_j)
            count = mask.sum()
            if count > 0:
                c2_range = df.loc[mask, 'c2_mean']
                print(f"  {bin_i}-{bin_j}: {count} points, "
                      f"c2 range [{c2_range.min():.6f}, {c2_range.max():.6f}]")

    return df


if __name__ == "__main__":
    main()
