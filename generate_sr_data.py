#!/usr/bin/env python3
"""
Generate training data for Symbolic Regression on TMC c2{2} correlation.
Runs NumPyro simulations for N in [10, 100] step 2 and extracts physics observables.

Usage:
    python generate_sr_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import simulation functions from numpyro_test
from numpyro_test import run_numpyro_tmc, calculate_exact_c2_robust

# =============================================================================
# Configuration
# =============================================================================

T_VAL = 0.25                    # Temperature (GeV)
N_RANGE = range(10, 102, 2)     # N from 10 to 100, step 2 (46 points)
DRAWS = 5000                    # Samples per chain
WARMUP = 2500                   # Warmup steps
CHAINS = 16                     # Parallel chains
TARGET_ACCEPT = 0.95            # NUTS acceptance rate

OUTPUT_PATH = Path("data/sr_training_data.csv")


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_features(px: np.ndarray, py: np.ndarray, N: int, T: float) -> dict:
    """
    Extract physics-motivated features from momentum samples.

    Args:
        px: Array of shape (n_samples, N) - x-component of momenta
        py: Array of shape (n_samples, N) - y-component of momenta
        N: Number of particles
        T: Temperature parameter

    Returns:
        Dictionary of features for this multiplicity
    """
    n_samples = px.shape[0]

    # --- A. Core momentum observables (particle 1 and 2 separately) ---
    # <p1^2>_Omega: First particle momentum squared
    p1_sq = px[:, 0]**2 + py[:, 0]**2
    mean_p1_sq = np.mean(p1_sq)

    # <p2^2>_Omega: Second particle momentum squared
    p2_sq = px[:, 1]**2 + py[:, 1]**2
    mean_p2_sq = np.mean(p2_sq)

    # --- B. Cross-correlation terms ---
    # <p1 . p2>_Omega: Dot product between particles 1 and 2
    p1_p2_dot = px[:, 0] * px[:, 1] + py[:, 0] * py[:, 1]
    mean_p1_p2 = np.mean(p1_p2_dot)

    # <|p1| * |p2|>: Product of magnitudes
    p1_mag = np.sqrt(p1_sq)
    p2_mag = np.sqrt(p2_sq)
    mean_p1_mag_p2_mag = np.mean(p1_mag * p2_mag)

    # --- C. Theoretical reference ---
    # <p^2>_F = 6*T^2 for Gamma(2, T) distribution
    mean_p2_F = 6 * T**2

    # --- D. Pre-computed ratios (hints for SR) ---
    inv_N = 1.0 / N
    inv_N_minus_2 = 1.0 / (N - 2) if N > 2 else np.nan

    # Key ratio from approximation formula
    ratio = (mean_p1_sq * mean_p2_sq) / (mean_p2_F**2)

    # --- E. Target: c2{2} from simulation ---
    phi_data = np.arctan2(py, px)
    Q2 = np.sum(np.exp(1j * 2 * phi_data), axis=1)
    c2_samples = (np.abs(Q2)**2 - N) / (N * (N - 1))
    c2_mean = np.mean(c2_samples)
    c2_std = np.std(c2_samples) / np.sqrt(n_samples)

    # --- F. Exact theory for validation ---
    c2_exact = calculate_exact_c2_robust(N, T)

    return {
        'N': N,
        'mean_p1_sq': mean_p1_sq,           # <p1^2>_Omega
        'mean_p2_sq': mean_p2_sq,           # <p2^2>_Omega
        'mean_p1_p2': mean_p1_p2,           # <p1 . p2>_Omega (dot product)
        'mean_p1_mag_p2_mag': mean_p1_mag_p2_mag,  # <|p1| * |p2|>
        'mean_p2_F': mean_p2_F,             # 6*T^2 (theoretical)
        'inv_N': inv_N,                     # 1/N
        'inv_N_minus_2': inv_N_minus_2,     # 1/(N-2)
        'ratio': ratio,                     # (<p1^2> * <p2^2>) / <p^2>_F^2
        'c2_mean': c2_mean,                 # Target: c2{2} simulation mean
        'c2_std': c2_std,                   # Error for weighting
        'c2_exact': c2_exact,               # Exact theory
    }


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Generate SR training data for all multiplicities."""
    print("=" * 70)
    print("Symbolic Regression Data Generation for TMC c2{2}")
    print("=" * 70)
    print(f"N range: [{min(N_RANGE)}, {max(N_RANGE)}], step {N_RANGE.step}")
    print(f"Temperature: T = {T_VAL} GeV")
    print(f"Samples per N: {DRAWS} draws x {CHAINS} chains = {DRAWS * CHAINS}")
    print(f"Output: {OUTPUT_PATH}")
    print("-" * 70)

    results = []

    for i, N in enumerate(N_RANGE):
        progress = f"[{i+1}/{len(N_RANGE)}]"
        print(f"{progress} Processing N = {N}...", end=" ", flush=True)

        # Run NumPyro simulation
        px, py = run_numpyro_tmc(
            N=N, T=T_VAL, draws=DRAWS, warmup=WARMUP,
            chains=CHAINS, target_accept=TARGET_ACCEPT,
            return_momenta=True, verbose=False
        )

        # Extract features
        features = extract_features(px, py, N, T_VAL)
        results.append(features)

        # Print progress
        c2_sim = features['c2_mean']
        c2_exact = features['c2_exact']
        rel_err = abs(c2_sim - c2_exact) / c2_exact * 100
        print(f"c2 = {c2_sim:.6f} (exact: {c2_exact:.6f}, err: {rel_err:.2f}%)")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("-" * 70)
    print(f"Saved {len(results)} data points to {OUTPUT_PATH}")
    print("\nFeature columns:")
    for col in df.columns:
        print(f"  - {col}")

    # Summary statistics
    print("\nData Summary:")
    print(df.describe().to_string())

    return df


if __name__ == "__main__":
    main()
