#!/usr/bin/env python3
"""
Symbolic Regression for TMC c2{2} using PySR.
Discovers physics formula from binned simulation data.

Data structure:
    Each row = (N, bin_i, bin_j) combination
    p1 quantities = averages over particles in bin_i
    p2 quantities = averages over particles in bin_j

Usage:
    python run_sr.py           # Default data
    python run_sr.py --high    # High-N data (N=50-100)

Requirements:
    pip install pysr pandas
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pysr import PySRRegressor

# Import bin configuration
from analyze_momentum_bins import PT_BINS, BIN_LABELS

# Plotting style
try:
    import scienceplots
    plt.style.use(['science', 'nature'])
except ImportError:
    plt.style.use('default')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("figs")
EQUATIONS_FILE = Path("sr_equations.csv")


# =============================================================================
# PySR Model Configuration
# =============================================================================

def create_pysr_model(niterations: int = 100, timeout: int = 3600) -> PySRRegressor:
    """
    Create PySR model with physics-appropriate configuration.

    Operator Selection Rationale:
    - Binary: +, -, *, / needed for physics formulas
    - Unary: square (p^2), inv (1/N terms) are physics-motivated
    """
    model = PySRRegressor(
        # --- Core search parameters ---
        niterations=niterations,
        populations=16,
        population_size=40,

        # --- Complexity control ---
        maxsize=25,
        maxdepth=6,

        # --- Operators (physics-motivated) ---
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "square",             # x^2 - common in physics
            "inv(x) = 1/x",       # 1/x - for 1/(N-2)^2 terms
        ],

        # --- Operator complexity penalties ---
        complexity_of_operators={
            "+": 1,
            "-": 1,
            "*": 1,
            "/": 2,
            "square": 1,
            "inv": 1,
        },
        complexity_of_constants=3,
        complexity_of_variables=1,

        # --- Constraints to guide search ---
        nested_constraints={
            "inv": {"inv": 0},
            "square": {"square": 0, "inv": 1},
        },

        # --- Output settings ---
        temp_equation_file=str(EQUATIONS_FILE),
        progress=True,
        verbosity=1,

        # --- Numerical stability ---
        parsimony=0.001,
        adaptive_parsimony_scaling=20.0,

        # --- Integer constants only ---
        # Disable constant optimization to keep simple integer coefficients
        should_optimize_constants=False,

        # --- Early stopping ---
        timeout_in_seconds=timeout,

        # --- Extra SymPy mappings ---
        extra_sympy_mappings={
            "inv": lambda x: 1/x,
        },

        # --- Reproducibility ---
        random_state=42,
        deterministic=True,

        # --- Runtime (deterministic requires serial) ---
        parallelism='serial',
    )
    return model


# =============================================================================
# Feature Preparation
# =============================================================================

def prepare_features_binned(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target y from binned data.

    Features are bin-averaged momentum quantities:
        - mean_p1_sq: <p^2> for particles in bin_i
        - mean_p2_sq: <p^2> for particles in bin_j
        - etc.

    Returns:
        X: DataFrame with features (keeps column names for PySR)
        y: Target array (n_samples,)
        weights: Sample weights (n_samples,)
        feature_names: List of feature names
    """
    # Primary features (physics-motivated)
    # Note: 'N' is reserved in SymPy, so rename to 'Npart'
    feature_cols = [
        'N',                      # Particle multiplicity
        'mean_p1_sq',             # <p^2>_bin_i
        'mean_p2_sq',             # <p^2>_bin_j
        'mean_p2_F',              # 6*T^2 (theoretical)
    ]

    # Keep as DataFrame so PySR uses column names in equations
    # Rename 'N' to 'Npart' to avoid SymPy reserved name conflict
    X = df[feature_cols].rename(columns={'N': 'Npart'})
    y = df['c2_mean'].values

    # Weights: inverse variance (higher weight for more precise measurements)
    weights = 1.0 / (df['c2_std'].values**2 + 1e-12)
    weights = weights / np.max(weights)  # Normalize to [0, 1]

    return X, y, weights, feature_cols


def prepare_features_inclusive(df: pd.DataFrame) -> tuple:
    """
    Prepare features from inclusive (pT >= cut) data.
    """
    feature_cols = [
        'N',
        'mean_p_sq',
        'mean_p2_F',
    ]

    # Keep as DataFrame so PySR uses column names in equations
    # Rename 'N' to 'Npart' to avoid SymPy reserved name conflict
    X = df[feature_cols].rename(columns={'N': 'Npart'})
    y = df['c2_mean'].values

    weights = 1.0 / (df['c2_std'].values**2 + 1e-12)
    weights = weights / np.max(weights)

    return X, y, weights, feature_cols


# =============================================================================
# Main SR Execution
# =============================================================================

def run_symbolic_regression_binned(data_path: Path, output_suffix: str = ""):
    """Run symbolic regression on binned data."""
    print("=" * 70)
    print("Symbolic Regression for TMC c2{2} (Binned Data)")
    print("=" * 70)

    # Load data
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please run 'python generate_sr_data.py' first.")
        return None, None

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points from {data_path}")
    print(f"Bin combinations: {df[['bin_i', 'bin_j']].drop_duplicates().values.tolist()}")

    # Prepare features
    X, y, weights, feature_names = prepare_features_binned(df)
    print(f"\nFeatures: {feature_names}")
    print(f"Target: c2_mean (range: [{y.min():.6f}, {y.max():.6f}])")

    # Create and fit model
    model = create_pysr_model()

    print("\n" + "=" * 70)
    print("Starting PySR search...")
    print("=" * 70)

    # Pass DataFrame directly so PySR uses column names in equations
    model.fit(X, y, weights=weights)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: Pareto Front of Equations")
    print("=" * 70)
    print(model)

    # Get best equation
    best_eq = model.get_best()
    print(f"\nBest equation (by score):")
    print(f"  {best_eq['equation']}")
    print(f"  Loss: {best_eq['loss']:.2e}")
    print(f"  Complexity: {best_eq['complexity']}")

    # Save equations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"sr_pareto_front_binned{output_suffix}.csv"
    if hasattr(model, 'equations_') and model.equations_ is not None:
        model.equations_.to_csv(output_file, index=False)
        print(f"\nPareto front saved to {output_file}")

    return model, df, output_suffix


def run_symbolic_regression_inclusive(data_path: Path, output_suffix: str = ""):
    """Run symbolic regression on inclusive data."""
    print("=" * 70)
    print("Symbolic Regression for TMC c2{2} (Inclusive Data)")
    print("=" * 70)

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return None, None, output_suffix

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points from {data_path}")

    X, y, weights, feature_names = prepare_features_inclusive(df)
    print(f"\nFeatures: {feature_names}")
    print(f"Target: c2_mean (range: [{y.min():.6f}, {y.max():.6f}])")

    model = create_pysr_model()

    print("\n" + "=" * 70)
    print("Starting PySR search...")
    print("=" * 70)

    # Pass DataFrame directly so PySR uses column names in equations
    model.fit(X, y, weights=weights)

    print("\n" + "=" * 70)
    print("RESULTS: Pareto Front of Equations")
    print("=" * 70)
    print(model)

    best_eq = model.get_best()
    print(f"\nBest equation (by score):")
    print(f"  {best_eq['equation']}")
    print(f"  Loss: {best_eq['loss']:.2e}")
    print(f"  Complexity: {best_eq['complexity']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"sr_pareto_front_inclusive{output_suffix}.csv"
    if hasattr(model, 'equations_') and model.equations_ is not None:
        model.equations_.to_csv(output_file, index=False)
        print(f"\nPareto front saved to {output_file}")

    return model, df, output_suffix


# =============================================================================
# Visualization
# =============================================================================

def plot_results_binned(model: PySRRegressor, df: pd.DataFrame, output_suffix: str = ""):
    """Create visualization for binned SR results."""
    if model is None or df is None:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, weights, feature_names = prepare_features_binned(df)
    y_pred = model.predict(X)

    # Color map for bin combinations
    bin_combos = [(bi, bj) for bi in BIN_LABELS for bj in BIN_LABELS]
    colors = plt.cm.tab10(np.linspace(0, 1, len(bin_combos)))
    combo_to_color = {combo: colors[i] for i, combo in enumerate(bin_combos)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Left: c2 vs N for each bin combination ---
    ax = axes[0]
    for combo in bin_combos:
        bin_i, bin_j = combo
        mask = (df['bin_i'] == bin_i) & (df['bin_j'] == bin_j)
        if mask.sum() == 0:
            continue

        N_vals = df.loc[mask, 'N'].values
        c2_vals = df.loc[mask, 'c2_mean'].values
        c2_pred = y_pred[mask.values]

        color = combo_to_color[combo]
        label = f'{bin_i}-{bin_j}'

        # Simulation points
        ax.scatter(N_vals, c2_vals, c=[color], s=20, alpha=0.6, marker='o')
        # SR prediction line
        sort_idx = np.argsort(N_vals)
        ax.plot(N_vals[sort_idx], c2_pred[sort_idx], c=color, linewidth=1.5,
                label=label, alpha=0.8)

    ax.set_xlabel(r'$N$ (Multiplicity)')
    ax.set_ylabel(r'$c_2\{2\}$')
    ax.set_title('SR Prediction vs Simulation')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Middle: Parity plot (colored by bin) ---
    ax = axes[1]
    for combo in bin_combos:
        bin_i, bin_j = combo
        mask = (df['bin_i'] == bin_i) & (df['bin_j'] == bin_j)
        if mask.sum() == 0:
            continue

        c2_sim = df.loc[mask, 'c2_mean'].values
        c2_pred = y_pred[mask.values]
        color = combo_to_color[combo]

        ax.scatter(c2_sim, c2_pred, c=[color], s=25, alpha=0.7,
                   label=f'{bin_i}-{bin_j}')

    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect')
    ax.set_xlabel(r'Simulation $c_2\{2\}$')
    ax.set_ylabel(r'SR Prediction')
    ax.set_title('Parity Plot')
    ax.set_aspect('equal')
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

    # --- Right: Residuals vs N ---
    ax = axes[2]
    residuals = y - y_pred
    rel_residuals = residuals / (np.abs(y) + 1e-10) * 100

    for combo in bin_combos:
        bin_i, bin_j = combo
        mask = (df['bin_i'] == bin_i) & (df['bin_j'] == bin_j)
        if mask.sum() == 0:
            continue

        N_vals = df.loc[mask, 'N'].values
        res_vals = rel_residuals[mask.values]
        color = combo_to_color[combo]

        ax.scatter(N_vals, res_vals, c=[color], s=20, alpha=0.6)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(r'$N$ (Multiplicity)')
    ax.set_ylabel(r'Relative Residual (\%)')
    ax.set_title('Residual Analysis')
    ax.grid(True, alpha=0.3)

    # Add best equation as suptitle
    best_eq = model.get_best()
    fig.suptitle(f"Best: {best_eq['equation']}", fontsize=9, y=1.02)

    fig.tight_layout()
    output_path = OUTPUT_DIR / f"sr_results_binned{output_suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


def plot_results_inclusive(model: PySRRegressor, df: pd.DataFrame, output_suffix: str = ""):
    """Create visualization for inclusive SR results."""
    if model is None or df is None:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, weights, feature_names = prepare_features_inclusive(df)
    y_pred = model.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Prediction vs N
    ax = axes[0]
    N = df['N'].values
    ax.errorbar(N, y, yerr=df['c2_std'].values, fmt='ko', markersize=4,
                capsize=2, label='Simulation')
    ax.plot(N, y_pred, 'r-', linewidth=2, label='SR Prediction')
    ax.plot(N, df['c2_exact'].values, 'b--', linewidth=1.5,
            alpha=0.7, label='Exact Theory')
    ax.set_xlabel(r'$N$ (Multiplicity)')
    ax.set_ylabel(r'$c_2\{2\}$')
    ax.set_title('Inclusive: SR vs Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Parity plot
    ax = axes[1]
    ax.scatter(y, y_pred, c='blue', alpha=0.6, s=40)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel(r'Simulation $c_2\{2\}$')
    ax.set_ylabel(r'SR Prediction')
    ax.set_title('Parity Plot')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_eq = model.get_best()
    fig.suptitle(f"Best: {best_eq['equation']}", fontsize=10, y=1.02)

    fig.tight_layout()
    output_path = OUTPUT_DIR / f"sr_results_inclusive{output_suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Symbolic Regression for TMC c2{2}'
    )
    parser.add_argument(
        '--high', action='store_true',
        help='Use high-N data (N=50-100, step=1)'
    )
    return parser.parse_args()


def main():
    """Main entry point - run SR on both binned and inclusive data."""
    args = parse_args()

    # Select data files based on preset
    suffix = "_high" if args.high else ""
    data_path_binned = DATA_DIR / f"sr_training_data_binned{suffix}.csv"
    data_path_inclusive = DATA_DIR / f"sr_training_data_inclusive{suffix}.csv"

    print("\n" + "=" * 70)
    print("TMC c2{2} SYMBOLIC REGRESSION")
    print("=" * 70)
    print(f"Preset: {'high' if args.high else 'default'}")
    print(f"pT bins: {PT_BINS} -> {BIN_LABELS}")
    print(f"Data (binned): {data_path_binned}")
    print(f"Data (inclusive): {data_path_inclusive}")
    print("=" * 70 + "\n")

    # --- Run on binned data ---
    print("\n[1/2] Running SR on BINNED data...")
    print("-" * 70)
    model_binned, df_binned, _ = run_symbolic_regression_binned(data_path_binned, suffix)

    if model_binned is not None:
        plot_results_binned(model_binned, df_binned, suffix)

        print("\n" + "-" * 70)
        print("LaTeX (binned):")
        try:
            print(model_binned.latex())
        except Exception as e:
            print(f"(Not available: {e})")

    # --- Run on inclusive data ---
    print("\n\n[2/2] Running SR on INCLUSIVE data...")
    print("-" * 70)
    model_inclusive, df_inclusive, _ = run_symbolic_regression_inclusive(data_path_inclusive, suffix)

    if model_inclusive is not None:
        plot_results_inclusive(model_inclusive, df_inclusive, suffix)

        print("\n" + "-" * 70)
        print("LaTeX (inclusive):")
        try:
            print(model_inclusive.latex())
        except Exception as e:
            print(f"(Not available: {e})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return model_binned, model_inclusive


if __name__ == "__main__":
    main()
