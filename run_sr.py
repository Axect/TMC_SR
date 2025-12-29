#!/usr/bin/env python3
"""
Symbolic Regression for TMC c2{2} using PySR.
Discovers physics formula from binned simulation data.

Data structure:
    Each row = (T, N, bin_i, bin_j) combination
    p1 quantities = averages over particles in bin_i
    p2 quantities = averages over particles in bin_j

Usage:
    python run_sr.py

Requirements:
    pip install pysr pandas
"""

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
DATA_PATH = DATA_DIR / "sr_training_data.csv"
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

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target y from binned data.

    Features are bin-averaged momentum quantities:
        - Npart: Particle multiplicity (renamed from N to avoid SymPy conflict)
        - mean_p1_sq: <p^2> for particles in bin_i
        - mean_p2_sq: <p^2> for particles in bin_j
        - mean_p2_F: 6*T^2 (theoretical reference, encodes temperature info)

    Note: T is NOT included directly as a feature since mean_p2_F = 6*T^2
          already contains the temperature information.

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


# =============================================================================
# Main SR Execution
# =============================================================================

def run_symbolic_regression(data_path: Path) -> tuple:
    """Run symbolic regression on binned data."""
    print("=" * 70)
    print("Symbolic Regression for TMC c2{2}")
    print("=" * 70)

    # Load data
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please run 'python generate_sr_data.py' first.")
        return None, None

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points from {data_path}")

    # Show data summary
    T_values = df['T'].unique()
    print(f"Temperature values: {sorted(T_values)} GeV")
    print(f"Bin combinations: {df[['bin_i', 'bin_j']].drop_duplicates().values.tolist()}")

    # Prepare features
    X, y, weights, feature_names = prepare_features(df)
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
    output_file = OUTPUT_DIR / "sr_pareto_front.csv"
    if hasattr(model, 'equations_') and model.equations_ is not None:
        model.equations_.to_csv(output_file, index=False)
        print(f"\nPareto front saved to {output_file}")

    return model, df


# =============================================================================
# Visualization
# =============================================================================

def plot_results(model: PySRRegressor, df: pd.DataFrame):
    """Create visualization for SR results."""
    if model is None or df is None:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, weights, feature_names = prepare_features(df)
    y_pred = model.predict(X)

    # Get unique T values for color coding
    T_values = sorted(df['T'].unique())

    # Color map for bin combinations
    bin_combos = [(bi, bj) for bi in BIN_LABELS for bj in BIN_LABELS]
    colors = plt.cm.tab10(np.linspace(0, 1, len(bin_combos)))
    combo_to_color = {combo: colors[i] for i, combo in enumerate(bin_combos)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Left: c2 vs N for each bin combination (all T values) ---
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
    output_path = OUTPUT_DIR / "sr_results.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point - run SR on binned data."""
    print("\n" + "=" * 70)
    print("TMC c2{2} SYMBOLIC REGRESSION")
    print("=" * 70)
    print(f"pT bins: {PT_BINS} -> {BIN_LABELS}")
    print(f"Data: {DATA_PATH}")
    print("=" * 70 + "\n")

    # --- Run symbolic regression ---
    print("Running SR on binned data...")
    print("-" * 70)
    model, df = run_symbolic_regression(DATA_PATH)

    if model is not None:
        plot_results(model, df)

        print("\n" + "-" * 70)
        print("LaTeX equation:")
        try:
            print(model.latex())
        except Exception as e:
            print(f"(Not available: {e})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return model


if __name__ == "__main__":
    main()
