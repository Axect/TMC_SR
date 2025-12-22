#!/usr/bin/env python3
"""
Symbolic Regression for TMC c2{2} using PySR.
Discovers physics formula from simulation data.

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

# Plotting style
try:
    import scienceplots
    plt.style.use(['science', 'nature'])
except ImportError:
    plt.style.use('default')

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("data/sr_training_data.csv")
OUTPUT_DIR = Path("figs")
EQUATIONS_FILE = Path("sr_equations.csv")


# =============================================================================
# PySR Model Configuration
# =============================================================================

def create_pysr_model() -> PySRRegressor:
    """
    Create PySR model with physics-appropriate configuration.

    Operator Selection Rationale:
    - Binary: +, -, *, / needed for physics formulas
    - Unary: square (p^2), inv (1/N terms) are physics-motivated
    """
    model = PySRRegressor(
        # --- Core search parameters ---
        niterations=100,          # Iterations (increase for better results)
        populations=16,           # Parallel populations
        population_size=40,       # Individuals per population

        # --- Complexity control ---
        maxsize=25,               # Maximum expression complexity
        maxdepth=6,               # Maximum nesting depth

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
            "/": 2,               # Slightly penalize division
            "square": 2,
            "inv": 2,
        },
        complexity_of_constants=2,
        complexity_of_variables=1,

        # --- Constraints to guide search ---
        nested_constraints={
            "inv": {"inv": 0},    # Prevent inv(inv(x))
            "square": {"square": 0, "inv": 1},
        },

        # --- Output settings ---
        equation_file=str(EQUATIONS_FILE),
        progress=True,
        verbosity=1,

        # --- Numerical stability ---
        parsimony=0.001,
        adaptive_parsimony_scaling=20.0,

        # --- Early stopping ---
        timeout_in_seconds=3600,  # 1 hour max

        # --- Extra SymPy mappings ---
        extra_sympy_mappings={
            "inv": lambda x: 1/x,
        },

        # --- Reproducibility ---
        random_state=42,
        deterministic=True,

        # --- Runtime ---
        procs=0,
        multithreading=True,
    )
    return model


# =============================================================================
# Feature Preparation
# =============================================================================

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target y for SR.

    Returns:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        weights: Sample weights (n_samples,)
        feature_names: List of feature names
    """
    # Primary features (physics-motivated)
    feature_cols = [
        'N',                      # Particle multiplicity
        'mean_p1_sq',             # <p1^2>_Omega
        'mean_p2_sq',             # <p2^2>_Omega
        'mean_p2_F',              # 6*T^2 (theoretical)
        'inv_N_minus_2',          # 1/(N-2) hint
        'ratio',                  # (<p1^2>*<p2^2>)/<p^2>_F^2 hint
    ]

    X = df[feature_cols].values
    y = df['c2_mean'].values

    # Weights: inverse variance (higher weight for more precise measurements)
    weights = 1.0 / (df['c2_std'].values**2 + 1e-12)
    weights = weights / np.max(weights)  # Normalize to [0, 1]

    return X, y, weights, feature_cols


# =============================================================================
# Main SR Execution
# =============================================================================

def run_symbolic_regression():
    """Run symbolic regression and save results."""
    print("=" * 70)
    print("Symbolic Regression for TMC c2{2}")
    print("=" * 70)

    # Load data
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Please run 'python generate_sr_data.py' first.")
        return None, None

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} data points from {DATA_PATH}")

    # Prepare features
    X, y, weights, feature_names = prepare_features(df)
    print(f"\nFeatures: {feature_names}")
    print(f"Target: c2_mean (range: [{y.min():.6f}, {y.max():.6f}])")

    # Create and fit model
    model = create_pysr_model()
    model.feature_names = feature_names

    print("\n" + "=" * 70)
    print("Starting PySR search...")
    print("=" * 70)

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
    if hasattr(model, 'equations_') and model.equations_ is not None:
        model.equations_.to_csv(OUTPUT_DIR / "sr_pareto_front.csv", index=False)
        print(f"\nPareto front saved to {OUTPUT_DIR / 'sr_pareto_front.csv'}")

    return model, df


def plot_results(model: PySRRegressor, df: pd.DataFrame):
    """Create visualization comparing SR prediction vs simulation."""
    if model is None or df is None:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, weights, feature_names = prepare_features(df)
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
    ax.set_title('Symbolic Regression Result')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Parity plot
    ax = axes[1]
    ax.scatter(y, y_pred, c='blue', alpha=0.6, s=40)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel(r'Simulation $c_2\{2\}$')
    ax.set_ylabel(r'SR Prediction $c_2\{2\}$')
    ax.set_title('Parity Plot')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add best equation as text
    best_eq = model.get_best()
    fig.suptitle(f"Best: {best_eq['equation']}", fontsize=10, y=1.02)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "sr_results.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

    plt.close(fig)


def main():
    """Main entry point."""
    model, df = run_symbolic_regression()

    if model is not None:
        plot_results(model, df)

        # Print LaTeX form of best equation
        print("\n" + "=" * 70)
        print("LaTeX representation:")
        print("=" * 70)
        try:
            latex = model.latex()
            print(latex)
        except Exception as e:
            print(f"(LaTeX conversion not available: {e})")

    return model


if __name__ == "__main__":
    main()
