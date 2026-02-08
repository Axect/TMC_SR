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

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pysr import PySRRegressor

# Import bin configuration
from analyze_momentum_bins import PT_BINS, BIN_LABELS

# Post-processing: refit constants via χ² minimization
import sympy as sp
from scipy.optimize import curve_fit
from fractions import Fraction

# Bin rank for weighting (high pT more accurate, low pT less accurate)
BIN_RANK = {'low': 1, 'mid': 2, 'high': 3}

# Weight schemes for SR
WEIGHT_SCHEMES = ['none', 'variance', 'bin_rank', 'bin_rank_variance']

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
        maxsize=20,
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
            "square": 3,
            "inv": 3,
        },
        complexity_of_constants=2,
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

        # --- Constant handling ---
        # Enable optimization so SR can discover shifts like (N-2).
        # Constants are then refitted precisely via χ² post-processing.
        should_optimize_constants=True,

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

def prepare_features(df: pd.DataFrame, weight_scheme: str = 'bin_rank_variance') -> tuple:
    """
    Prepare feature matrix X and target y from binned data.

    Features are non-dimensionalized momentum quantities:
        - r1: <p1²>/<p²>_F — dimensionless momentum ratio for bin_i
        - r2: <p2²>/<p²>_F — dimensionless momentum ratio for bin_j
        - Npart: particle multiplicity (SR must discover the N-2 shift itself)

    Non-dimensionalization ensures SR discovers formulas with integer/rational
    constants. The target large-N formula becomes: c2 = r1·r2 / (2·(Npart-2)²).

    Args:
        df: Input dataframe
        weight_scheme: One of 'none', 'variance', 'bin_rank', 'bin_rank_variance'

    Returns:
        X: DataFrame with features (keeps column names for PySR)
        y: Target array (n_samples,)
        weights: Sample weights (n_samples,)
        feature_names: List of feature names
    """
    # Non-dimensionalized features for clean integer/rational SR constants
    # Dividing by <p²>_F makes momentum features dimensionless and O(1)
    # N is kept as-is: SR must discover the (N-2) shift itself
    X = pd.DataFrame({
        'r1': df['mean_p1_sq'] / df['mean_p2_F'],   # <p1²>/<p²>_F (dimensionless)
        'r2': df['mean_p2_sq'] / df['mean_p2_F'],   # <p2²>/<p²>_F (dimensionless)
        'Npart': df['N'].astype(float),              # multiplicity (unreduced)
    })
    feature_names = list(X.columns)
    y = df['c2_mean'].values

    # Calculate weights based on scheme
    if weight_scheme == 'none':
        weights = np.ones(len(df))
    elif weight_scheme == 'variance':
        # Inverse variance only
        weights = 1.0 / (df['c2_std'].values**2 + 1e-12)
    elif weight_scheme == 'bin_rank':
        # Bin rank only (high-high=9, low-low=1)
        bin_i_rank = df['bin_i'].map(BIN_RANK).values
        bin_j_rank = df['bin_j'].map(BIN_RANK).values
        weights = bin_i_rank * bin_j_rank
    elif weight_scheme == 'bin_rank_variance':
        # Combine bin rank × inverse variance
        bin_i_rank = df['bin_i'].map(BIN_RANK).values
        bin_j_rank = df['bin_j'].map(BIN_RANK).values
        w_bin = bin_i_rank * bin_j_rank
        w_var = 1.0 / (df['c2_std'].values**2 + 1e-12)
        weights = w_bin * w_var
    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    weights = weights / np.max(weights)  # Normalize to [0, 1]

    return X, y, weights, feature_names


# =============================================================================
# Main SR Execution
# =============================================================================

def filter_by_n_range(df: pd.DataFrame, n_range: tuple) -> pd.DataFrame:
    """Filter dataframe by N range.

    Args:
        df: Input dataframe with 'N' column
        n_range: Tuple of (min_N, max_N). None means no limit.

    Returns:
        Filtered dataframe
    """
    n_min, n_max = n_range
    mask = pd.Series(True, index=df.index)
    if n_min is not None:
        mask &= (df['N'] >= n_min)
    if n_max is not None:
        mask &= (df['N'] <= n_max)
    return df[mask].copy()


def run_symbolic_regression(
    data_path: Path,
    n_range: tuple = None,
    range_name: str = None,
    weight_scheme: str = 'bin_rank_variance'
) -> tuple:
    """Run symbolic regression on binned data.

    Args:
        data_path: Path to training data CSV
        n_range: Optional tuple (min_N, max_N) to filter data. None means no limit.
        range_name: Optional name for this N range (for output naming)
        weight_scheme: One of 'none', 'variance', 'bin_rank', 'bin_rank_variance'

    Returns:
        (model, df, range_name, weight_scheme) tuple
    """
    range_suffix = f" [{range_name}]" if range_name else ""
    print("=" * 70)
    print(f"Symbolic Regression for TMC c2{{2}}{range_suffix}")
    print(f"Weight scheme: {weight_scheme}")
    print("=" * 70)

    # Load data
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please run 'python generate_sr_data.py' first.")
        return None, None, None, None

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points from {data_path}")

    # Filter by N range if specified
    if n_range is not None:
        df = filter_by_n_range(df, n_range)
        n_min, n_max = n_range
        range_str = f"{n_min if n_min else ''} <= N <= {n_max if n_max else ''}"
        print(f"Filtered to {len(df)} data points with {range_str}")

    # Show data summary
    T_values = df['T'].unique()
    print(f"Temperature values: {sorted(T_values)} GeV")
    print(f"Bin combinations: {df[['bin_i', 'bin_j']].drop_duplicates().values.tolist()}")

    # Prepare features with specified weight scheme
    X, y, weights, feature_names = prepare_features(df, weight_scheme=weight_scheme)
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

    # Save equations with range and weight scheme in filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_suffix = f"_{range_name}" if range_name else ""
    file_suffix += f"_w{weight_scheme}"
    output_file = OUTPUT_DIR / f"sr_pareto_front{file_suffix}.csv"
    if hasattr(model, 'equations_') and model.equations_ is not None:
        model.equations_.to_csv(output_file, index=False)
        print(f"\nPareto front saved to {output_file}")

    return model, df, range_name, weight_scheme


# =============================================================================
# Constant Refitting (χ² post-processing)
# =============================================================================

def refit_constants(model, df, weight_scheme='bin_rank_variance'):
    """
    Refit constants in PySR's best equation via χ² minimization.

    Two-stage approach:
      1. PySR discovers formula structure (with approximate float constants)
      2. curve_fit refits constants using measurement uncertainties (c2_std)
      3. Fraction identifies nearest simple rationals (denominator ≤ 20)

    Args:
        model: Fitted PySRRegressor
        df: Training data DataFrame
        weight_scheme: Weight scheme used during fitting

    Returns:
        dict with template, fitted expression, parameters, χ²/dof,
        or None if refitting is not applicable / fails.
    """
    X, y, _, _ = prepare_features(df, weight_scheme)
    sigma = df['c2_std'].values

    # Get best equation as SymPy expression
    try:
        sympy_expr = model.sympy()
    except Exception as e:
        print(f"Could not extract SymPy expression: {e}")
        return None

    # Find floating-point constants (these are PySR's optimized values)
    float_atoms = sorted(sympy_expr.atoms(sp.Float), key=lambda x: float(x))

    if not float_atoms:
        print(f"\nNo float constants to refit. Expression: {sympy_expr}")
        return None

    # Replace each constant with a named parameter symbol
    param_symbols = []
    initial_values = []
    expr_template = sympy_expr
    for i, f_val in enumerate(float_atoms):
        p = sp.Symbol(f'c{i}')
        expr_template = expr_template.subs(f_val, p)
        param_symbols.append(p)
        initial_values.append(float(f_val))

    # Build callable: f(X_array, c0, c1, ...)
    feat_syms = [sp.Symbol(name) for name in X.columns]
    f_numeric = sp.lambdify(
        feat_syms + param_symbols, expr_template, modules='numpy'
    )

    def model_func(X_arr, *params):
        cols = [X_arr[:, i] for i in range(X_arr.shape[1])]
        return f_numeric(*cols, *params)

    # χ² minimization via weighted least-squares
    try:
        popt, pcov = curve_fit(
            model_func, X.values, y,
            p0=initial_values,
            sigma=sigma,
            absolute_sigma=True,
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"\nCurve fitting failed: {e}")
        return None

    # Goodness of fit
    y_pred = model_func(X.values, *popt)
    chi2 = np.sum(((y - y_pred) / sigma) ** 2)
    dof = len(y) - len(popt)
    chi2_dof = chi2 / dof if dof > 0 else float('inf')

    # Identify nearest rationals and build clean expression
    print("\n" + "=" * 70)
    print("CONSTANT REFITTING (χ² minimization)")
    print("=" * 70)
    print(f"\nTemplate:  {expr_template}")
    print(f"\nFitted constants:")

    expr_rational = expr_template
    param_results = {}
    for p_sym, val, err in zip(param_symbols, popt, perr):
        frac = Fraction(val).limit_denominator(20)
        rat = sp.Rational(frac.numerator, frac.denominator)
        expr_rational = expr_rational.subs(p_sym, rat)
        param_results[str(p_sym)] = {
            'value': val, 'error': err, 'rational': str(frac),
        }
        print(f"  {p_sym} = {val:.6f} ± {err:.6f}  →  {frac}")

    expr_rational = sp.simplify(expr_rational)

    print(f"\nχ²/dof = {chi2_dof:.3f}  (χ² = {chi2:.1f}, dof = {dof})")
    print(f"\nRational formula:  {expr_rational}")
    print(f"LaTeX:  ${sp.latex(expr_rational)}$")

    return {
        'template': expr_template,
        'fitted_expr': expr_rational,
        'params': param_results,
        'chi2': chi2,
        'dof': dof,
        'chi2_dof': chi2_dof,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_results(
    model: PySRRegressor,
    df: pd.DataFrame,
    range_name: str = None,
    weight_scheme: str = None,
    refit_result: dict = None
):
    """Create visualization for SR results, organized by temperature.

    Layout: n_temps rows × 2 columns
        - Column 1: c2 vs N (simulation points + SR prediction lines)
        - Column 2: Parity plot (simulation vs prediction)

    Args:
        model: Fitted PySR model
        df: Data used for training
        range_name: Optional name for N range (for output naming)
        weight_scheme: Weight scheme used (for output naming)
    """
    if model is None or df is None:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, weights, feature_names = prepare_features(df)

    # Use refitted rational formula for predictions if available
    if refit_result is not None and 'fitted_expr' in refit_result:
        feat_syms = [sp.Symbol(name) for name in X.columns]
        f_rational = sp.lambdify(feat_syms, refit_result['fitted_expr'], modules='numpy')
        y_pred = f_rational(*[X[col].values for col in X.columns])
    else:
        y_pred = model.predict(X)

    df = df.copy()
    df['y_pred'] = y_pred

    # Get unique T values
    T_values = sorted(df['T'].unique())
    n_temps = len(T_values)

    # Color map for bin combinations
    bin_combos = [(bi, bj) for bi in BIN_LABELS for bj in BIN_LABELS]
    colors = plt.cm.tab10(np.linspace(0, 1, len(bin_combos)))
    combo_to_color = {combo: colors[i] for i, combo in enumerate(bin_combos)}

    # Create figure: n_temps rows × 2 columns
    fig, axes = plt.subplots(n_temps, 2, figsize=(8, 3 * n_temps))

    # Handle single temperature case
    if n_temps == 1:
        axes = axes.reshape(1, -1)

    for row, T in enumerate(T_values):
        df_T = df[df['T'] == T]

        # --- Left column: c2 vs N ---
        ax = axes[row, 0]
        for combo in bin_combos:
            bin_i, bin_j = combo
            mask = (df_T['bin_i'] == bin_i) & (df_T['bin_j'] == bin_j)
            if mask.sum() == 0:
                continue

            N_vals = df_T.loc[mask, 'N'].values
            c2_vals = df_T.loc[mask, 'c2_mean'].values
            c2_pred = df_T.loc[mask, 'y_pred'].values

            color = combo_to_color[combo]
            label = f'{bin_i}-{bin_j}' if row == 0 else None

            # Simulation points
            ax.scatter(N_vals, c2_vals, c=[color], s=20, alpha=0.6, marker='o')
            # SR prediction line
            sort_idx = np.argsort(N_vals)
            ax.plot(N_vals[sort_idx], c2_pred[sort_idx], c=color, linewidth=1.5,
                    label=label, alpha=0.8)

        ax.set_xlabel(r'$N$ (Multiplicity)')
        ax.set_ylabel(r'$c_2\{2\}$')
        ax.set_title(f'$T = {T:.2f}$ GeV')
        ax.grid(True, alpha=0.3)

        # Add legend only to first row
        if row == 0:
            ax.legend(fontsize=6, ncol=3, loc='upper right')

        # --- Right column: Parity plot ---
        ax = axes[row, 1]
        for combo in bin_combos:
            bin_i, bin_j = combo
            mask = (df_T['bin_i'] == bin_i) & (df_T['bin_j'] == bin_j)
            if mask.sum() == 0:
                continue

            c2_sim = df_T.loc[mask, 'c2_mean'].values
            c2_pred = df_T.loc[mask, 'y_pred'].values
            color = combo_to_color[combo]

            ax.scatter(c2_sim, c2_pred, c=[color], s=25, alpha=0.7)

        # Perfect prediction line
        y_T = df_T['c2_mean'].values
        y_pred_T = df_T['y_pred'].values
        lims = [min(y_T.min(), y_pred_T.min()), max(y_T.max(), y_pred_T.max())]
        ax.plot(lims, lims, 'k--', linewidth=2)

        ax.set_xlabel(r'Simulation $c_2\{2\}$')
        ax.set_ylabel(r'SR Prediction')
        ax.set_title(f'Parity ($T = {T:.2f}$ GeV)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Add equation as suptitle (prefer refitted rational formula)
    if refit_result is not None and 'fitted_expr' in refit_result:
        # Display rational formula: substitute Npart→N for cleaner display
        display_expr = refit_result['fitted_expr'].subs(
            sp.Symbol('Npart'), sp.Symbol('N')
        )
        rational_latex = sp.latex(display_expr)
        chi2_str = rf"$\chi^2$/dof = {refit_result['chi2_dof']:.2f}"
        fig.suptitle(
            rf"$c_2\{{2\}} = {rational_latex}$  ({chi2_str})",
            fontsize=10, y=1.02
        )
    else:
        best_eq = model.get_best()
        range_label = f" [{range_name}]" if range_name else ""
        weight_label = f" (w={weight_scheme})" if weight_scheme else ""
        fig.suptitle(
            f"Best{range_label}{weight_label}: {best_eq['equation']}",
            fontsize=9, y=1.01
        )

    fig.tight_layout()
    file_suffix = f"_{range_name}" if range_name else ""
    file_suffix += f"_w{weight_scheme}" if weight_scheme else ""
    if refit_result is not None:
        file_suffix += "_refitted"
    output_path = OUTPUT_DIR / f"sr_results{file_suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Symbolic Regression for TMC c2{2}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sr.py --n-min 10 --n-max 50      # Custom range: 10 <= N <= 50
  python run_sr.py --n-max 50                 # N <= 50 (no lower bound)
  python run_sr.py --n-min 50                 # N >= 50 (no upper bound)
  python run_sr.py                            # All data (no N filtering)
  python run_sr.py --n-min 10 --n-max 70 --weight bin_rank  # Custom range + single weight

Available weight schemes:
  none              : No weighting (uniform)
  variance          : 1/c2_std² (inverse variance)
  bin_rank          : bin_i_rank × bin_j_rank (high=3, mid=2, low=1)
  bin_rank_variance : bin_rank × inverse_variance (combined)
  all               : Run all weight schemes (default)
        """
    )
    parser.add_argument(
        '--n-min',
        type=int,
        default=None,
        help='Minimum N value (inclusive). If not specified, no lower bound.'
    )
    parser.add_argument(
        '--n-max',
        type=int,
        default=None,
        help='Maximum N value (inclusive). If not specified, no upper bound.'
    )
    parser.add_argument(
        '--weight', '-w',
        type=str,
        default='all',
        choices=WEIGHT_SCHEMES + ['all'],
        help='Weight scheme to use (default: all)'
    )
    return parser.parse_args()


def main():
    """Main entry point - run SR on binned data."""
    args = parse_args()

    # Build N range from arguments
    n_range = (args.n_min, args.n_max)
    has_n_filter = args.n_min is not None or args.n_max is not None

    # Create range name for output files
    if has_n_filter:
        n_min_str = str(args.n_min) if args.n_min is not None else ""
        n_max_str = str(args.n_max) if args.n_max is not None else ""
        range_name = f"N{n_min_str}to{n_max_str}"
    else:
        range_name = "Nall"

    # Determine which weight schemes to run
    if args.weight == 'all':
        weights_to_run = WEIGHT_SCHEMES
    else:
        weights_to_run = [args.weight]

    # Display N range info
    if args.n_min is not None and args.n_max is not None:
        n_range_str = f"{args.n_min} <= N <= {args.n_max}"
    elif args.n_min is not None:
        n_range_str = f"N >= {args.n_min}"
    elif args.n_max is not None:
        n_range_str = f"N <= {args.n_max}"
    else:
        n_range_str = "All N (no filter)"

    print("\n" + "=" * 70)
    print("TMC c2{2} SYMBOLIC REGRESSION")
    print("=" * 70)
    print(f"pT bins: {PT_BINS} -> {BIN_LABELS}")
    print(f"Data: {DATA_PATH}")
    print(f"N range: {n_range_str}")
    print(f"Weight schemes: {weights_to_run}")
    print("=" * 70 + "\n")

    results = {}

    # --- Run symbolic regression for each weight scheme ---
    total_runs = len(weights_to_run)
    run_count = 0

    for weight_scheme in weights_to_run:
        run_count += 1
        print("\n" + "#" * 70)
        print(f"# Run {run_count}/{total_runs}: {range_name} + weight={weight_scheme}")
        print("#" * 70 + "\n")

        model, df, _, _ = run_symbolic_regression(
            DATA_PATH,
            n_range=n_range if has_n_filter else None,
            range_name=range_name,
            weight_scheme=weight_scheme
        )

        if model is not None:
            # Plot 1: Original PySR equation
            plot_results(model, df, range_name=range_name,
                         weight_scheme=weight_scheme)

            # Stage 2: Refit constants via χ² minimization
            refit_result = refit_constants(model, df, weight_scheme)

            # Plot 2: Refitted rational formula
            if refit_result is not None:
                plot_results(model, df, range_name=range_name,
                             weight_scheme=weight_scheme, refit_result=refit_result)

            print("\n" + "-" * 70)
            print(f"LaTeX equation [{range_name}, w={weight_scheme}]:")
            try:
                print(model.latex())
            except Exception as e:
                print(f"(Not available: {e})")

            results[(range_name, weight_scheme)] = model

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for (range_name, weight_scheme), model in results.items():
        best_eq = model.get_best()
        print(f"\n{range_name} (w={weight_scheme}):")
        print(f"  Equation: {best_eq['equation']}")
        print(f"  Loss: {best_eq['loss']:.2e}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
