#!/usr/bin/env python3
"""
Consensus Symbolic Regression: Find robust formulas across N ranges.

Strategy C: Run SR on multiple overlapping N ranges independently, then
identify equation structures that appear consistently across ranges.

Workflow:
    1. Run PySR on predefined overlapping N ranges (--run)
       OR load existing Pareto front CSVs (default)
    2. Extract top equations from each range's Pareto front
    3. Normalize equation structures (replace numeric constants with placeholders)
    4. Find structures that appear across multiple N ranges (consensus)
    5. Refit consensus structures on the full dataset via χ² minimization

Usage:
    python consensus_sr.py                                # Analyze existing Pareto fronts
    python consensus_sr.py --run -w bin_rank_variance      # Run SR first, then analyze
    python consensus_sr.py --run -l relative                # Relative error loss + consensus
    python consensus_sr.py --top-k 10                      # Consider top 10 equations per range
"""

import argparse
import numpy as np
import pandas as pd
import sympy as sp
from pathlib import Path
from collections import defaultdict
from scipy.optimize import curve_fit
from fractions import Fraction

from run_sr import (
    run_symbolic_regression, prepare_features, get_c2_largeN,
    DATA_PATH, OUTPUT_DIR, WEIGHT_SCHEMES, LOSS_TYPES
)
from analyze_momentum_bins import BIN_LABELS

# Plotting style
import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use(['science', 'nature'])
except ImportError:
    plt.style.use('default')


# =============================================================================
# Configuration
# =============================================================================

# Predefined overlapping N ranges for consensus analysis
# N>=10 baseline: large-N approximation is only valid above N~10
N_RANGES = [
    ((10, 30),  "N10to30"),
    ((20, 50),  "N20to50"),
    ((30, 70),  "N30to70"),
    ((50, 100), "N50to100"),
    ((10, 60),  "N10to60"),
    ((10, 100), "N10to100"),
]


# =============================================================================
# Equation Parsing
# =============================================================================

def parse_pysr_equation(eq_str: str):
    """
    Parse PySR equation string into a SymPy expression.

    PySR uses Julia-style function calls (square(), inv()) which need
    to be mapped to SymPy equivalents.

    Args:
        eq_str: Equation string from PySR's Pareto front

    Returns:
        SymPy expression, or None if parsing fails
    """
    try:
        local_dict = {
            'r1': sp.Symbol('r1'),
            'r2': sp.Symbol('r2'),
            'Npart': sp.Symbol('Npart'),
            'square': lambda x: x**2,
            'inv': lambda x: 1 / x,
        }
        expr = sp.sympify(eq_str, locals=local_dict)
        return expr
    except Exception:
        return None


def extract_template(expr):
    """
    Replace numeric constants with placeholder symbols to get structural template.

    E.g., r1*r2/(1.98*(Npart - 2.01)**2) → r1*r2/(C0*(Npart - C1)**2)

    Floats close to small integers (|val - round(val)| < 0.1, |round| <= 3) are
    treated as structural integers and preserved, so that e.g. (Npart - 2.01)
    and (Npart - 2) produce the same template.

    Args:
        expr: SymPy expression

    Returns:
        (template_expr, canonical_repr, original_constants)
        or (None, None, []) if extraction fails
    """
    if expr is None:
        return None, None, []

    def _is_structural(atom):
        """Check if a numeric atom is a structural small integer (keep as-is)."""
        val = float(atom)
        nearest = round(val)
        if abs(val - nearest) < 0.1 and abs(nearest) <= 3:
            return True
        return False

    # Pass 1: Snap near-integer floats to exact integers
    template = expr
    for atom in list(expr.atoms(sp.Float)):
        if _is_structural(atom):
            template = template.subs(atom, sp.Integer(round(float(atom))))

    # Pass 2: Replace remaining non-structural numbers with placeholders
    numbers = []
    for atom in template.atoms(sp.Number):
        if isinstance(atom, sp.Float):
            numbers.append(atom)
        elif isinstance(atom, sp.Integer) and abs(atom) > 3:
            numbers.append(atom)

    numbers = sorted(numbers, key=lambda x: float(x))

    constants = []
    for i, num in enumerate(numbers):
        placeholder = sp.Symbol(f'C{i}')
        template = template.subs(num, placeholder)
        constants.append(float(num))

    # Canonicalize: simplify and get structural representation
    try:
        template = sp.simplify(template)
    except Exception:
        pass

    canonical = sp.srepr(template)
    return template, canonical, constants


# =============================================================================
# Pareto Front Loading
# =============================================================================

def find_pareto_files(weight_scheme: str, loss_type: str = 'mse',
                      log_transform: bool = False) -> dict:
    """
    Find existing Pareto front CSV files for all N ranges.

    Args:
        weight_scheme: Weight scheme string
        loss_type: 'mse', 'relative', or 'log_scatter'
        log_transform: Whether log-log transform was used

    Returns:
        dict mapping range_name -> Path
    """
    found = {}
    for (n_min, n_max), range_name in N_RANGES:
        file_suffix = f"_{range_name}_w{weight_scheme}"
        if log_transform:
            file_suffix += "_log"
        if loss_type != 'mse':
            file_suffix += f"_L{loss_type}"
        path = OUTPUT_DIR / f"sr_pareto_front{file_suffix}.csv"
        if path.exists():
            found[range_name] = path
    return found


def load_pareto_fronts(weight_scheme: str, loss_type: str = 'raw',
                       log_transform: bool = False) -> dict:
    """
    Load all existing Pareto front CSVs for a given configuration.

    Returns:
        dict mapping range_name -> DataFrame
    """
    files = find_pareto_files(weight_scheme, loss_type, log_transform)
    fronts = {}
    for range_name, path in files.items():
        try:
            df = pd.read_csv(path)
            fronts[range_name] = df
            print(f"  Loaded {range_name}: {len(df)} equations from {path}")
        except Exception as e:
            print(f"  Failed to load {range_name}: {e}")
    return fronts


# =============================================================================
# Consensus Analysis
# =============================================================================

def find_consensus(fronts: dict, top_k: int = 5) -> list:
    """
    Find equation structures that appear across multiple N ranges.

    For each N range's Pareto front, extracts the top-K equations (by score),
    normalizes their structure (replacing constants with placeholders), and
    counts how many distinct ranges produce each structure.

    Args:
        fronts: dict mapping range_name -> Pareto front DataFrame
        top_k: Number of top equations to consider per range

    Returns:
        List of consensus results, sorted by number of ranges (descending).
        Each entry: {template, n_ranges, ranges, entries}
    """
    # canonical_repr -> list of (range_name, equation, loss, constants, template)
    structures = defaultdict(list)

    for range_name, df in fronts.items():
        # Get top equations by score (PySR's loss-complexity trade-off metric)
        if 'score' in df.columns:
            top = df.nlargest(top_k, 'score')
        elif 'loss' in df.columns:
            top = df.nsmallest(top_k, 'loss')
        else:
            print(f"  Warning: {range_name} has no 'score' or 'loss' column, skipping")
            continue

        for _, row in top.iterrows():
            eq_str = row.get('equation', '')
            expr = parse_pysr_equation(eq_str)
            if expr is None:
                continue

            template, canonical, constants = extract_template(expr)
            if canonical is None:
                continue

            structures[canonical].append({
                'range': range_name,
                'equation': eq_str,
                'expression': expr,
                'template': template,
                'constants': constants,
                'loss': row.get('loss', float('inf')),
                'complexity': row.get('complexity', 0),
                'score': row.get('score', 0),
            })

    # Build results sorted by number of distinct ranges
    results = []
    for canonical, entries in structures.items():
        ranges = set(e['range'] for e in entries)
        results.append({
            'template': entries[0]['template'],
            'canonical': canonical,
            'n_ranges': len(ranges),
            'ranges': ranges,
            'entries': entries,
            'avg_loss': np.mean([e['loss'] for e in entries]),
        })

    results.sort(key=lambda x: (-x['n_ranges'], x['avg_loss']))
    return results


# =============================================================================
# Consensus Refitting
# =============================================================================

def refit_consensus_template(template_expr, df,
                             weight_scheme='bin_rank_variance',
                             initial_constants=None,
                             n_min=None):
    """
    Refit constants in a consensus template on the full dataset.

    Args:
        template_expr: SymPy expression with placeholder constants (C0, C1, ...)
        df: Full training data DataFrame
        weight_scheme: Weight scheme for prepare_features
        initial_constants: Optional initial guess for constants
        n_min: Minimum N for refitting (excludes Bessel regime where
               the large-N polynomial approximation breaks down)

    Returns:
        dict with fitted expression, parameters, chi2/dof, or None on failure
    """
    if n_min is not None:
        df = df[df['N'] >= n_min].copy()
        if len(df) == 0:
            print(f"    No data with N >= {n_min}")
            return None

    X, y, _, _ = prepare_features(df, weight_scheme)
    sigma = df['c2_std'].values

    # Identify placeholder constants (symbols not in feature set)
    feature_syms = {sp.Symbol('r1'), sp.Symbol('r2'), sp.Symbol('Npart')}
    placeholders = sorted(
        template_expr.free_symbols - feature_syms,
        key=str
    )

    if not placeholders:
        # No constants to fit — evaluate directly
        feat_syms = [sp.Symbol(name) for name in X.columns]
        f_numeric = sp.lambdify(feat_syms, template_expr, modules='numpy')
        y_pred = f_numeric(*[X[col].values for col in X.columns])
        y_pred = np.full_like(y, y_pred) if np.ndim(y_pred) == 0 else np.asarray(y_pred)
        chi2 = np.sum(((y - y_pred) / sigma) ** 2)
        dof = len(y)
        return {
            'template': template_expr,
            'fitted_expr': template_expr,
            'params': {},
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2 / dof,
        }

    # Build callable: f(X_array, C0, C1, ...)
    feat_syms = [sp.Symbol(name) for name in X.columns]
    f_numeric = sp.lambdify(
        feat_syms + placeholders, template_expr, modules='numpy'
    )

    def model_func(X_arr, *params):
        cols = [X_arr[:, i] for i in range(X_arr.shape[1])]
        return f_numeric(*cols, *params)

    # Initial guess
    if initial_constants is not None and len(initial_constants) == len(placeholders):
        p0 = initial_constants
    else:
        p0 = [1.0] * len(placeholders)

    try:
        popt, pcov = curve_fit(
            model_func, X.values, y,
            p0=p0, sigma=sigma, absolute_sigma=True,
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"    Refit failed: {e}")
        return None

    # Goodness of fit
    y_pred = model_func(X.values, *popt)
    y_pred = np.full_like(y, y_pred) if np.ndim(y_pred) == 0 else np.asarray(y_pred)
    chi2 = np.sum(((y - y_pred) / sigma) ** 2)
    dof = len(y) - len(popt)
    chi2_dof = chi2 / dof if dof > 0 else float('inf')

    # Rationalize constants
    expr_rational = template_expr
    param_results = {}
    for p_sym, val, err in zip(placeholders, popt, perr):
        frac = Fraction(val).limit_denominator(20)
        rat = sp.Rational(frac.numerator, frac.denominator)
        expr_rational = expr_rational.subs(p_sym, rat)
        param_results[str(p_sym)] = {
            'value': val, 'error': err, 'rational': str(frac),
        }

    expr_rational = sp.simplify(expr_rational)

    return {
        'template': template_expr,
        'fitted_expr': expr_rational,
        'params': param_results,
        'chi2': chi2,
        'dof': dof,
        'chi2_dof': chi2_dof,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_consensus_results(consensus_results, df,
                           weight_scheme='bin_rank_variance',
                           loss_type='mse',
                           max_formulas=5):
    """
    Plot parity plots for top consensus formulas on the full dataset.

    Args:
        consensus_results: List of (template, refit_result) tuples
        df: Full training data
        weight_scheme: Weight scheme used
        loss_type: Loss type used (for output naming)
        max_formulas: Maximum number of formulas to plot
    """
    n_formulas = min(len(consensus_results), max_formulas)
    if n_formulas == 0:
        print("No consensus formulas to plot.")
        return

    X, y_target, _, _ = prepare_features(df, weight_scheme)

    # Plot in raw c2 space
    y_sim = df['c2_mean'].values

    fig, axes = plt.subplots(1, n_formulas, figsize=(4 * n_formulas, 4))
    if n_formulas == 1:
        axes = [axes]

    for idx, (template, refit) in enumerate(consensus_results[:n_formulas]):
        ax = axes[idx]

        if refit is None:
            ax.set_title("Refit failed", fontsize=8)
            continue

        # Evaluate the fitted formula
        feat_syms = [sp.Symbol(name) for name in X.columns]
        f_eval = sp.lambdify(feat_syms, refit['fitted_expr'], modules='numpy')
        y_pred = f_eval(*[X[col].values for col in X.columns])
        y_pred = np.full_like(y_sim, y_pred) if np.ndim(y_pred) == 0 else np.asarray(y_pred)

        # Color by temperature
        T_values = sorted(df['T'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(T_values)))

        for i_T, T in enumerate(T_values):
            mask = df['T'] == T
            ax.scatter(y_sim[mask], y_pred[mask], c=[colors[i_T]], s=10,
                       alpha=0.5, label=f'T={T:.2f}')

        # Perfect prediction line
        lims = [min(y_sim.min(), y_pred.min()), max(y_sim.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1.5)

        # Display formula
        display_expr = refit['fitted_expr'].subs(
            sp.Symbol('Npart'), sp.Symbol('N')
        )
        chi2_str = f"$\\chi^2$/dof = {refit['chi2_dof']:.2f}"
        ax.set_title(f"${sp.latex(display_expr)}$\n{chi2_str}", fontsize=7)
        ax.set_xlabel(r'Simulation $c_2\{2\}$')
        ax.set_ylabel(r'Consensus prediction')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

    fig.suptitle("Consensus SR: Top Formulas (full dataset refit)", fontsize=10, y=1.02)
    fig.tight_layout()

    loss_suffix = f"_L{loss_type}" if loss_type != 'mse' else ""
    output_path = OUTPUT_DIR / f"sr_consensus_w{weight_scheme}{loss_suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConsensus plot saved to {output_path}")
    plt.close(fig)


# =============================================================================
# Run SR on Multiple Ranges
# =============================================================================

def run_all_ranges(weight_scheme='bin_rank_variance', loss_type='mse',
                   log_transform=False):
    """
    Run PySR on all predefined N ranges.

    This is the expensive step — each range takes O(minutes to hours).

    Args:
        weight_scheme: Weight scheme to use
        loss_type: Loss function type ('mse', 'relative', 'log_scatter')
        log_transform: If True, use log-log transform
    """
    total = len(N_RANGES)
    for idx, ((n_min, n_max), range_name) in enumerate(N_RANGES):
        print(f"\n{'#' * 70}")
        print(f"# Range {idx+1}/{total}: {range_name} ({n_min} <= N <= {n_max})")
        print(f"{'#' * 70}\n")

        model, df, _, _ = run_symbolic_regression(
            DATA_PATH,
            n_range=(n_min, n_max),
            range_name=range_name,
            weight_scheme=weight_scheme,
            loss_type=loss_type,
            log_transform=log_transform,
        )

        if model is not None:
            print(f"  Best equation: {model.get_best()['equation']}")
            print(f"  Loss: {model.get_best()['loss']:.2e}")
        else:
            print(f"  WARNING: SR failed for {range_name}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Consensus Symbolic Regression across N ranges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python consensus_sr.py                              # Analyze existing results
  python consensus_sr.py --run -w bin_rank_variance    # Run SR on all ranges, then analyze
  python consensus_sr.py --run -l relative             # Relative error loss + consensus
  python consensus_sr.py --run -l log_scatter -w none  # Log-scatter loss + uniform weights
        """
    )
    parser.add_argument(
        '--run', action='store_true',
        help='Run SR on all predefined N ranges before analysis (expensive!)'
    )
    parser.add_argument(
        '--weight', '-w', type=str, default='bin_rank_variance',
        choices=WEIGHT_SCHEMES,
        help='Weight scheme (default: bin_rank_variance)'
    )
    parser.add_argument(
        '--loss', '-l', type=str, default='mse',
        choices=LOSS_TYPES,
        help='Loss function: mse (default), relative, or log_scatter'
    )
    parser.add_argument(
        '--top-k', type=int, default=5,
        help='Number of top equations per range to consider (default: 5)'
    )
    parser.add_argument(
        '--refit-n-min', type=int, default=20,
        help='Minimum N for refitting consensus constants (default: 20). '
             'Excludes Bessel regime where large-N approximation fails.'
    )
    parser.add_argument(
        '--log', action='store_true', default=False,
        help='Log-log transform: log(features) + log(c2) target with standard MSE'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_transform = args.log

    print("=" * 70)
    print("CONSENSUS SYMBOLIC REGRESSION (Strategy C)")
    print("=" * 70)
    print(f"Weight scheme: {args.weight}")
    print(f"Loss type:     {args.loss}")
    if log_transform:
        print(f"Log-log:       ENABLED")
    print(f"Top-K:         {args.top_k}")
    print(f"Refit N min:   {args.refit_n_min}")
    print(f"N ranges:      {[name for _, name in N_RANGES]}")
    print("=" * 70)

    # Step 1: Optionally run SR on all ranges
    if args.run:
        print("\n--- PHASE 1: Running SR on all N ranges ---\n")
        run_all_ranges(weight_scheme=args.weight, loss_type=args.loss,
                       log_transform=log_transform)

    # Step 2: Load Pareto fronts
    print("\n--- PHASE 2: Loading Pareto fronts ---\n")
    fronts = load_pareto_fronts(args.weight, args.loss, log_transform)

    if len(fronts) < 2:
        print(f"\nERROR: Need at least 2 Pareto fronts for consensus analysis.")
        print(f"Found {len(fronts)}. Run with --run flag to generate them,")
        print(f"or run 'python run_sr.py' manually for different N ranges.")
        print(f"\nExpected files in {OUTPUT_DIR}/:")
        for (n_min, n_max), range_name in N_RANGES:
            suffix = f"_{range_name}_w{args.weight}"
            if log_transform:
                suffix += "_log"
            if args.loss != 'mse':
                suffix += f"_L{args.loss}"
            print(f"  sr_pareto_front{suffix}.csv")
        return

    # Step 3: Find consensus structures
    print(f"\n--- PHASE 3: Consensus analysis (top-{args.top_k} per range) ---\n")
    consensus = find_consensus(fronts, top_k=args.top_k)

    if not consensus:
        print("No consensus structures found.")
        return

    # Display results
    print("=" * 70)
    print("CONSENSUS STRUCTURES")
    print("=" * 70)

    for i, result in enumerate(consensus[:10]):
        n_ranges = result['n_ranges']
        ranges = sorted(result['ranges'])
        template = result['template']
        avg_loss = result['avg_loss']

        print(f"\n{'─' * 60}")
        print(f"Structure #{i+1}: appears in {n_ranges}/{len(fronts)} ranges")
        print(f"  Template:  {template}")
        print(f"  Ranges:    {', '.join(ranges)}")
        print(f"  Avg loss:  {avg_loss:.2e}")

        # Show individual instances
        for entry in result['entries'][:3]:
            print(f"    [{entry['range']}] {entry['equation']}  "
                  f"(loss={entry['loss']:.2e}, score={entry['score']:.4f})")

    # Step 4: Refit consensus structures on full dataset
    print(f"\n--- PHASE 4: Refitting consensus structures on full dataset ---\n")

    df_full = pd.read_csv(DATA_PATH)
    refit_n_min = args.refit_n_min
    n_refit = (df_full['N'] >= refit_n_min).sum() if refit_n_min else len(df_full)
    print(f"Full dataset: {len(df_full)} data points, "
          f"N = {df_full['N'].min()}-{df_full['N'].max()}")
    if refit_n_min:
        print(f"Refitting on N >= {refit_n_min}: {n_refit} data points")

    refit_results = []
    for i, result in enumerate(consensus[:10]):
        if result['n_ranges'] < 2:
            continue

        template = result['template']
        print(f"\nRefitting structure #{i+1}: {template}")

        # Use average constants from all entries as initial guess
        all_constants = [e['constants'] for e in result['entries']]
        if all_constants and all(len(c) == len(all_constants[0]) for c in all_constants):
            avg_constants = np.mean(all_constants, axis=0).tolist()
        else:
            avg_constants = None

        refit = refit_consensus_template(
            template, df_full,
            weight_scheme=args.weight,
            initial_constants=avg_constants,
            n_min=refit_n_min,
        )

        if refit is not None:
            print(f"  Fitted:    {refit['fitted_expr']}")
            print(f"  chi2/dof:  {refit['chi2_dof']:.3f}")
            for p_name, p_info in refit['params'].items():
                print(f"    {p_name} = {p_info['value']:.6f} +/- {p_info['error']:.6f}"
                      f"  -> {p_info['rational']}")
            refit_results.append((template, refit))

    # Step 5: Plot top consensus formulas
    if refit_results:
        print(f"\n--- PHASE 5: Plotting consensus results ---\n")
        plot_consensus_results(
            refit_results, df_full,
            weight_scheme=args.weight,
            loss_type=args.loss
        )

    # Final summary
    print("\n" + "=" * 70)
    print("CONSENSUS SUMMARY")
    print("=" * 70)

    if refit_results:
        print(f"\nTop consensus formulas (appearing in 2+ ranges, refitted on full data):\n")
        for i, (template, refit) in enumerate(refit_results[:5]):
            display = refit['fitted_expr'].subs(sp.Symbol('Npart'), sp.Symbol('N'))
            print(f"  #{i+1}: c2 = {display}")
            print(f"       chi2/dof = {refit['chi2_dof']:.3f}")
            print(f"       LaTeX: ${sp.latex(display)}$")
    else:
        print("\nNo consensus formulas could be refitted.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
