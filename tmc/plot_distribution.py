#!/usr/bin/env python3
"""
Plot momentum distribution from Rust TMC sampler output.
Compares sampled distribution against theoretical Gamma(2, T) distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

try:
    import scienceplots
    plt.style.use(['science', 'nature'])
except ImportError:
    plt.style.use('default')


def plot_distribution(parquet_path: str, T: float = 0.25, output: str = None):
    """
    Load parquet file and plot momentum distribution.

    Args:
        parquet_path: Path to parquet file with px, py columns
        T: Temperature parameter (GeV)
        output: Output filename (default: distribution_check.png)
    """
    # Load data
    df = pd.read_parquet(parquet_path)
    px = df['px'].values
    py = df['py'].values

    # Calculate magnitude p_T
    p_mag = np.sqrt(px**2 + py**2)

    print(f"Loaded {len(p_mag)} samples")
    print(f"  <p>  = {np.mean(p_mag):.4f} (theory: {2*T:.4f})")
    print(f"  <p²> = {np.mean(p_mag**2):.4f} (theory: {6*T**2:.4f})")

    # Check momentum conservation
    print(f"  Sum(px) = {np.sum(px):.6e}")
    print(f"  Sum(py) = {np.sum(py):.6e}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: p_T distribution
    ax = axes[0]
    ax.hist(p_mag, bins=100, density=True, alpha=0.5,
            color='black', label='Rust Samples')

    # Theoretical Curve: P(p) = (p/T^2)*exp(-p/T)
    p_axis = np.linspace(0, np.max(p_mag), 200)
    pdf_theory = (p_axis / T**2) * np.exp(-p_axis / T)
    ax.plot(p_axis, pdf_theory, 'r--', linewidth=2,
            label=r'Theory $\propto p e^{-p/T}$')

    ax.set_xlabel(r'$p_T$ [GeV]')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'Momentum Magnitude Distribution')
    ax.legend()
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.2)

    # Right: 2D scatter plot (subsample for visualization)
    ax = axes[1]
    n_plot = min(5000, len(px))
    idx = np.random.choice(len(px), n_plot, replace=False)
    ax.scatter(px[idx], py[idx], s=1, alpha=0.3, c='blue')
    ax.set_xlabel(r'$p_x$ [GeV]')
    ax.set_ylabel(r'$p_y$ [GeV]')
    ax.set_title(r'2D Momentum Distribution')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Add circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [T, 2*T, 3*T]:
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'r--', alpha=0.5, linewidth=1)

    # Save
    if output is None:
        output = "figs/" + Path(parquet_path).stem + "_distribution.png"

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"\nPlot saved as {output}")
    #plt.show()


def plot_c2_2(parquet_path: str = "data/c2_2_results.parquet", output: str = None):
    """
    Plot c2{2} results with error bars and compare with theory.

    Args:
        parquet_path: Path to parquet file with N, c2_2, c2_2_err, c2_2_approx columns
        output: Output filename (default: figs/c2_2_comparison.png)
    """
    df = pd.read_parquet(parquet_path)
    N = df['N'].values
    c2_2 = df['c2_2'].values
    c2_2_err = df['c2_2_err'].values
    c2_2_approx = df['c2_2_approx'].values

    # Convert to sqrt for better visualization
    # Handle negative values by taking abs before sqrt
    sqrt_c2_2 = np.sqrt(np.abs(c2_2)) * np.sign(c2_2)
    sqrt_c2_2_err = c2_2_err / (2 * np.sqrt(np.abs(c2_2) + 1e-10))
    sqrt_c2_2_approx = np.sqrt(c2_2_approx)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: c2{2} with error bars
    ax = axes[0]
    ax.errorbar(N, c2_2, yerr=c2_2_err, fmt='ko', markersize=4, capsize=3,
                label='Simulation')
    ax.plot(N, c2_2_approx, 'r-', linewidth=2, label=r'Theory: $\frac{1}{2(N-2)^2}$')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$N$ (Multiplicity)')
    ax.set_ylabel(r'$c_2\{2\}$')
    ax.set_title(r'TMC-induced $c_2\{2\}$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: sqrt(c2{2}) for flow-like comparison
    ax = axes[1]
    ax.errorbar(N, sqrt_c2_2, yerr=sqrt_c2_2_err, fmt='ko', markersize=4, capsize=3,
                label='Simulation')
    ax.plot(N, sqrt_c2_2_approx, 'r-', linewidth=2, label=r'Theory')
    ax.set_xlabel(r'$N$ (Multiplicity)')
    ax.set_ylabel(r'$\sqrt{c_2\{2\}}$')
    ax.set_title(r'TMC Elliptic Flow')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output is None:
        output = "figs/c2_2_comparison.png"

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Plot saved as {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot TMC simulation results")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Distribution subcommand
    dist_parser = subparsers.add_parser("dist", help="Plot momentum distribution")
    dist_parser.add_argument("parquet", nargs="?", default="data/samples_N20.parquet",
                             help="Path to parquet file (default: data/samples_N20.parquet)")
    dist_parser.add_argument("-T", "--temperature", type=float, default=0.25,
                             help="Temperature parameter in GeV (default: 0.25)")
    dist_parser.add_argument("-o", "--output", type=str, default=None,
                             help="Output filename")

    # c2_2 subcommand
    c2_parser = subparsers.add_parser("c2", help="Plot c2{2} results")
    c2_parser.add_argument("parquet", nargs="?", default="data/c2_2_results.parquet",
                           help="Path to parquet file (default: data/c2_2_results.parquet)")
    c2_parser.add_argument("-o", "--output", type=str, default=None,
                           help="Output filename")

    args = parser.parse_args()

    if args.command == "dist":
        plot_distribution(args.parquet, T=args.temperature, output=args.output)
    elif args.command == "c2":
        plot_c2_2(args.parquet, output=args.output)
    else:
        # Default: run both if data exists
        if Path("data/samples_N20.parquet").exists():
            plot_distribution("data/samples_N20.parquet")
        if Path("data/c2_2_results.parquet").exists():
            plot_c2_2()
