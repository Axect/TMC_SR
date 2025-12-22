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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot momentum distribution from parquet file")
    parser.add_argument("parquet", nargs="?", default="data/samples.parquet",
                        help="Path to parquet file (default: data/samples.parquet)")
    parser.add_argument("-T", "--temperature", type=float, default=0.25,
                        help="Temperature parameter in GeV (default: 0.25)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output filename (default: <input>_distribution.png)")

    args = parser.parse_args()
    plot_distribution(args.parquet, T=args.temperature, output=args.output)
