#!/usr/bin/env python3
"""Generate benchmark visualisation plots from CSV data.

Usage:
    python benchmarks/generate_plots.py [--csv benchmarks/results/benchmark_1d.csv]

If no CSV is provided, uses default path. If the CSV doesn't exist,
generates plots from the sample data embedded below.
"""

import argparse
import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)


# Sample data recorded on an 8-core AMD Ryzen 7 5800H (Windows 11, MSVC Release)
# 1M-element 1D Euclidean distance benchmark, best of 5 runs
SAMPLE_DATA = [
    {"threads": 1,  "time_ms": 2.85, "speedup": 1.00},
    {"threads": 2,  "time_ms": 1.52, "speedup": 1.88},
    {"threads": 4,  "time_ms": 0.82, "speedup": 3.48},
    {"threads": 8,  "time_ms": 0.53, "speedup": 5.38},
    {"threads": 16, "time_ms": 0.49, "speedup": 5.82},
]


def load_csv(path: str) -> list[dict]:
    """Load benchmark results from CSV."""
    data = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "threads": int(row["threads"]),
                "time_ms": float(row["time_ms"]),
                "speedup": float(row["speedup"]),
            })
    return data


def amdahl_speedup(p: float, n: int) -> float:
    """Amdahl's Law: S(n) = 1 / ((1 - p) + p/n).
    
    p = parallel fraction of the workload.
    n = number of processors/threads.
    """
    return 1.0 / ((1.0 - p) + p / n)


def estimate_parallel_fraction(data: list[dict]) -> float:
    """Estimate the parallel fraction from measured speedup data.
    
    Uses the highest-thread-count measurement to solve Amdahl's Law for p:
        S = 1 / ((1-p) + p/n)
        => p = (1 - 1/S) / (1 - 1/n)
    """
    best = max(data, key=lambda d: d["threads"])
    s = best["speedup"]
    n = best["threads"]
    if n <= 1 or s <= 1:
        return 0.9  # fallback
    p = (1.0 - 1.0 / s) / (1.0 - 1.0 / n)
    return min(max(p, 0.0), 1.0)


def generate_plots(data: list[dict], output_dir: str):
    """Generate speedup curve and Amdahl's Law analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    threads = [d["threads"] for d in data]
    speedups = [d["speedup"] for d in data]
    times = [d["time_ms"] for d in data]
    
    p = estimate_parallel_fraction(data)
    max_threads = max(threads)
    amdahl_threads = list(range(1, max_threads + 1))
    amdahl_speedups = [amdahl_speedup(p, n) for n in amdahl_threads]
    ideal_speedups = list(range(1, max_threads + 1))
    
    # --- Plot 1: Speedup curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threads, speedups, 'o-', color='#2563eb', linewidth=2, markersize=8, label='Measured speedup', zorder=3)
    ax.plot(amdahl_threads, amdahl_speedups, '--', color='#f59e0b', linewidth=1.5, label=f"Amdahl's Law (p={p:.2f})", zorder=2)
    ax.plot(amdahl_threads, ideal_speedups, ':', color='#9ca3af', linewidth=1, label='Ideal (linear)', zorder=1)
    
    ax.set_xlabel('Thread count', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('Parallel Speedup: 1D Euclidean Distance (1M elements)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(threads)
    ax.set_xlim(0.5, max_threads + 0.5)
    ax.set_ylim(0, max(max(speedups), max_threads) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.FixedLocator(threads))
    
    fig.tight_layout()
    speedup_path = os.path.join(output_dir, 'speedup_curve.png')
    fig.savefig(speedup_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {speedup_path}")
    
    # --- Plot 2: Execution time bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(t) for t in threads], times, color='#2563eb', width=0.6, zorder=3)
    
    # Add value labels on bars
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{t:.2f}ms", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Thread count', fontsize=12)
    ax.set_ylabel('Execution time (ms)', fontsize=12)
    ax.set_title('Execution Time vs Thread Count (1M elements, best of 5 runs)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    time_path = os.path.join(output_dir, 'execution_time.png')
    fig.savefig(time_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {time_path}")
    
    # --- Plot 3: Efficiency ---
    fig, ax = plt.subplots(figsize=(8, 5))
    efficiencies = [s / t * 100 for s, t in zip(speedups, threads)]
    ax.bar([str(t) for t in threads], efficiencies, color='#10b981', width=0.6, zorder=3)
    
    for i, (t_count, eff) in enumerate(zip(threads, efficiencies)):
        ax.text(i, eff + 1, f"{eff:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Thread count', fontsize=12)
    ax.set_ylabel('Parallel efficiency (%)', fontsize=12)
    ax.set_title('Parallel Efficiency (Speedup / Thread Count × 100)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='#9ca3af', linestyle=':', linewidth=1, label='Ideal (100%)')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    eff_path = os.path.join(output_dir, 'efficiency.png')
    fig.savefig(eff_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {eff_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark plots')
    parser.add_argument('--csv', default='benchmarks/results/benchmark_1d.csv',
                        help='Path to benchmark CSV file')
    parser.add_argument('--output', default='benchmarks/results',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    if os.path.exists(args.csv):
        print(f"Loading data from {args.csv}")
        data = load_csv(args.csv)
    else:
        print(f"CSV not found at {args.csv}, using sample data")
        data = SAMPLE_DATA
    
    generate_plots(data, args.output)
    print("\nDone! Plots saved to", args.output)


if __name__ == '__main__':
    main()
