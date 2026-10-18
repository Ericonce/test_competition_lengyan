"""
signal_length_stat.py
Analyze and visualize signal length distribution of CSV files under a specified directory.
"""

import os
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from data_process.csv_features import resolve_feature_columns
except ModuleNotFoundError:
    from csv_features import resolve_feature_columns


def collect_csv_files(input_dir: str):
    """Recursively collect all CSV file paths under the given directory."""
    return sorted(
        path for path in Path(input_dir).rglob("*.csv") if path.name != "_index.csv"
    )


def count_signal_rows(path: Path) -> int:
    """Count data rows without assuming whether the first row is a header."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as stream:
                rows = [row for row in csv.reader(stream) if any(cell.strip() for cell in row)]
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        if last_error:
            raise last_error
        return 0
    if not rows:
        return 0
    try:
        resolve_feature_columns(rows[0])
    except ValueError:
        return len(rows)
    return max(0, len(rows) - 1)


def get_signal_lengths(csv_files):
    """Read row count (signal length) for each CSV, skipping empty files."""
    lengths = []
    for idx, f in enumerate(tqdm(csv_files)):
        if idx > 1000:
            break
        n = count_signal_rows(f)
        lengths.append((f, n))
    return lengths


def print_stats(lengths):
    """Print signal length statistics to console."""
    if not lengths:
        print("No valid CSV files found.")
        return

    vals = [x[1] for x in lengths]
    print(f"Total CSV files: {len(lengths)}")
    print(f"  Min length:    {min(vals)}")
    print(f"  Max length:    {max(vals)}")
    print(f"  Mean length:   {np.mean(vals):.1f}")
    print(f"  Median length: {np.median(vals):.1f}")
    print(f"  Std dev:       {np.std(vals):.1f}")
    print(f"  Total rows:    {sum(vals)}")

    # Length bin distribution
    bins = [0, 64, 128, 256, 512, 1024, 2048, 4096, float("inf")]
    labels = ["<64", "64-128", "128-256", "256-512", "512-1024", "1024-2048", "2048-4096", ">4096"]
    print("\nLength bin distribution:")
    for lo, hi, lb in zip(bins[:-1], bins[1:], labels):
        cnt = sum(1 for v in vals if lo <= v < hi)
        if cnt > 0:
            print(f"  {lb}: {cnt} ({cnt / len(vals) * 100:.1f}%)")


def plot_distribution(lengths, output_path):
    """Plot histogram (absolute counts) + scaled KDE of signal lengths."""
    vals = np.array([x[1] for x in lengths])

    fig, ax = plt.subplots(figsize=(9, 5))

    # Histogram with absolute counts (removed density=True)
    counts, bin_edges, patches = ax.hist(vals, bins=30, alpha=0.65,
                                          color="#4472C4", edgecolor="white",
                                          label="Count")

    # Scaled KDE to match absolute count scale
    from scipy.stats import gaussian_kde
    if len(vals) > 1:
        kde = gaussian_kde(vals)
        x_range = np.linspace(vals.min(), vals.max(), 200)
        # Scale KDE: pdf(x) * N * bin_width => absolute count per bin
        bin_width = bin_edges[1] - bin_edges[0]
        kde_scaled = kde(x_range) * len(vals) * bin_width
        ax.plot(x_range, kde_scaled, color="#ED7D31", linewidth=2, label="KDE (scaled)")

    # Mark mean
    ax.axvline(np.mean(vals), color="#70AD47", linestyle="--", linewidth=1.5,
               label=f"Mean={vals.mean():.0f}")

    ax.set_xlabel("Signal Length (rows)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("CSV Signal Length Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nDistribution plot saved: {output_path}")


def plot_by_dir(lengths, input_dir, output_path):
    """Box plot of signal lengths grouped by top-level subdirectory."""
    input_path = Path(input_dir)
    dir_data = {}
    for f, n in lengths:
        rel = f.relative_to(input_path)
        top_dir = str(rel.parts[0]) if len(rel.parts) > 1 else "root"
        dir_data.setdefault(top_dir, []).append(n)

    if len(dir_data) < 2:
        print("\nFewer than 2 subdirectories found; skipping per-directory plot.")
        return

    box_data = list(dir_data.values())
    box_labels = list(dir_data.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(box_data, labels=box_labels, patch_artist=True,
               boxprops=dict(facecolor="#4472C4", alpha=0.7))
    ax.set_xlabel("Subdirectory", fontsize=12)
    ax.set_ylabel("Signal Length (rows)", fontsize=12)
    ax.set_title("Signal Length Distribution by Subdirectory", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Per-directory plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize CSV signal length distribution")
    parser.add_argument("--input_dir", "-d",
                        default=r'G:\competition\2026第五届天智杯\赛前准备\雷达型号识别\test_competition_lengyan-main\test_competition_lengyan-main\datasets\synthetic_radar_overlap',
                        help="Directory containing CSV files (recursive search)")
    parser.add_argument("--output_dir", "-o", default="./data_process",
                        help="Output directory for plots (default: ./data_process)")
    args = parser.parse_args()

    csv_files = collect_csv_files(args.input_dir)
    print(f"Found {len(csv_files)} CSV file(s)\n")

    if not csv_files:
        print("No CSV files found. Exiting.")
        return

    lengths = get_signal_lengths(csv_files)
    print_stats(lengths)

    # Visualization
    hist_path = os.path.join(args.output_dir, "signal_length_dist.png")
    box_path = os.path.join(args.output_dir, "signal_length_by_dir.png")
    os.makedirs(args.output_dir, exist_ok=True)

    plot_distribution(lengths, hist_path)
    plot_by_dir(lengths, args.input_dir, box_path)

    print("\nAll done!")


if __name__ == "__main__":
    main()
