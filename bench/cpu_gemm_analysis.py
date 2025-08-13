#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(csv_path="results.csv", json_path="results.json"):
    """Load results from CSV and JSON files"""
    csv_data = pd.read_csv(csv_path) if Path(csv_path).exists() else None

    json_data = None
    if Path(json_path).exists():
        with open(json_path, "r") as f:
            json_data = json.load(f)

    return csv_data, json_data


def plot_performance_comparison(csv_data, outfile="build/cpu_gemm_performance.png"):
    """Plot performance comparison of different implementations"""

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Filter data by language and optimization
    cpp_data = csv_data[csv_data["kernel"].str.contains("cpp")]
    mojo_data = csv_data[csv_data["kernel"].str.contains("mojo")]

    # Extract optimization levels
    cpp_opts = []
    cpp_gflops = []
    cpp_times = []

    for _, row in cpp_data.iterrows():
        if "default" in row["kernel"]:
            cpp_opts.append("default")
        elif "o3" in row["kernel"]:
            cpp_opts.append("-O3")
        else:
            cpp_opts.append("-O2")
        cpp_gflops.append(row["throughput_gflops"])
        cpp_times.append(row["time_seconds"])

    mojo_opts = []
    mojo_gflops = []
    mojo_times = []

    for _, row in mojo_data.iterrows():
        if "o3" in row["kernel"]:
            mojo_opts.append("-O3")
        else:
            mojo_opts.append("default")
        mojo_gflops.append(row["throughput_gflops"])
        mojo_times.append(row["time_seconds"])

    # Plot 1: Throughput comparison
    x_pos = np.arange(len(cpp_opts))
    width = 0.35

    ax1.bar(
        x_pos - width / 2, cpp_gflops, width, label="C++", color="skyblue", alpha=0.8
    )
    ax1.bar(
        x_pos + width / 2,
        [
            mojo_gflops[0] if i == 0 else mojo_gflops[1] if len(mojo_gflops) > 1 else 0
            for i in range(len(cpp_opts))
        ],
        width,
        label="Mojo",
        color="lightcoral",
        alpha=0.8,
    )

    ax1.set_xlabel("Optimization Level")
    ax1.set_ylabel("Throughput (GFLOPS)")
    ax1.set_title("GEMM Throughput Comparison (1024×1024×1024)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cpp_opts)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(cpp_gflops):
        ax1.text(
            i - width / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9
        )

    # Plot 2: Execution time comparison
    ax2.bar(
        x_pos - width / 2, cpp_times, width, label="C++", color="skyblue", alpha=0.8
    )
    ax2.bar(
        x_pos + width / 2,
        [
            mojo_times[0] if i == 0 else mojo_times[1] if len(mojo_times) > 1 else 0
            for i in range(len(cpp_opts))
        ],
        width,
        label="Mojo",
        color="lightcoral",
        alpha=0.8,
    )

    ax2.set_xlabel("Optimization Level")
    ax2.set_ylabel("Execution Time (seconds)")
    ax2.set_title("GEMM Execution Time Comparison (1024×1024×1024)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cpp_opts)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(cpp_times):
        ax2.text(
            i - width / 2, v + 0.2, f"{v:.2f}s", ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()

    # Create build directory if it doesn't exist
    Path("build").mkdir(exist_ok=True)

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved performance comparison to {outfile}")

    return fig


def plot_optimization_impact(csv_data, outfile="build/optimization_impact.png"):
    """Plot the impact of optimization flags"""

    plt.figure(figsize=(12, 8))

    # Prepare data
    languages = []
    optimizations = []
    speedups = []
    throughputs = []

    # Get C++ default as baseline
    cpp_default = csv_data[csv_data["kernel"] == "naive_cpp_gemm_default"][
        "throughput_gflops"
    ].iloc[0]

    for _, row in csv_data.iterrows():
        kernel = row["kernel"]
        gflops = row["throughput_gflops"]

        if "cpp" in kernel:
            lang = "C++"
            if "default" in kernel:
                opt = "default"
            elif "o3" in kernel:
                opt = "-O3"
            else:
                opt = "-O2"
        else:
            lang = "Mojo"
            if "o3" in kernel:
                opt = "-O3"
            else:
                opt = "default"

        languages.append(lang)
        optimizations.append(opt)
        throughputs.append(gflops)
        speedups.append(gflops / cpp_default)  # Speedup vs C++ default

    # Create the plot with clean styling
    colors = {"C++": "#4285F4", "Mojo": "#FF6B6B"}
    markers = {"default": "o", "-O2": "s", "-O3": "^"}
    marker_sizes = {"default": 200, "-O2": 180, "-O3": 160}

    # Plot points with GFLOPS in legend
    plotted_labels = set()
    for lang, opt, throughput, speedup in zip(
        languages, optimizations, throughputs, speedups
    ):
        marker = markers[opt]
        size = marker_sizes[opt]
        color = colors[lang]

        # Create label with GFLOPS value
        label = f"{lang} {opt} ({throughput:.3f} GFLOPS)"

        # Only add to legend if not already plotted
        if label not in plotted_labels:
            plt.scatter(
                throughput,
                speedup,
                c=color,
                marker=marker,
                s=size,
                alpha=0.8,
                edgecolors="white",
                linewidth=2,
                label=label,
            )
            plotted_labels.add(label)
        else:
            plt.scatter(
                throughput,
                speedup,
                c=color,
                marker=marker,
                s=size,
                alpha=0.8,
                edgecolors="white",
                linewidth=2,
            )

    plt.xlabel("Throughput (GFLOPS)", fontsize=12, fontweight="bold")
    plt.ylabel("Speedup vs C++ Default", fontsize=12, fontweight="bold")
    plt.title(
        "Optimization Impact on GEMM Performance\n(1024×1024×1024 Matrix Multiplication)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Improve grid
    plt.grid(True, alpha=0.3, linestyle="--")

    # Better legend positioning with GFLOPS values
    legend = plt.legend(
        loc="upper left", fontsize=10, framealpha=0.9, fancybox=True, shadow=True
    )
    legend.get_frame().set_facecolor("white")

    # Add horizontal line at 1.0 (baseline) with better styling
    plt.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="C++ Default Baseline",
    )

    # Set axis limits with some padding
    plt.xlim(0.1, 1.2)
    plt.ylim(0.8, 7.2)

    # Add performance zones with subtle background colors
    plt.axhspan(0.8, 2.0, alpha=0.1, color="red")
    plt.axhspan(2.0, 5.0, alpha=0.1, color="yellow")
    plt.axhspan(5.0, 7.2, alpha=0.1, color="green")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved optimization impact plot to {outfile}")


def print_analysis_summary(csv_data):
    """Print detailed analysis summary"""
    print("\n" + "=" * 60)
    print("NAIVE GEMM PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 60)

    print("\nMatrix Size: 1024×1024×1024")
    print("Total FLOPs: 2,147,483,648 (2.15 billion)")
    print("Algorithm: Naive triple-nested loop (i-j-k order)")

    print("\nPERFORMANCE RESULTS:")
    print("-" * 40)

    for _, row in csv_data.iterrows():
        kernel = row["kernel"]
        time_s = row["time_seconds"]
        gflops = row["throughput_gflops"]

        # Parse kernel name
        if "cpp" in kernel:
            lang = "C++"
            if "default" in kernel:
                opt = "default"
            elif "o3" in kernel:
                opt = "-O3"
            else:
                opt = "-O2"
        else:
            lang = "Mojo"
            if "o3" in kernel:
                opt = "-O3"
            else:
                opt = "default"

        print(f"{lang:4} {opt:8}: {time_s:8.3f}s | {gflops:8.3f} GFLOPS")

    # Calculate speedups
    cpp_default = csv_data[csv_data["kernel"] == "naive_cpp_gemm_default"][
        "throughput_gflops"
    ].iloc[0]
    best_perf = csv_data["throughput_gflops"].max()
    best_kernel = csv_data.loc[csv_data["throughput_gflops"].idxmax(), "kernel"]

    print("\nKEY INSIGHTS:")
    print("-" * 40)
    print(f"• Best Performance: {best_perf:.3f} GFLOPS ({best_kernel})")
    print(
        f"• C++ Optimization Impact: {csv_data[csv_data['kernel'].str.contains('cpp')]['throughput_gflops'].max() / cpp_default:.1f}x speedup"
    )
    print(
        f"• C++ vs Mojo (optimized): {csv_data[csv_data['kernel'].str.contains('cpp')]['throughput_gflops'].max() / csv_data[csv_data['kernel'].str.contains('mojo')]['throughput_gflops'].max():.1f}x faster"
    )
    print(
        f"• Mojo vs C++ (unoptimized): {csv_data[csv_data['kernel'].str.contains('mojo')]['throughput_gflops'].max() / cpp_default:.1f}x faster"
    )


def main():
    """Main analysis function"""
    print("Loading GEMM benchmark results...")

    csv_data, json_data = load_results()

    if csv_data is None:
        print("Error: results.csv not found!")
        return

    print(f"Found {len(csv_data)} benchmark results")

    # Generate plots
    plot_performance_comparison(csv_data)
    plot_optimization_impact(csv_data)

    # Print analysis
    print_analysis_summary(csv_data)

    print("\nAnalysis complete! Check the build/ directory for generated plots.")


if __name__ == "__main__":
    main()
