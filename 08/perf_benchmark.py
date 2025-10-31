#!/usr/bin/env python3
"""
Performance Benchmarks for Lecture 08

Generates clear, high-resolution visuals comparing:
- Multiple groupby calls vs single groupby with multiple aggregations
- apply vs transform for vectorizable operations
- Effect of categorical dtype optimization on memory usage

Outputs (saved next to the lecture media for easy embedding):
- 08/media/perf_combined.png (combined visualization)

Usage:
  uv run python 08/perf_benchmark.py --rows 100_000_000 --groups 5_000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


@dataclass
class BenchResult:
    label: str
    seconds: float


def human_bytes(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    for u in units:
        if size < 1024.0:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} PB"


def measure_memory_bytes() -> int:
    if psutil is None:
        return 0
    return psutil.Process().memory_info().rss


def generate_data(
    num_rows: int, num_groups: int, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "group": rng.integers(0, num_groups, size=num_rows, dtype=np.int32),
        "value1": rng.standard_normal(num_rows).astype(np.float64),
        "value2": rng.standard_normal(num_rows).astype(np.float64),
        "value3": rng.standard_normal(num_rows).astype(np.float64),
        "category": rng.choice(["A", "B", "C", "D"], size=num_rows),
    })
    return df


def time_op(label: str, func) -> BenchResult:
    t0 = time.perf_counter()
    _ = func()
    t1 = time.perf_counter()
    return BenchResult(label=label, seconds=(t1 - t0))


def benchmark_groupby(df: pd.DataFrame) -> List[BenchResult]:
    results: List[BenchResult] = []

    def multi_calls():
        a = df.groupby("group")["value1"].sum()
        b = df.groupby("group")["value2"].sum()
        c = df.groupby("group")["value3"].sum()
        return a, b, c

    def single_agg():
        return df.groupby("group").agg({
            "value1": "sum",
            "value2": "sum",
            "value3": "sum",
        })

    results.append(time_op("Multiple groupby calls", multi_calls))
    results.append(time_op("Single groupby with agg", single_agg))
    return results


def benchmark_apply_vs_transform(df: pd.DataFrame) -> List[BenchResult]:
    results: List[BenchResult] = []

    def via_apply():
        return df.groupby("group")["value1"].apply(
            lambda s: (s - s.mean()) / s.std(ddof=1)
        )

    def via_transform():
        g = df.groupby("group")["value1"]
        return (df["value1"] - g.transform("mean")) / g.transform("std")

    results.append(time_op("apply (z-score)", via_apply))
    results.append(time_op("transform (z-score)", via_transform))
    return results


def benchmark_memory_and_dtypes(df: pd.DataFrame) -> Dict[str, int]:
    before = df.memory_usage(deep=True).sum()
    df_opt = df.copy()
    df_opt["group"] = pd.Categorical(
        df_opt["group"]
    )  # key optimization for group labels
    df_opt["category"] = pd.Categorical(df_opt["category"])
    after = df_opt.memory_usage(deep=True).sum()
    return {"before": int(before), "after": int(after)}


def plot_bars(pairs: List[BenchResult], title: str, outfile: str):
    labels = [p.label for p in pairs]
    secs = [p.seconds for p in pairs]

    plt.figure(figsize=(12, 7), dpi=150)
    bars = plt.bar(
        labels,
        secs,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(pairs)],
    )
    plt.ylabel("Seconds (lower is better)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    for bar, s in zip(bars, secs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{s:.3f}s",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_memory(before_after: Dict[str, int], title: str, outfile: str):
    labels = ["Before", "After (categorical)"]
    sizes = [before_after["before"], before_after["after"]]
    human = [human_bytes(x) for x in sizes]

    plt.figure(figsize=(12, 7), dpi=150)
    bars = plt.bar(
        labels, [s / (1024**2) for s in sizes], color=["#9467bd", "#8c564b"]
    )
    plt.ylabel("Memory (MB)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    for bar, h in zip(bars, human):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            h,
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_combined(
    grp_results: List[BenchResult],
    at_results: List[BenchResult],
    mem_stats: Dict[str, int],
    outfile: str,
):
    """Create a combined 1x3 subplot visualization of all benchmarks."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.suptitle("Performance Benchmarks (Lower is Better)", fontsize=16, fontweight="bold")

    # Left: GroupBy methods
    ax = axes[0]
    labels = [p.label for p in grp_results]
    secs = [p.seconds for p in grp_results]
    bars = ax.bar(labels, secs, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Seconds")
    ax.set_title("GroupBy: Multiple Calls vs Single agg")
    ax.grid(axis="y", alpha=0.3)
    for bar, s in zip(bars, secs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{s:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.tick_params(axis="x", labelsize=9)

    # Middle: apply vs transform
    ax = axes[1]
    labels = [p.label for p in at_results]
    secs = [p.seconds for p in at_results]
    bars = ax.bar(labels, secs, color=["#2ca02c", "#d62728"])
    ax.set_ylabel("Seconds")
    ax.set_title("apply vs transform (z-score per group)")
    ax.grid(axis="y", alpha=0.3)
    for bar, s in zip(bars, secs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{s:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.tick_params(axis="x", labelsize=9)

    # Right: Memory optimization
    ax = axes[2]
    labels = ["Before", "After (categorical)"]
    sizes = [mem_stats["before"], mem_stats["after"]]
    human = [human_bytes(x) for x in sizes]
    bars = ax.bar(labels, [s / (1024**2) for s in sizes], color=["#9467bd", "#8c564b"])
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Impact of Categorical Dtypes")
    ax.grid(axis="y", alpha=0.3)
    for bar, h in zip(bars, human):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            h,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Render performance comparisons for Lecture 08"
    )
    parser.add_argument(
        "--rows", type=int, default=100_000_000, help="Number of rows to generate"
    )
    parser.add_argument(
        "--groups", type=int, default=5_000, help="Number of groups"
    )
    parser.add_argument(
        "--media-dir",
        type=str,
        default="08/media",
        help="Directory to write images",
    )
    args = parser.parse_args()

    print(f"Generating data: rows={args.rows:,}, groups={args.groups:,}")
    df = generate_data(args.rows, args.groups)

    # GroupBy comparisons
    print("Benchmarking groupby methods...")
    grp_results = benchmark_groupby(df)
    for r in grp_results:
        print(f"  {r.label}: {r.seconds:.3f}s")

    # apply vs transform
    print("Benchmarking apply vs transform...")
    at_results = benchmark_apply_vs_transform(df)
    for r in at_results:
        print(f"  {r.label}: {r.seconds:.3f}s")

    # Memory & dtype optimization
    print("Benchmarking memory optimizations...")
    mem_stats = benchmark_memory_and_dtypes(df)
    print(
        f"  Before: {human_bytes(mem_stats['before'])}, After: {human_bytes(mem_stats['after'])}"
    )

    # Create combined visualization
    print("Creating combined visualization...")
    plot_combined(
        grp_results,
        at_results,
        mem_stats,
        f"{args.media_dir}/perf_combined.png",
    )

    print("Done. Combined image written to:")
    print(f"  - {args.media_dir}/perf_combined.png")


if __name__ == "__main__":
    main()
