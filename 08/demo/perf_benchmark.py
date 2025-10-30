#!/usr/bin/env python3
"""
Performance Benchmarks for Lecture 08

Generates clear, high-resolution visuals comparing:
- Multiple groupby calls vs single groupby with multiple aggregations
- apply vs transform for vectorizable operations
- Effect of categorical dtype optimization on speed and memory
- Full-table aggregation vs chunked processing

Outputs (saved next to the lecture media for easy embedding):
- 08/media/perf_groupby_methods.png
- 08/media/perf_memory_optimization.png
- 08/media/perf_chunking.png

Usage:
  uv run python 08/demo/perf_benchmark.py --rows 1_000_000 --groups 5_000
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


def benchmark_chunking(
    df: pd.DataFrame, chunk_size: int = 50_000
) -> List[BenchResult]:
    results: List[BenchResult] = []

    def full():
        return df.groupby("group").agg({
            "value1": "sum",
            "value2": "sum",
            "value3": "sum",
        })

    def chunked():
        parts = []
        for i in range(0, len(df), chunk_size):
            part = df.iloc[i : i + chunk_size]
            parts.append(
                part.groupby("group").agg({
                    "value1": "sum",
                    "value2": "sum",
                    "value3": "sum",
                })
            )
        return pd.concat(parts).groupby(level=0).sum()

    results.append(time_op("Full table aggregation", full))
    results.append(time_op(f"Chunked (size={chunk_size:,})", chunked))
    return results


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


def main():
    parser = argparse.ArgumentParser(
        description="Render performance comparisons for Lecture 08"
    )
    parser.add_argument(
        "--rows", type=int, default=1_000_000, help="Number of rows to generate"
    )
    parser.add_argument(
        "--groups", type=int, default=5_000, help="Number of groups"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=50_000,
        help="Chunk size for chunked benchmark",
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
    plot_bars(
        grp_results,
        "GroupBy Methods: Multiple Calls vs Single agg",
        f"{args.media_dir}/perf_groupby_methods.png",
    )

    # apply vs transform
    print("Benchmarking apply vs transform...")
    at_results = benchmark_apply_vs_transform(df)
    for r in at_results:
        print(f"  {r.label}: {r.seconds:.3f}s")
    plot_bars(
        at_results,
        "apply vs transform for z-score per group",
        f"{args.media_dir}/perf_apply_vs_transform.png",
    )

    # Memory & dtype optimization
    print("Benchmarking memory optimizations...")
    mem_stats = benchmark_memory_and_dtypes(df)
    print(
        f"  Before: {human_bytes(mem_stats['before'])}, After: {human_bytes(mem_stats['after'])}"
    )
    plot_memory(
        mem_stats,
        "Memory Impact of Categorical Dtypes",
        f"{args.media_dir}/perf_memory_optimization.png",
    )

    # Chunked processing
    print("Benchmarking chunked aggregation...")
    chunk_results = benchmark_chunking(df, chunk_size=args.chunk)
    for r in chunk_results:
        print(f"  {r.label}: {r.seconds:.3f}s")
    plot_bars(
        chunk_results,
        f"Chunked vs Full Aggregation (chunk={args.chunk:,})",
        f"{args.media_dir}/perf_chunking.png",
    )

    print("Done. Images written to:")
    print(f"  - {args.media_dir}/perf_groupby_methods.png")
    print(f"  - {args.media_dir}/perf_apply_vs_transform.png")
    print(f"  - {args.media_dir}/perf_memory_optimization.png")
    print(f"  - {args.media_dir}/perf_chunking.png")


if __name__ == "__main__":
    main()
