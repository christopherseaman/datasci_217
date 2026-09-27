#!/usr/bin/env python3
"""
NumPy Performance Demonstration
Applies a heart monitor's calibration offset to one million readings, first with a
list comprehension and then with array arithmetic, and times both.
"""

import numpy as np
import time

OFFSET_BPM = 2  # the wrist monitor reads 2 bpm low


def measure_python_list(heart_rates):
    """Time the list comprehension."""
    print("=== Python List Approach ===")

    start = time.perf_counter()
    calibrated = [bpm + OFFSET_BPM for bpm in heart_rates]
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    print(f"Time: {elapsed_ms:.2f} ms")
    print(f"Result sample: {calibrated[:5]}")

    return elapsed_ms


def measure_numpy_array(heart_rates):
    """Time the array arithmetic."""
    print("\n=== NumPy Array Approach ===")

    start = time.perf_counter()
    calibrated = heart_rates + OFFSET_BPM
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    print(f"Time: {elapsed_ms:.2f} ms")
    print(f"Result sample: {calibrated[:5]}")

    return elapsed_ms


def main():
    """Run performance comparison."""
    print("NumPy Performance Comparison")
    print("=" * 40)

    # Five heart rates repeated 200,000 times: list * n repeats the list.
    heart_rates = [72, 88, 104, 65, 91] * 200_000
    print(f"Heart-rate readings: {len(heart_rates)}")
    print(f"First five (bpm): {heart_rates[:5]}")
    print(f"Operation: add the monitor's {OFFSET_BPM} bpm calibration offset to every reading\n")

    python_time = measure_python_list(heart_rates)
    numpy_time = measure_numpy_array(np.array(heart_rates))

    print("\n" + "=" * 40)
    print(f"Speedup: {python_time / numpy_time:.1f}x faster!")
    print(f"Time saved: {python_time - numpy_time:.2f} ms")
    print("\nTiming is machine-dependent; vectorized arithmetic does the work in array operations.")


if __name__ == "__main__":
    main()
