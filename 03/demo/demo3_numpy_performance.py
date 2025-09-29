#!/usr/bin/env python3
"""
NumPy Performance Demonstration
Shows why NumPy is essential for data science
"""

import numpy as np
import time

def measure_python_list():
    """Measure Python list performance."""
    print("=== Python List Approach ===")

    # Create data
    data = list(range(1_000_000))

    # Time the operation
    start = time.time()
    result = [x * 2 for x in data]
    end = time.time()

    elapsed_ms = (end - start) * 1000
    print(f"Time: {elapsed_ms:.2f} ms")
    print(f"Result sample: {result[:5]}")

    return elapsed_ms

def measure_numpy_array():
    """Measure NumPy array performance."""
    print("\n=== NumPy Array Approach ===")

    # Create data
    data = np.arange(1_000_000)

    # Time the operation
    start = time.time()
    result = data * 2
    end = time.time()

    elapsed_ms = (end - start) * 1000
    print(f"Time: {elapsed_ms:.2f} ms")
    print(f"Result sample: {result[:5]}")

    return elapsed_ms

def main():
    """Run performance comparison."""
    print("NumPy Performance Comparison")
    print("=" * 40)
    print("Operation: Multiply 1 million numbers by 2\n")

    # Run tests
    python_time = measure_python_list()
    numpy_time = measure_numpy_array()

    # Show comparison
    print("\n" + "=" * 40)
    print(f"Speedup: {python_time / numpy_time:.1f}x faster!")
    print(f"Time saved: {python_time - numpy_time:.2f} ms")
    print("\n✅ NumPy is 10-100x faster for numerical operations")

if __name__ == "__main__":
    main()