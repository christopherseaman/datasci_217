#!/usr/bin/env python3
"""Demo 2: create arrays, read their properties, convert types, and select parts."""

import numpy as np


def show_creation():
    """Build arrays with the creation functions from the lecture."""
    print("=== Creating Arrays ===")

    readings = np.array([98.6, 101.2, 99.5, 103.1, 97.9, 100.8])
    positions = np.arange(6)
    placeholders = np.zeros(6)

    print(f"From a list:   {readings}")
    print(f"np.arange(6):  {positions}")
    print(f"np.zeros(6):   {placeholders}")
    print()

    return readings


def show_properties(readings):
    """Report shape, ndim, size, and dtype."""
    print("=== Array Properties ===")
    print(f"shape: {readings.shape}")
    print(f"ndim:  {readings.ndim}")
    print(f"size:  {readings.size}")
    print(f"dtype: {readings.dtype}")
    print()


def show_data_types():
    """Convert numeric text to numbers with astype()."""
    print("=== Data Types ===")

    text_readings = np.array(["98.6", "101.2", "99.5"])
    print(f"As text:    {text_readings}  dtype: {text_readings.dtype}")

    numbers = text_readings.astype(float)
    print(f"As floats:  {numbers}  dtype: {numbers.dtype}")
    print(f"As ints:    {numbers.astype(int)}  decimals dropped, not rounded")
    print()


def show_one_dimensional(readings):
    """Select single elements and slices from a 1D array."""
    print("=== Indexing and Slicing: 1D ===")
    print(f"readings:       {readings}")
    print(f"readings[0]:    {readings[0]}")
    print(f"readings[-1]:   {readings[-1]}")
    print(f"readings[2:5]:  {readings[2:5]}")
    print(f"readings[::2]:  {readings[::2]}")
    print()


def show_two_dimensional():
    """Select cells, rows, columns, and blocks from a 2D array."""
    print("=== Indexing and Slicing: 2D ===")

    bp = np.array([[128, 131, 126],   # patient 0: systolic at visits 1-3
                   [142, 145, 139],   # patient 1
                   [118, 121, 119]])  # patient 2
    print(f"bp shape: {bp.shape}, dtype: {bp.dtype}")
    print(bp)
    print(f"bp[1, 2] (patient 1, visit 3): {bp[1, 2]}")
    print(f"bp[1]    (every visit for patient 1): {bp[1]}")
    print(f"bp[:, 0] (visit 1 for every patient): {bp[:, 0]}")
    print("bp[:2, 1:] (patients 0-1, visits 2-3):")
    print(bp[:2, 1:])
    print()

    print("=== Vectorized Arithmetic ===")
    print("bp - 120 (mmHg above 120), every cell at once:")
    print(bp - 120)


def main():
    """Run the array practice for Demo 2."""
    print("NumPy Arrays: Creation, Properties, and Selection")
    print("=" * 50)
    print()

    readings = show_creation()
    show_properties(readings)
    show_data_types()
    show_one_dimensional(readings)
    show_two_dimensional()


if __name__ == "__main__":
    main()
