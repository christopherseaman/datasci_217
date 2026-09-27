#!/usr/bin/env python3
"""Demo 2: create arrays, read their properties, convert types, and select parts."""

import numpy as np


def show_creation():
    """Build arrays with the creation functions from the lecture."""
    print("=== Creating Arrays ===")

    # Six patients' body temperatures in degrees Fahrenheit.
    temps_f = np.array([98.6, 101.2, 99.5, 103.1, 97.9, 100.8])
    positions = np.arange(6)
    placeholders = np.zeros(6)

    print(f"Temperatures (°F): {temps_f}")
    print(f"np.arange(6):      {positions}")
    print(f"np.zeros(6):       {placeholders}")
    print()

    return temps_f


def show_properties(temps_f):
    """Report shape, ndim, size, and dtype."""
    print("=== Array Properties ===")
    print(f"shape: {temps_f.shape}")
    print(f"ndim:  {temps_f.ndim}")
    print(f"size:  {temps_f.size}")
    print(f"dtype: {temps_f.dtype}")
    print()


def show_data_types():
    """Convert numeric text to numbers with astype()."""
    print("=== Data Types ===")

    # Temperatures in °F as they arrive from a text file.
    text_temps = np.array(["98.6", "101.2", "99.5"])
    print(f"As text:    {text_temps}  dtype: {text_temps.dtype}")

    numbers = text_temps.astype(float)
    print(f"As floats:  {numbers}  dtype: {numbers.dtype}")
    print(f"As ints:    {numbers.astype(int)}  decimals dropped, not rounded")
    print()


def show_one_dimensional(temps_f):
    """Select single elements and slices from a 1D array."""
    print("=== Indexing and Slicing: 1D ===")
    print(f"temps_f (°F):  {temps_f}")
    print(f"temps_f[0]:    {temps_f[0]}")
    print(f"temps_f[-1]:   {temps_f[-1]}")
    print(f"temps_f[2:5]:  {temps_f[2:5]}")
    print(f"temps_f[::2]:  {temps_f[::2]}")
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

    temps_f = show_creation()
    show_properties(temps_f)
    show_data_types()
    show_one_dimensional(temps_f)
    show_two_dimensional()


if __name__ == "__main__":
    main()
