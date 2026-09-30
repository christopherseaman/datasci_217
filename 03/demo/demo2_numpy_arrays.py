#!/usr/bin/env python3
"""Demo 2.3: convert types, create arrays, read their properties, calculate, apply ufuncs, and select parts."""

import numpy as np


def show_data_types():
    """Convert numeric text to numbers with astype(), and back to plain Python values."""
    print("=== NumPy Data Types ===")

    # Temperatures in °F as they arrive from a text file.
    text_temps = np.array(["98.6", "101.2", "99.5"])
    print(f"As text:    {text_temps}  dtype: {text_temps.dtype}")

    numbers = text_temps.astype(float)
    print(f"As floats:  {numbers}  dtype: {numbers.dtype}")
    print(f"As ints:    {numbers.astype(int)}  decimals dropped, not rounded")
    print(f"As a list:  {numbers.tolist()}  plain Python floats")
    print()


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


def show_random_arrays():
    """Simulate a week of heart rates with a seeded generator."""
    print("=== Random Arrays ===")

    rng = np.random.default_rng(seed=42)
    # 2 patients x 7 days x 3 readings a day, 60 through 100 bpm.
    week = rng.integers(60, 101, size=(2, 7, 3))
    print(f"week shape: {week.shape}, ndim: {week.ndim}, size: {week.size}")
    print("Patient 0, days 1-3 (one row per day, three readings each):")
    print(week[0, :3])
    print()

    return week


def show_arithmetic(temps_f):
    """Apply one number to every element, then combine two arrays position by position."""
    print("=== Vectorized Arithmetic ===")

    evening_f = np.array([99.1, 100.4, 99.0, 102.0, 98.2, 101.5])
    print(f"Above 98.6 °F:      {temps_f - 98.6}")
    print(f"Evening (°F):       {evening_f}")
    print(f"Evening - morning:  {evening_f - temps_f}")
    print()

    return evening_f


def show_ufuncs(temps_f, evening_f):
    """Apply NumPy's element-by-element functions: to two arrays position by position, then to one."""
    print("=== Universal Functions ===")

    # Each patient's higher temperature of the day, morning or evening.
    print(f"Higher of the two (°F): {np.maximum(temps_f, evening_f)}")

    # Body surface area by the Mosteller formula: the square root of height (cm) x weight (kg) / 3600.
    height_cm = np.array([170, 158, 182])
    weight_kg = np.array([72, 55, 90])
    print(f"Height (cm):            {height_cm}")
    print(f"Weight (kg):            {weight_kg}")
    print(f"Body surface area (m²): {np.sqrt(height_cm * weight_kg / 3600)}")
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
    print("bp - 120 (mmHg above 120), every cell at once:")
    print(bp - 120)
    print()


def show_three_dimensional(week):
    """Select from the patients x days x readings array."""
    print("=== Indexing: 3D ===")
    print(f"week[0, 6]    (patient 0, day 7): {week[0, 6]}")
    print(f"week[1, :, 0] (patient 1, first reading each day): {week[1, :, 0]}")
    print(f"week[:, :, 0].shape (every patient's first reading each day): {week[:, :, 0].shape}")


def main():
    """Run the array practice for Demo 2."""
    print("NumPy Basics: Types, Arrays, and Indexing")
    print("=" * 50)
    print()

    show_data_types()
    temps_f = show_creation()
    show_properties(temps_f)
    week = show_random_arrays()
    evening_f = show_arithmetic(temps_f)
    show_ufuncs(temps_f, evening_f)
    show_one_dimensional(temps_f)
    show_two_dimensional()
    show_three_dimensional(week)


if __name__ == "__main__":
    main()
