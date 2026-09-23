#!/usr/bin/env python3
"""
Blood Pressure Analysis with NumPy
Demonstrates practical NumPy operations
"""

import numpy as np

MMHG_TO_KPA = 0.133

def create_sample_data():
    """Create realistic diastolic blood pressure data."""
    rng = np.random.default_rng(42)  # Reproducible results

    n_patients = 100
    n_visits = 5

    # Generate diastolic readings in mmHg (70-100 range)
    readings = rng.integers(70, 101, size=(n_patients, n_visits))

    print(f"Created data: {n_patients} patients, {n_visits} visits")
    print(f"Array shape: {readings.shape}")
    print(f"Data type: {readings.dtype}\n")

    return readings

def demo_basic_operations(readings):
    """Demonstrate basic NumPy operations."""
    print("=== Basic Operations ===")

    # Arithmetic operations
    print(f"First patient's readings: {readings[0]}")
    print(f"In kPa (x0.133): {readings[0] * MMHG_TO_KPA}")
    print(f"Calibrated (+3 mmHg): {readings[0] + 3}")
    print()

def demo_views_and_copies(readings):
    """Show that a slice shares its data and .copy() does not."""
    print("=== Views vs Copies ===")

    # Work on an independent block so the rest of the demo sees the original readings.
    practice = readings[:2, :3].copy()
    print("Practice block (2 patients, 3 visits):")
    print(practice)

    view = practice[0]           # a view: another window onto practice
    independent = practice[0].copy()

    view[0] = 0                  # writing through the view reaches practice
    independent[1] = 0           # writing to the copy stays local

    print(f"After view[0] = 0, practice row 0: {practice[0]}")
    print(f"After independent[1] = 0, the copy: {independent}")
    print(f"View shares memory with practice: {np.shares_memory(view, practice)}")
    print(f"Copy shares memory with practice: {np.shares_memory(independent, practice)}")
    print(f"Original readings row 0, untouched: {readings[0]}")
    print()


def calibrate_in_place(values):
    """First draft: += changes the caller's array instead of making a new one."""
    values += 3


def calibrate(values):
    """Return the readings plus the cuff's 3 mmHg correction as a new array."""
    return values + 3


def demo_function_changes_input(readings):
    """Show a function changing the caller's array, then the version that returns a new one."""
    print("=== Functions and the Caller's Array ===")

    # Work on copies of patient 0's row so the rest of the demo sees the original readings.
    row = readings[0].copy()
    print("calibrate_in_place(row) on a copy of patient 0's readings:")
    print(f"  row before: {row}")
    calibrate_in_place(row)
    print(f"  row after:  {row}")

    row = readings[0].copy()
    calibrated = calibrate(row)
    print("calibrated = calibrate(row) on a fresh copy:")
    print(f"  row after:  {row}")
    print(f"  calibrated: {calibrated}")
    print()


def demo_statistical_operations(readings):
    """Demonstrate statistical operations."""
    print("=== Statistical Operations ===")

    # Overall statistics
    print(f"Overall average: {readings.mean():.1f} mmHg")
    print(f"Overall std dev: {readings.std():.1f} mmHg")
    print(f"Highest reading: {readings.max()}")
    print(f"Lowest reading: {readings.min()}")
    print()

    # Axis-specific operations
    patient_averages = readings.mean(axis=1)  # Average per patient
    visit_averages = readings.mean(axis=0)  # Average per visit

    print("Patient averages (first 5):")
    print(patient_averages[:5])
    print("\nVisit averages:")
    for i, avg in enumerate(visit_averages, 1):
        print(f"  Visit {i}: {avg:.1f}")
    print()

def demo_std_by_hand(readings):
    """Recompute the standard deviation from its definition and compare with .std()."""
    print("=== Standard Deviation by Hand ===")

    # Distance of every reading from the overall mean, squared, averaged, then square-rooted.
    by_hand = np.sqrt(((readings - readings.mean()) ** 2).mean())
    print(f"readings.std(): {readings.std():.4f} mmHg")
    print(f"By hand:        {by_hand:.4f} mmHg")
    print()

def demo_boolean_indexing(readings):
    """Demonstrate boolean indexing."""
    print("=== Boolean Indexing ===")

    # Calculate patient averages
    patient_averages = readings.mean(axis=1)

    # Find the patients who run high
    uncontrolled = patient_averages > 90
    print(f"Patients averaging above 90 mmHg: {uncontrolled.sum()}")
    print(f"First five of their averages: {patient_averages[uncontrolled][:5]}")

    # Multiple conditions
    stage_2 = patient_averages >= 90
    stage_1 = (patient_averages >= 80) & (patient_averages < 90)
    normal = patient_averages < 80

    print(f"\nDiastolic stages:")
    print(f"  Stage 2 (90+): {stage_2.sum()} patients")
    print(f"  Stage 1 (80-89): {stage_1.sum()} patients")
    print(f"  Normal (below 80): {normal.sum()} patients")
    print()

def demo_conditional_labels(readings):
    """Label values by a rule with np.where()."""
    print("=== Conditional Labels (np.where) ===")

    patient_averages = readings.mean(axis=1)
    labels = np.where(patient_averages >= 90, "refer", "monitor")
    print(f"First five averages: {patient_averages[:5]}")
    print(f"First five labels:   {labels[:5]}")
    print(f"Patients to refer: {(labels == 'refer').sum()}")

    # The same call can substitute values instead of text labels.
    kept = np.where(readings[0] >= 90, readings[0], 0)
    print(f"\nPatient 0 readings:    {readings[0]}")
    print(f"Stage 2 visits only:   {kept}")
    print()


def demo_array_reshaping(readings):
    """Demonstrate array reshaping."""
    print("=== Array Reshaping ===")

    # Get first 12 readings
    sample = readings.flatten()[:12]
    print(f"Flattened sample (12 readings): {sample}")

    # Reshape to different dimensions
    reshaped = sample.reshape(3, 4)
    print(f"\nReshaped to 3x4:")
    print(reshaped)

    # Transpose
    print(f"\nTransposed (4x3):")
    print(reshaped.T)
    print()

def demo_practical_analysis(readings):
    """Demonstrate practical analysis workflow."""
    print("=== Practical Analysis Workflow ===")

    # Find the visit with the highest and lowest clinic-wide average
    visit_averages = readings.mean(axis=0)
    lowest_idx = visit_averages.argmin()
    highest_idx = visit_averages.argmax()

    print(f"Lowest-average visit: #{lowest_idx + 1} (avg: {visit_averages[lowest_idx]:.1f} mmHg)")
    print(f"Highest-average visit: #{highest_idx + 1} (avg: {visit_averages[highest_idx]:.1f} mmHg)")

    # Find the 4 patients with the highest averages. Three patients tie for fifth
    # at 90.6 mmHg, so stopping at four keeps every rank unambiguous.
    patient_averages = readings.mean(axis=1)
    top_4_indices = np.argsort(patient_averages)[-4:][::-1]

    print(f"\nHighest 4 patient averages:")
    for rank, idx in enumerate(top_4_indices, 1):
        print(f"  #{rank}: Patient {idx:3d}, average {patient_averages[idx]:.1f} mmHg")

    # Calculate the change from the first visit to the last
    change = readings[:, -1] - readings[:, 0]
    rose = (change > 0).sum()
    average_rise = change[change > 0].mean()

    print(f"\nChange from visit 1 to visit 5:")
    print(f"  Patients whose reading rose: {rose}")
    print(f"  Average rise: {average_rise:.1f} mmHg")
    print()

def main():
    """Run all NumPy demos."""
    print("Blood Pressure Analysis with NumPy")
    print("=" * 50)
    print()

    # Create data
    readings = create_sample_data()

    # Run demos
    demo_basic_operations(readings)
    demo_views_and_copies(readings)
    demo_function_changes_input(readings)
    demo_statistical_operations(readings)
    demo_std_by_hand(readings)
    demo_boolean_indexing(readings)
    demo_conditional_labels(readings)
    demo_array_reshaping(readings)
    demo_practical_analysis(readings)

    print("NumPy analysis complete.")

if __name__ == "__main__":
    main()
