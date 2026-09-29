#!/usr/bin/env python3
"""Demo 3.1-3.3: select, reshape, summarize, label, and rank diastolic blood-pressure readings."""

import numpy as np

MMHG_TO_KPA = 0.133


def create_sample_data():
    """Create reproducible diastolic blood-pressure readings: 100 patients, 5 visits each."""
    rng = np.random.default_rng(42)

    n_patients = 100
    n_visits = 5

    # Diastolic readings in mmHg, 70 through 100.
    readings = rng.integers(70, 101, size=(n_patients, n_visits))

    print(f"Created data: {n_patients} patients, {n_visits} visits")
    print(f"Array shape: {readings.shape}")
    print(f"Data type: {readings.dtype}\n")

    return readings


def demo_basic_operations(readings):
    """Warm up with Demo 2's vectorized arithmetic on one patient's row."""
    print("=== Basic Operations ===")
    print(f"First patient's readings: {readings[0]}")
    print(f"In kPa (x0.133): {readings[0] * MMHG_TO_KPA}")
    print(f"Calibrated (+3 mmHg): {readings[0] + 3}")
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


def demo_boolean_indexing(readings):
    """Select readings, and whole rows, with masks on the 2-D array."""
    print("=== Boolean Indexing ===")

    # One True or False per reading: stage 2 is 90 mmHg or above.
    stage_2 = readings >= 90
    print(f"Readings of 90 mmHg or above: {stage_2.sum()} of {readings.size}")
    print(f"First five of them: {readings[stage_2][:5]}")

    stage_1 = (readings >= 80) & (readings < 90)
    print(f"Readings from 80 to 89 mmHg: {stage_1.sum()}")

    # A mask on one column keeps whole rows, so the result stays 2-D.
    high_first = readings[readings[:, 0] >= 98]
    print(f"Patients whose visit 1 was 98 mmHg or above: {high_first.shape[0]}")
    print("Their first three rows:")
    print(high_first[:3])
    print()


def demo_fancy_indexing(readings):
    """Pick columns by a list of positions."""
    print("=== Fancy Indexing ===")
    first_last = readings[:, [0, -1]]
    print(f"readings[:, [0, -1]] shape: {first_last.shape}")
    print("Visit 1 and visit 5, first three patients:")
    print(first_last[:3])
    print()


def demo_array_reshaping(readings):
    """Reshape 12 readings into a grid, then transpose it."""
    print("=== Array Reshaping ===")

    sample = readings.flatten()[:12]
    print(f"Flattened sample (12 readings): {sample}")

    reshaped = sample.reshape(3, 4)
    print("\nReshaped to 3x4:")
    print(reshaped)

    print("\nTransposed (4x3):")
    print(reshaped.T)
    print()


def demo_summary_statistics(readings):
    """Summarize every reading, then each patient (axis=1) and each visit (axis=0)."""
    print("=== Summary Statistics ===")

    print(f"Overall average: {readings.mean():.1f} mmHg")
    print(f"Overall median:  {np.median(readings):.1f} mmHg")
    print(f"Overall std dev: {readings.std():.1f} mmHg")
    print(f"Highest reading: {readings.max()}")
    print(f"Lowest reading: {readings.min()}")
    print(f"25th, 50th, 75th percentiles: {np.percentile(readings, [25, 50, 75])}")
    print()

    patient_averages = readings.mean(axis=1)  # one per patient
    visit_averages = readings.mean(axis=0)  # one per visit

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


def demo_conditional_labels(readings):
    """Label patients by rule with np.where() and np.select()."""
    print("=== Conditional Labels (np.where and np.select) ===")

    patient_averages = readings.mean(axis=1)
    labels = np.where(patient_averages >= 90, "refer", "monitor")
    print(f"First five averages: {patient_averages[:5]}")
    print(f"First five labels:   {labels[:5]}")
    print(f"Patients to refer: {(labels == 'refer').sum()}")

    # The same call can substitute values instead of text labels.
    kept = np.where(readings[0] >= 90, readings[0], 0)
    print(f"\nPatient 0 readings:    {readings[0]}")
    print(f"Stage 2 visits only:   {kept}")

    # Three labels: the first true condition wins, so the highest band comes first.
    stages = np.select(
        [patient_averages >= 90, patient_averages >= 80],
        ["stage 2", "stage 1"],
        default="normal",
    )
    print(f"\nFirst five stages: {stages[:5]}")
    print(f"  Stage 2 (90 mmHg or above): {(stages == 'stage 2').sum()} patients")
    print(f"  Stage 1 (80-89): {(stages == 'stage 1').sum()} patients")
    print(f"  Normal (below 80): {(stages == 'normal').sum()} patients")
    print()


def demo_sorting_and_ranking(readings):
    """Find extremes by position, rank patients, sort within rows, and follow the change over time."""
    print("=== Sorting and Ranking ===")

    visit_averages = readings.mean(axis=0)
    lowest_idx = visit_averages.argmin()
    highest_idx = visit_averages.argmax()
    print(f"Lowest-average visit: #{lowest_idx + 1} (avg: {visit_averages[lowest_idx]:.1f} mmHg)")
    print(f"Highest-average visit: #{highest_idx + 1} (avg: {visit_averages[highest_idx]:.1f} mmHg)")

    # The 4 patients with the highest averages. Three patients tie for fifth at
    # 90.6 mmHg, so stopping at four keeps every rank unambiguous.
    patient_averages = readings.mean(axis=1)
    top_4 = np.argsort(patient_averages)[-4:][::-1]
    print("\nHighest 4 patient averages:")
    for rank, idx in enumerate(top_4, 1):
        print(f"  #{rank}: Patient {idx:3d}, average {patient_averages[idx]:.1f} mmHg")
    print("Their readings, readings[top_4]:")
    print(readings[top_4])

    print("\nEach row sorted, np.sort(readings[:3], axis=1):")
    print(np.sort(readings[:3], axis=1))

    change = readings[:, -1] - readings[:, 0]
    rose = (change > 0).sum()
    average_rise = change[change > 0].mean()
    print("\nChange from visit 1 to visit 5:")
    print(f"  Patients whose reading rose: {rose}")
    print(f"  Average rise: {average_rise:.1f} mmHg")
    print()


def main():
    """Run the Demo 3 steps in the lecture's order."""
    print("Blood Pressure Analysis with NumPy")
    print("=" * 50)
    print()

    readings = create_sample_data()

    demo_basic_operations(readings)
    demo_function_changes_input(readings)
    demo_views_and_copies(readings)
    demo_boolean_indexing(readings)
    demo_fancy_indexing(readings)
    demo_array_reshaping(readings)
    demo_summary_statistics(readings)
    demo_std_by_hand(readings)
    demo_conditional_labels(readings)
    demo_sorting_and_ranking(readings)

    print("NumPy analysis complete.")


if __name__ == "__main__":
    main()
