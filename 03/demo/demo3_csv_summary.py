#!/usr/bin/env python3
"""Demo 3.4: summarize encounters.csv by clinic with Lecture 02 file reading and arrays.

Run it from the 03/demo folder so the relative filename below resolves:
    python demo3_csv_summary.py
It only prints; it writes no files.
"""

import numpy as np


def load_rows(filename):
    """Return one list of fields per data row, skipping the header line."""
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    return [line.strip().split(",") for line in lines[1:] if line.strip()]


def main():
    """Summarize the bundled fixture."""
    rows = load_rows("encounters.csv")

    print("Clinic Encounter Summary")
    print("=" * 50)
    print(f"Encounters: {len(rows)}")

    systolic = np.array([row[2] for row in rows]).astype(int)
    clinics = np.array([row[3] for row in rows])
    print(f"Systolic: shape {systolic.shape}, dtype {systolic.dtype}")

    print("\n=== Systolic pressure (mmHg) ===")
    print(f"Average: {systolic.mean():.1f}")
    print(f"Lowest:  {systolic.min()}")
    print(f"Highest: {systolic.max()}")

    print("\n=== Blood pressure stages (boolean masks) ===")
    stage_2 = systolic >= 140
    stage_1 = (systolic >= 130) & (systolic < 140)
    elevated = (systolic >= 120) & (systolic < 130)
    normal = systolic < 120
    print(f"Stage 2 (140+):     {stage_2.sum()}")
    print(f"Stage 1 (130-139):  {stage_1.sum()}")
    print(f"Elevated (120-129): {elevated.sum()}")
    print(f"Normal (below 120): {normal.sum()}")
    print(f"Stages total: {stage_2.sum() + stage_1.sum() + elevated.sum() + normal.sum()}")

    print("\n=== Follow-up labels (np.where) ===")
    labels = np.where(systolic >= 140, "refer", "routine")
    print(f"First five readings: {systolic[:5]}")
    print(f"First five labels:   {labels[:5]}")
    print(f"Needs referral: {(labels == 'refer').sum()}")

    print("\n=== Clinics (the counting a cut | sort | uniq -c pipeline does) ===")
    names = sorted(set(clinics))
    averages = []  # one average per clinic, in the same order as names
    for clinic in names:
        in_clinic = clinics == clinic  # one True or False per encounter, lined up with systolic
        average = systolic[in_clinic].mean()
        averages.append(average)
        print(f"{clinic}: {in_clinic.sum()} encounters, average {average:.1f} mmHg")
    highest = np.array(averages).argmax()  # a position in averages, which is the same position in names
    print(f"Highest-average clinic: {names[highest]} ({averages[highest]:.1f} mmHg)")


if __name__ == "__main__":
    main()
