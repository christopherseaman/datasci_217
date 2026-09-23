# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.3.3",
# ]
# ///
"""Build the synthetic clinic encounter fixture used by Lecture 03's Demo 3.

The file is synthetic: identifiers are sequential (P0001, P0002, ...) and every
reading is drawn from a seeded generator, so no real patient is involved and the
demo prints the same numbers on every machine.

Usage:
    python scripts/build_lecture03_demo_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "03" / "demo" / "encounters.csv"
SEED = 23  # chosen so the first rows carry both follow-up labels the demo prints

# Clinic, number of encounters, mean systolic pressure (mmHg), age range (years).
# The means differ by clinic the way a real case mix does: nephrology and
# cardiology see more uncontrolled hypertension than obstetrics or dermatology.
CLINICS = (
    ("Cardiology", 260, 138.0, (40, 91)),
    ("Dermatology", 145, 122.0, (18, 81)),
    ("Endocrinology", 190, 132.0, (25, 86)),
    ("Nephrology", 125, 142.0, (35, 91)),
    ("Neurology", 150, 130.0, (25, 86)),
    ("Obstetrics", 120, 115.0, (18, 46)),
    ("Oncology", 130, 126.0, (30, 91)),
    ("Primary Care", 380, 127.0, (18, 91)),
)
SYSTOLIC_SD = 12.0
SYSTOLIC_LIMITS = (88, 190)  # plausible clinic readings, hypotensive to hypertensive crisis


def draw_systolic(rng, mean, count):
    """Draw whole-mmHg readings around a clinic mean, redrawing implausible ones.

    Redrawing rather than clipping keeps the tails smooth; clipping would stack
    several encounters on the boundary value itself.
    """
    low, high = SYSTOLIC_LIMITS
    readings = rng.normal(mean, SYSTOLIC_SD, size=count).round()
    outside = (readings < low) | (readings > high)
    while outside.any():
        readings[outside] = rng.normal(mean, SYSTOLIC_SD, size=outside.sum()).round()
        outside = (readings < low) | (readings > high)
    return readings.astype(int)


def build_rows(rng):
    """Draw one encounter per row, then shuffle the clinics together."""
    ages = []
    systolic = []
    clinics = []
    for name, count, mean, (low, high) in CLINICS:
        ages.append(rng.integers(low, high, size=count))
        systolic.append(draw_systolic(rng, mean, count))
        clinics.append(np.repeat(name, count))

    ages = np.concatenate(ages)
    systolic = np.concatenate(systolic)
    clinics = np.concatenate(clinics)

    order = rng.permutation(ages.size)
    return ages[order], systolic[order], clinics[order]


def main():
    """Write the fixture and report what it contains."""
    rng = np.random.default_rng(SEED)
    ages, systolic, clinics = build_rows(rng)

    lines = ["patient_id,age,systolic_bp,clinic"]
    for number, (age, reading, clinic) in enumerate(zip(ages, systolic, clinics), start=1):
        lines.append(f"P{number:04d},{age},{reading},{clinic}")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {ages.size} encounters")
    print(f"Systolic mmHg: min {systolic.min()}, mean {systolic.mean():.1f}, max {systolic.max()}")
    print(f"Age years: min {ages.min()}, max {ages.max()}")


if __name__ == "__main__":
    main()
