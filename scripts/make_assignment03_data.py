#!/usr/bin/env python3
"""Build the supplied dataset for Assignment 03.

Course tooling, not part of the student handout: the assignment ships the CSV
this writes, and `03/assignment_checks/_public_checks.py` carries a copy of it
(`SUPPLIED_READINGS`), so a submission is scored against the file students were
given, never against its own copy. Rebuilding the data means updating that copy.

    python scripts/make_assignment03_data.py

The file is a synthetic telemetry-ward export: 300 patients, one row each, with
12 hourly automated systolic readings in mmHg. Monitor M04's cuff reads high,
which is the drift the assignment asks students to find.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "03" / "assignment" / "data" / "bp_readings.csv"

SEED = 20260309
HOURS = 12
MONITOR_PATIENTS = {"M01": 58, "M02": 44, "M03": 61, "M04": 39, "M05": 52, "M06": 46}
DRIFTING_MONITOR = "M04"
DRIFT_MMHG = 9.0
# Morning surge across the 12 monitored hours, in mmHg around each patient's baseline.
HOURLY_PATTERN = np.array([2.0, 5.0, 7.5, 9.0, 6.0, 3.0, 1.0, -1.0, -2.0, -3.0, -4.5, -5.0])
BASELINE_MEAN = 127.0
BASELINE_SD = 13.0
READING_SD = 5.0


def build_readings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return patient ids, monitor ids, and the (patients, hours) reading grid."""
    rng = np.random.default_rng(SEED)

    monitors = []
    for monitor, patients in MONITOR_PATIENTS.items():
        monitors.extend([monitor] * patients)
    monitors = np.array(monitors)
    rng.shuffle(monitors)
    count = len(monitors)

    baseline = np.clip(rng.normal(BASELINE_MEAN, BASELINE_SD, size=count), 95.0, 178.0)
    noise = rng.normal(0.0, READING_SD, size=(count, HOURS))
    drift = np.where(monitors == DRIFTING_MONITOR, DRIFT_MMHG, 0.0)

    readings = baseline[:, None] + HOURLY_PATTERN[None, :] + noise + drift[:, None]
    readings = np.clip(np.rint(readings), 80, 200).astype(int)

    patients = np.array([f"P{number:04d}" for number in range(1, count + 1)])
    return patients, monitors, readings


def main() -> None:
    patients, monitors, readings = build_readings()

    header = "patient_id,monitor," + ",".join(f"sbp_h{hour:02d}" for hour in range(1, HOURS + 1))
    lines = [header]
    for patient, monitor, row in zip(patients, monitors, readings):
        lines.append(f"{patient},{monitor}," + ",".join(str(value) for value in row))
    text = "\n".join(lines) + "\n"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"{OUTPUT}: {len(readings)} patients x {HOURS} hours")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
