"""Small, import-safe vital-sign helpers for the Lecture 02 demos."""


def mean_reading(readings):
    """Return the average reading, or None when there are no readings."""
    if not readings:
        return None
    return sum(readings) / len(readings)


def highest_reading(readings):
    """Return the largest reading, or None when there are no readings."""
    if not readings:
        return None
    return max(readings)


def get_systolic(encounters):
    """Return the systolic readings stored in encounter records."""
    readings = []
    for encounter in encounters:
        readings.append(encounter["systolic"])
    return readings


def format_readings(encounters):
    """Return one display line for each encounter record."""
    lines = []
    for encounter in encounters:
        lines.append(f"{encounter['patient_id']}: {encounter['systolic']} mmHg")
    return lines
