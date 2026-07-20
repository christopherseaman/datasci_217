"""Supply the CSV-to-ndarray boundary for Lecture 03 Demo 3."""

import csv

import numpy as np


def load_measurements(filename):
    """Return the fixture's numeric fields as a homogeneous 2D ndarray."""
    rows = []

    with open(filename, "r", encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            rows.append([float(row["baseline"]), float(row["follow_up"])])

    return np.array(rows, dtype=np.float64)
