"""Supply the CSV-to-ndarray boundary for Assignment 03."""

import csv
from pathlib import Path

import numpy as np


def load_measurements(filename):
    """Return the numeric fixture fields as a homogeneous 2D ndarray."""
    data_path = Path(filename)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    rows = []
    with data_path.open("r", encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            rows.append([float(row["baseline"]), float(row["follow_up"])])

    return np.array(rows, dtype=np.float64)
