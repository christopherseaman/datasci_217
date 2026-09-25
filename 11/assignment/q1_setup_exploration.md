---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Q1: Setup and Exploration

**6 points**

Before analyzing Chicago beach weather, confirm that you have the same frozen release as everyone else. Then look at each station's coverage, an ordinary sensor distribution, and the shape of the time series.

[`assignment.md`](assignment.md#q1-setup-and-exploration) defines each Q1 file. This notebook saves three of them.

## 1.1 Setup

Run this cell first. It prints the versions of Python and the packages in use, loads the manifest, and shows the first rows of the release.

```python
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

print("Python", sys.version.split()[0])
print("NumPy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__,
      "| Matplotlib", matplotlib.__version__)

DATA_PATH = Path("data/chicago_beach_sensors_2022_2024.csv")
MANIFEST_PATH = Path("data/release_manifest.json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SOURCE_TIMEZONE = "America/Chicago"

with MANIFEST_PATH.open(encoding="utf-8") as handle:
    manifest = json.load(handle)

weather = pd.read_csv(DATA_PATH)
weather.head()
```

## 1.2 Release Audit

Measure each fact from the file itself rather than copying the manifest into both columns. The Lecture 11 demo's `01_setup` notebook computes a file's hash with `hashlib.sha256(path.read_bytes()).hexdigest()` and its size with `path.stat().st_size`. In the `column_names` row, write both lists of names joined with `|` in file order.

```python
AUDIT_COLUMNS = ["check_name", "expected", "observed", "passed"]
AUDIT_CHECKS = [
    "release_filename", "release_sha256", "release_byte_size", "row_count",
    "column_count", "column_names", "source_timezone",
]

# TODO: Measure each observed fact and build the seven rows in AUDIT_CHECKS order.
# TODO: Save output/q1_release_audit.csv with AUDIT_COLUMNS and index=False.
```

> **Checkpoint: `output/q1_release_audit.csv`**
> First line `check_name,expected,observed,passed`; 8 lines.

## 1.3 Station Coverage

Parse and localize the timestamps here only to count coverage; Q2 saves the cleaned table.

```python
COVERAGE_COLUMNS = [
    "station_name", "expected_hours", "observed_hours", "missing_hours",
    "coverage_pct", "first_timestamp", "last_timestamp",
]

# TODO: Localize with ambiguous="NaT" and nonexistent="NaT", then count each station's valid hours.
# TODO: Count the expected elapsed UTC hours from local 2022-01-01 up to local 2025-01-01.
# TODO: Save output/q1_station_coverage.csv, one row per station.
```

> **Checkpoint: `output/q1_station_coverage.csv`**
> First line `station_name,expected_hours,observed_hours,missing_hours,coverage_pct,first_timestamp,last_timestamp`; 3 lines.

## 1.4 First Visualizations

Create one figure with at least two labeled panels: an ordinary sensor distribution and a time-series preview for each station. A short slice or a daily summary is easier to read than every point.

```python
# TODO: Build the figure and save it with plt.savefig("output/q1_visualizations.png").
```

> **Checkpoint: `output/q1_visualizations.png`**

## Check Your Work

- [ ] The seven audit rows are in order, each observed value measured from the file, and each `passed` is True.
- [ ] Coverage counts the valid localized hours over the full release window.
- [ ] The figure has titles, labeled axes, a distribution, and a time-series preview.
- [ ] Each CSV's first line and line count match its checkpoint.

Next: [`q2_data_cleaning.ipynb`](q2_data_cleaning.ipynb)
