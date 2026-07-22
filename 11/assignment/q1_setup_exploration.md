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

**8 points** | Phases 1-2

Before analyzing Chicago beach weather, confirm that you have the same frozen release as everyone else. Then get acquainted with station coverage, ordinary sensor distributions, and the shape of the time series.

Read the exact Q1 schemas in [`assignment.md`](assignment.md). Produce all three Q1 artifacts.

## Setup

```python
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/chicago_beach_sensors_2022_2024.csv")
MANIFEST_PATH = Path("data/release_manifest.json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

with MANIFEST_PATH.open(encoding="utf-8") as handle:
    manifest = json.load(handle)

weather = pd.read_csv(DATA_PATH)
weather.head()
```

## Release Audit

Observe file facts independently rather than copying all manifest values into both sides of the comparison.

```python
AUDIT_COLUMNS = ["check_name", "expected", "observed", "passed"]
AUDIT_CHECKS = [
    "release_filename", "release_sha256", "release_byte_size", "row_count",
    "column_count", "column_names", "source_timezone",
]

# TODO: Compute the observed release facts and build the seven ordered rows.
# TODO: Save output/q1_release_audit.csv with AUDIT_COLUMNS and index=False.
```

## Station Coverage

Parse and localize timestamps only for this coverage audit. The cleaned handoff is Q2's job.

```python
COVERAGE_COLUMNS = [
    "station_name", "expected_hours", "observed_hours", "missing_hours",
    "coverage_pct", "first_timestamp", "last_timestamp",
]
SOURCE_TIMEZONE = "America/Chicago"

# TODO: Localize with ambiguous/nonexistent="NaT" and summarize valid station-hours.
# TODO: Save output/q1_station_coverage.csv in station-name order.
```

## First Visualizations

Create one figure with at least two labeled panels: an ordinary distribution and a station time-series preview. A short slice or aggregated preview is easier to read than every point.

```python
# TODO: Build the two-panel figure and save output/q1_visualizations.png.
```

## Checkpoint

- [ ] Seven independently observed release checks pass in the required order.
- [ ] Coverage uses the full release window and valid localized station-hour keys.
- [ ] The figure contains a labeled distribution and station time-series preview.
- [ ] CSV files have exact columns and no accidental index.

Next: [`q2_data_cleaning.ipynb`](q2_data_cleaning.ipynb)
