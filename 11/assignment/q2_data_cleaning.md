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

# Q2: Data Cleaning

**10 points** | Phase 3

Sensor readings can contain invalid values without making the entire observation useless. In this phase, reject invalid station-time keys, preserve valid rows, and turn only out-of-range sensor values into missing data. Do not smooth away real gaps or unusual but valid weather.

## Setup

```python
from pathlib import Path
import json

import numpy as np
import pandas as pd

DATA_PATH = Path("data/chicago_beach_sensors_2022_2024.csv")
MANIFEST_PATH = Path("data/release_manifest.json")
OUTPUT_DIR = Path("output")
SOURCE_TIMEZONE = "America/Chicago"

raw = pd.read_csv(DATA_PATH)
with MANIFEST_PATH.open(encoding="utf-8") as handle:
    manifest = json.load(handle)
RELEASE_COLUMNS = manifest["columns"]
SENSOR_COLUMNS = RELEASE_COLUMNS[2:]
```

## Valid Station-Time Keys

Parse naive local timestamps, localize with `ambiguous="NaT"` and `nonexistent="NaT"`, and convert accepted timestamps to UTC. Reject invalid/unparseable keys, including the six ambiguous fall-back rows.

```python
# TODO: Validate station names and timestamps, reject invalid keys, and add
# measurement_timestamp_utc to retained rows.
```

## Sensor Rules

Apply every inclusive range and code rule in [`assignment.md`](assignment.md). Coerce unparseable values to missing. Do not interpolate, fill, or clip. Only solar values in `[-20, 0)` become zero; solar values outside `[-20, 1500]` become missing.

```python
OUTPUT_COLUMNS = RELEASE_COLUMNS + ["measurement_timestamp_utc"]
AUDIT_COLUMNS = ["rule", "affected_values", "result"]
MISSINGNESS_COLUMNS = [
    "station_name", "column_name", "missing_count", "missing_pct",
]

# TODO: Apply each documented sensor rule and record affected-value counts.
# TODO: Sort by UTC then station and save output/q2_cleaned_observations.csv.
```

## Audit and Missingness

Use concise, unique, nonblank rule descriptions and clear result categories. Your `rule` wording does not need to match a prescribed phrase: grading checks the required result categories and affected counts, not exact prose. Report post-cleaning missingness for every station and every sensor measurement column.

```python
# TODO: Save output/q2_cleaning_audit.csv with key/timestamp and sensor rules.
# TODO: Save output/q2_missingness.csv in station and release-column order.
```

## Checkpoint

- [ ] The exact 15 source columns are followed only by UTC timestamp.
- [ ] Exactly the invalid keys, including six ambiguous rows, were rejected.
- [ ] Invalid sensor values became missing without dropping otherwise valid rows.
- [ ] No interpolation, filling, or general outlier clipping was used.
- [ ] Every cleaning rule and post-cleaning missing count is auditable.

Next: [`q3_data_wrangling.ipynb`](q3_data_wrangling.ipynb)
