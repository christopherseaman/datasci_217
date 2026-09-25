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

**11 points**

A sensor reading can hold one invalid value without making the whole observation useless. In this phase, reject the rows whose station and time cannot be trusted, keep every other row, and turn only rule-breaking sensor values into missing data. Do not smooth away real gaps or unusual but valid weather.

## 2.1 Setup

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

## 2.2 Valid Station-Time Keys

Parse the naive local timestamps, localize them with `ambiguous="NaT"` and `nonexistent="NaT"`, and convert the accepted ones to UTC. Reject the rows whose station is not a release station or whose time became `NaT`: in this release, ambiguous fall-back hours.

```python
# TODO: Localize, reject invalid keys, and add measurement_timestamp_utc to the kept rows.
```

## 2.3 Sensor Rules

Apply every rule in the table in [`assignment.md`](assignment.md#q2-data-cleaning). Turn unreadable values into missing first with `pd.to_numeric(..., errors="coerce")`. Do not fill, interpolate, or clip. Solar values from -20 up to but not including 0 become 0; solar values outside -20 to 1500 become missing. Count the values each rule changes as you go; section 2.4 saves the counts.

```python
OUTPUT_COLUMNS = RELEASE_COLUMNS + ["measurement_timestamp_utc"]

# TODO: Apply each sensor rule and record how many values it changed.
# TODO: Sort by UTC time, then station, and save output/q2_cleaned_observations.csv.
```

> **Checkpoint: `output/q2_cleaned_observations.csv`**
> First line `station_name,measurement_timestamp,air_temperature_c,wet_bulb_temperature_c,relative_humidity_pct,rain_intensity_mm_per_hour,interval_rain_mm,total_rain_mm,precipitation_type_code,wind_direction_deg,wind_speed_mps,maximum_wind_speed_mps,barometric_pressure_hpa,solar_radiation_w_m2,battery_voltage_v,measurement_timestamp_utc`; the 50,895 release rows less your `rows_rejected` count, plus the header.

## 2.4 Audit and Missingness

The audit has one row per rule: the key rule with `result` `rows_rejected`, one row for each of the 13 sensor columns with `result` `set_missing`, and the solar near-zero correction with `result` `set_to_zero`. `affected_values` is the number of rows or values the rule changed, 0 when it changed none; a sensor value that was already missing is not counted. `rule` is your own short, unique description. Then count what is missing in each station's cleaned sensor columns.

```python
AUDIT_COLUMNS = ["rule", "affected_values", "result"]
MISSINGNESS_COLUMNS = ["station_name", "column_name", "missing_count", "missing_pct"]

# TODO: Save output/q2_cleaning_audit.csv, one row per rule.
# TODO: Save output/q2_missingness.csv, one row per station and sensor column.
```

> **Checkpoint: `output/q2_cleaning_audit.csv`**
> First line `rule,affected_values,result`; 16 lines.

> **Checkpoint: `output/q2_missingness.csv`**
> First line `station_name,column_name,missing_count,missing_pct`; 27 lines.

## Check Your Work

- [ ] The 15 release columns are followed only by `measurement_timestamp_utc`.
- [ ] Only rows with an unknown station or a `NaT` time were rejected.
- [ ] Invalid sensor values became missing without dropping otherwise valid rows.
- [ ] Nothing was filled, interpolated, or clipped.
- [ ] Every rule has an audit row, and every station and sensor column has a missingness row.

Next: [`q3_data_wrangling.ipynb`](q3_data_wrangling.ipynb)
