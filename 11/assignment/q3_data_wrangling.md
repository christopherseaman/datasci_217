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

# Q3: Data Wrangling

**12 points** | Phase 4

An absent source row is different from a measured zero. Build a complete station-by-elapsed-hour panel so later lags refer to exact hours and sensor dropouts remain visible.

## Setup

```python
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("output/q2_cleaned_observations.csv")
OUTPUT_DIR = Path("output")
SOURCE_TIMEZONE = "America/Chicago"
STATIONS = ["Foster Weather Station", "Oak Street Weather Station"]
SENSOR_COLUMNS = [
    "air_temperature_c", "wet_bulb_temperature_c", "relative_humidity_pct",
    "rain_intensity_mm_per_hour", "interval_rain_mm", "total_rain_mm",
    "precipitation_type_code", "wind_direction_deg", "wind_speed_mps",
    "maximum_wind_speed_mps", "barometric_pressure_hpa",
    "solar_radiation_w_m2", "battery_voltage_v",
]

clean = pd.read_csv(INPUT_PATH, parse_dates=["measurement_timestamp_utc"])
```

## Complete Hourly Panel

Construct UTC endpoints from local `2022-01-01 00:00:00` and local `2025-01-01 00:00:00`. Cross every elapsed UTC hour with both stations, then left join observations. Do not fill structural sensor gaps.

```python
PANEL_COLUMNS = [
    "station_name", "measurement_timestamp_utc", *SENSOR_COLUMNS,
    "source_observed", "hour", "day_of_week", "month",
]

# TODO: Build the complete key grid, join source rows, and derive local calendar columns.
# TODO: Sort by UTC then station and save output/q3_hourly_panel.csv.
```

## Gap Summary

A gap run is one or more consecutive `source_observed == False` rows within a station. Count runs and the longest run in elapsed hours.

```python
SUMMARY_COLUMNS = [
    "station_name", "expected_hours", "observed_hours", "missing_hours",
    "gap_runs", "longest_gap_hours",
]

# TODO: Summarize complete-panel coverage and gap runs by station.
# TODO: Save output/q3_panel_summary.csv in station-name order.
```

## Checkpoint

- [ ] Both stations have exactly the same complete UTC-hour sequence.
- [ ] `source_observed` distinguishes source rows from structural gaps.
- [ ] Structural gaps remain missing in every sensor column.
- [ ] Local `hour`, `day_of_week`, and `month` describe the same UTC instant.
- [ ] Gap runs restart within each station.

Next: [`q4_feature_engineering.ipynb`](q4_feature_engineering.ipynb)
