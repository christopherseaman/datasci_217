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

**11 points**

An absent source row is different from a measured zero. Build a complete station-by-hour panel so later lags refer to exact hours and sensor dropouts stay visible.

## 3.1 Setup

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

clean = pd.read_csv(INPUT_PATH)
clean["measurement_timestamp_utc"] = pd.to_datetime(clean["measurement_timestamp_utc"], utc=True)
```

## 3.2 Complete Hourly Panel

Convert local 2022-01-01 00:00 and local 2025-01-01 00:00 to UTC and build every hour between them with `pd.date_range(..., freq="h", inclusive="left")`. Cross the hours with both stations (`how="cross"`, Lecture 06), then left-join the cleaned rows with `indicator=True`. Do not fill the hours that have no source row.

```python
PANEL_COLUMNS = [
    "station_name", "measurement_timestamp_utc", *SENSOR_COLUMNS,
    "source_observed", "hour", "day_of_week", "month",
]

# TODO: Build the station-by-hour grid, join the cleaned rows, and add the local calendar columns.
# TODO: Sort by UTC time, then station, and save output/q3_hourly_panel.csv.
```

> **Checkpoint: `output/q3_hourly_panel.csv`**
> First line `station_name,measurement_timestamp_utc,air_temperature_c,wet_bulb_temperature_c,relative_humidity_pct,rain_intensity_mm_per_hour,interval_rain_mm,total_rain_mm,precipitation_type_code,wind_direction_deg,wind_speed_mps,maximum_wind_speed_mps,barometric_pressure_hpa,solar_radiation_w_m2,battery_voltage_v,source_observed,hour,day_of_week,month`; 2 times the `expected_hours` in your `q1_station_coverage.csv`, plus the header.

## 3.3 Gap Summary

A gap run is one or more consecutive `source_observed == False` hours within a station. Count the runs and the longest run. A `for` loop over one station's `source_observed` values in time order works (Lecture 02): a run starts at each False that follows a True or starts the series.

```python
SUMMARY_COLUMNS = [
    "station_name", "expected_hours", "observed_hours", "missing_hours",
    "gap_runs", "longest_gap_hours",
]

# TODO: Summarize each station's panel coverage and gap runs.
# TODO: Save output/q3_panel_summary.csv, one row per station.
```

> **Checkpoint: `output/q3_panel_summary.csv`**
> First line `station_name,expected_hours,observed_hours,missing_hours,gap_runs,longest_gap_hours`; 3 lines.

## Check Your Work

- [ ] Both stations have the same complete sequence of UTC hours.
- [ ] `source_observed` separates source rows from gaps.
- [ ] Gap hours stay missing in every sensor column.
- [ ] `hour`, `day_of_week`, and `month` describe the same instant in Chicago time.
- [ ] Gap runs restart for each station.

Next: [`q4_feature_engineering.ipynb`](q4_feature_engineering.ipynb)
