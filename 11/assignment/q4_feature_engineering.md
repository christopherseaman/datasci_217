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

# Q4: Feature Engineering

**12 points**

At each cutoff hour you may use measurements from that hour or earlier. The answer is the air temperature exactly one elapsed hour later. Build the features on the complete panel so every lag has an exact meaning in hours.

## 4.1 Setup

```python
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_PATH = Path("output/q3_hourly_panel.csv")
SOURCE_TIMEZONE = "America/Chicago"

panel = pd.read_csv(PANEL_PATH)
panel["measurement_timestamp_utc"] = pd.to_datetime(panel["measurement_timestamp_utc"], utc=True)
panel = panel.sort_values(["station_name", "measurement_timestamp_utc"])
```

## 4.2 Forecast Rows and Features

Group by station before every `shift` and `rolling` call so one station's history never leaks into the other's (Lecture 09). The table in [`assignment.md`](assignment.md#q4-feature-engineering) gives each column's rule. The 24-hour mean covers the cutoff row and the 23 rows before it, ignores missing values, and uses `min_periods=1`, with no shift.

The cyclic features use Lecture 10's formula, `2 * np.pi * value / cycle_length`: wind direction with a 360-degree cycle, the target's Chicago hour (0 to 23) with 24, and the target's Chicago day of the year minus 1 with 366 every year.

```python
FEATURE_COLUMNS = [
    "row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc",
    "target_air_temperature_c", "model_eligible", "air_temperature_c_t",
    "relative_humidity_pct_t", "interval_rain_mm_t", "wind_speed_mps_t",
    "maximum_wind_speed_mps_t", "barometric_pressure_hpa_t",
    "solar_radiation_w_m2_t", "wind_direction_sin_t", "wind_direction_cos_t",
    "air_temperature_lag_1h_c", "air_temperature_lag_24h_c",
    "air_temperature_lag_168h_c", "air_temperature_mean_past_24h_c",
    "air_temperature_change_1h_c", "target_hour_sin", "target_hour_cos",
    "target_day_of_year_sin", "target_day_of_year_cos",
]

# TODO: Build the next-hour target and every fixed feature within station.
# TODO: Mark eligibility, build row_id, select FEATURE_COLUMNS, sort by cutoff then station,
# and save output/q4_features.csv without dropping ineligible rows.
```

> **Checkpoint: `output/q4_features.csv`**
> First line `row_id,station_name,cutoff_timestamp_utc,target_timestamp_utc,target_air_temperature_c,model_eligible,air_temperature_c_t,relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,maximum_wind_speed_mps_t,barometric_pressure_hpa_t,solar_radiation_w_m2_t,wind_direction_sin_t,wind_direction_cos_t,air_temperature_lag_1h_c,air_temperature_lag_24h_c,air_temperature_lag_168h_c,air_temperature_mean_past_24h_c,air_temperature_change_1h_c,target_hour_sin,target_hour_cos,target_day_of_year_sin,target_day_of_year_cos`; the same line count as `q3_hourly_panel.csv`.

## 4.3 Feature Manifest

Write one row per fixed predictor. The offsets are the earliest and latest hour the feature reads, relative to the cutoff: 0 is the cutoff hour and -1 the hour before, so `latest_offset_hours` is never above 0. `station_name` and the target calendar features read no earlier hour, so theirs are 0 and 0. `role` is `categorical` for `station_name` and `numeric` for the rest. `source` is your own short description of what the feature reads.

```python
MANIFEST_COLUMNS = [
    "feature_name", "source", "earliest_offset_hours",
    "latest_offset_hours", "role",
]

# TODO: Build the manifest in fixed-predictor order and save output/q4_feature_manifest.csv.
```

> **Checkpoint: `output/q4_feature_manifest.csv`**
> First line `feature_name,source,earliest_offset_hours,latest_offset_hours,role`; 20 lines.

## 4.4 Timing Checks

Show that your features mean what they say: print one station's rows around a gap and compare a lag and a target with the panel values by eye, and confirm that every manifest `latest_offset_hours` is 0 or less.

```python
# TODO: Print a few rows that show one exact lag, one exact next-hour target, and the manifest offsets.
```

## Check Your Work

- [ ] Features and targets restart for each station and use elapsed UTC hours.
- [ ] The rolling mean includes the cutoff, ignores missing values, and uses `min_periods=1`.
- [ ] Eligibility needs both the cutoff and the next-hour temperatures observed.
- [ ] Each `row_id` is the station slug and the cutoff's UTC hour, and none repeats.
- [ ] The manifest lists all 19 fixed predictors and nothing else.

Next: [`q5_pattern_analysis.ipynb`](q5_pattern_analysis.ipynb)
