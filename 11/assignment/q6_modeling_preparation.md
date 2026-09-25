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

# Q6: Modeling Preparation

**11 points**

A random split would let future weather into training. Keep only the eligible forecast rows and cut the fixed train, validation, and test periods by each row's target time.

## 6.1 Setup

```python
from pathlib import Path

import pandas as pd

SOURCE_TIMEZONE = "America/Chicago"
features = pd.read_csv("output/q4_features.csv")
for column in ["cutoff_timestamp_utc", "target_timestamp_utc"]:
    features[column] = pd.to_datetime(features[column], utc=True)

FIXED_PREDICTORS = [
    "station_name", "air_temperature_c_t", "relative_humidity_pct_t",
    "interval_rain_mm_t", "wind_speed_mps_t", "maximum_wind_speed_mps_t",
    "barometric_pressure_hpa_t", "solar_radiation_w_m2_t",
    "wind_direction_sin_t", "wind_direction_cos_t",
    "air_temperature_lag_1h_c", "air_temperature_lag_24h_c",
    "air_temperature_lag_168h_c", "air_temperature_mean_past_24h_c",
    "air_temperature_change_1h_c", "target_hour_sin", "target_hour_cos",
    "target_day_of_year_sin", "target_day_of_year_cos",
]
ID_COLUMNS = [
    "row_id", "station_name", "cutoff_timestamp_utc", "target_timestamp_utc",
]
X_COLUMNS = ID_COLUMNS + FIXED_PREDICTORS[1:]
Y_COLUMNS = ["row_id", "target_air_temperature_c"]
```

## 6.2 Fixed Chronological Splits

Keep the rows with `model_eligible` True, then assign each to train, validation, or test by comparing its `target_timestamp_utc` with the local-midnight boundaries in [`assignment.md`](assignment.md#split-boundaries), written as `pd.Timestamp("2024-01-01", tz="America/Chicago")`. Keep rows with missing predictors; Q7's imputer handles them.

```python
# TODO: Filter the eligible rows and label each one train, validation, or test.
```

## 6.3 Save X and y

Sort each split by target time, then station, and save X and y from the same sorted rows so their `row_id` values line up.

```python
# TODO: Save q6_X_train.csv, q6_X_validation.csv, and q6_X_test.csv with X_COLUMNS.
# TODO: Save q6_y_train.csv, q6_y_validation.csv, and q6_y_test.csv with Y_COLUMNS.
```

> **Checkpoint: `output/q6_X_train.csv`, `output/q6_X_validation.csv`, `output/q6_X_test.csv`**
> First line `row_id,station_name,cutoff_timestamp_utc,target_timestamp_utc,air_temperature_c_t,relative_humidity_pct_t,interval_rain_mm_t,wind_speed_mps_t,maximum_wind_speed_mps_t,barometric_pressure_hpa_t,solar_radiation_w_m2_t,wind_direction_sin_t,wind_direction_cos_t,air_temperature_lag_1h_c,air_temperature_lag_24h_c,air_temperature_lag_168h_c,air_temperature_mean_past_24h_c,air_temperature_change_1h_c,target_hour_sin,target_hour_cos,target_day_of_year_sin,target_day_of_year_cos`; that split's `n_rows` in `q6_split_summary.csv`, plus the header.

> **Checkpoint: `output/q6_y_train.csv`, `output/q6_y_validation.csv`, `output/q6_y_test.csv`**
> First line `row_id,target_air_temperature_c`; the same line count as the matching X file.

## 6.4 Split Summary

```python
SUMMARY_COLUMNS = ["split", "n_rows", "target_start", "target_end", "n_features"]

# TODO: Save output/q6_split_summary.csv with rows train, validation, and test.
```

> **Checkpoint: `output/q6_split_summary.csv`**
> First line `split,n_rows,target_start,target_end,n_features`; 4 lines.

## Check Your Work

- [ ] Only eligible Q4 rows are in the splits.
- [ ] The splits use target times and the exact local boundaries.
- [ ] Each X and y pair lists the same `row_id` values in the same order.
- [ ] X holds the identifiers and all 18 numeric predictors; `n_features` counts all 19 predictors.

Next: [`q7_modeling.ipynb`](q7_modeling.ipynb)
