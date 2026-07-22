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

**12 points** | Phase 7

Random splitting would allow future weather into training. Keep only eligible forecast rows and create the fixed train, validation, and test periods from each target instant.

## Setup

```python
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("output/q4_features.csv")
features = pd.read_csv(
    INPUT_PATH,
    parse_dates=["cutoff_timestamp_utc", "target_timestamp_utc"],
)

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

## Fixed Chronological Splits

Apply the contract boundaries to eligible rows by target instant. Keep missing predictor values; Q7's train-fitted imputer handles them.

```python
# TODO: Filter model_eligible rows and assign train, validation, or test from
# the target instant corresponding to each America/Chicago boundary.
```

## Save X and y Handoffs

For each split, sort by target UTC then station. X and y must use the same unique row IDs in the same order.

```python
# TODO: Save q6_X_train/validation/test.csv with X_COLUMNS.
# TODO: Save q6_y_train/validation/test.csv with Y_COLUMNS.
```

## Split Summary

```python
SUMMARY_COLUMNS = ["split", "n_rows", "target_start", "target_end", "n_features"]

# TODO: Save output/q6_split_summary.csv in train, validation, test order.
```

## Checkpoint

- [ ] Only eligible Q4 rows enter Q6.
- [ ] Splits use target instants and exact local boundaries.
- [ ] Each X/y pair has identical unique IDs and row order.
- [ ] X contains identifiers plus every fixed predictor in fixed order.
- [ ] Predictor missingness is preserved for train-fitted preprocessing.

Next: [`q7_modeling.ipynb`](q7_modeling.ipynb)
