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

**16 points** | Phase 5

At each cutoff, you may use measurements available at or before that hour. The answer is air temperature exactly one elapsed hour later. Build features on the complete panel so every lag has an exact temporal meaning.

## Setup

```python
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_PATH = Path("output/q3_hourly_panel.csv")
panel = pd.read_csv(PANEL_PATH, parse_dates=["measurement_timestamp_utc"])
panel = panel.sort_values(["station_name", "measurement_timestamp_utc"])
```

## Forecast Rows and Features

Use grouped shifts and rolling operations so station histories never mix. The 24-hour mean includes cutoff and its prior 23 rows, ignores missing values, and uses `min_periods=1`.

Derive target calendar cycles after converting the target instant to `America/Chicago`. For target local integer `hour` (0-23), use `hour_angle = 2 * np.pi * hour / 24`, `target_hour_sin = np.sin(hour_angle)`, and `target_hour_cos = np.cos(hour_angle)`. For target local `dayofyear` (1-366), use `day_of_year_angle = 2 * np.pi * (dayofyear - 1) / 366`, `target_day_of_year_sin = np.sin(day_of_year_angle)`, and `target_day_of_year_cos = np.cos(day_of_year_angle)`. The denominator remains 366 for every year.

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

# TODO: Build exact next-hour targets and all fixed features within station.
# TODO: Mark eligibility, construct row IDs, select FEATURE_COLUMNS, and save
# output/q4_features.csv without dropping ineligible panel rows.
```

## Feature Manifest

Document one row per fixed predictor. Offsets are relative to cutoff; predictors may use offset 0 or earlier but never later. The `source` field may contain any concise, nonblank student description. Feature names, row order, offsets, and roles are fixed; exact `source` prose is not.

```python
MANIFEST_COLUMNS = [
    "feature_name", "source", "earliest_offset_hours",
    "latest_offset_hours", "role",
]

# TODO: Build the ordered fixed-predictor manifest and save
# output/q4_feature_manifest.csv.
```

## Timing Checks

```python
# TODO: Assert at least one exact lag value, one exact next-hour target, and
# that every manifest latest_offset_hours is <= 0.
```

## Checkpoint

- [ ] Features and targets restart within station and use elapsed UTC hours.
- [ ] Rolling mean timing and missing-value behavior match the contract.
- [ ] Eligibility requires observed current and exact next-hour temperatures.
- [ ] Row IDs are unique station slugs plus cutoff UTC hour.
- [ ] The manifest includes all and only fixed predictors in fixed order.

Next: [`q5_pattern_analysis.ipynb`](q5_pattern_analysis.ipynb)
