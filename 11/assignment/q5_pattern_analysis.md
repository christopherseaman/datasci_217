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

# Q5: Pattern Analysis

**8 points** | Phase 6

Exploration can leak future information. Restrict this phase to targets before local 2024, then describe seasonal and hourly behavior without looking ahead to validation or test outcomes.

## Setup

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = Path("output/q4_features.csv")
features = pd.read_csv(
    INPUT_PATH,
    parse_dates=["cutoff_timestamp_utc", "target_timestamp_utc"],
)

# TODO: Create training_rows from target instants before local 2024.
```

## Monthly Station Summary

Summarize observed target air temperatures by station and local target year/month. `n_observed` counts nonmissing targets.

```python
SUMMARY_COLUMNS = [
    "station_name", "year", "month", "n_observed",
    "mean_air_temperature_c", "std_air_temperature_c",
    "min_air_temperature_c", "max_air_temperature_c",
]

# TODO: Aggregate and save output/q5_monthly_station_summary.csv.
```

## Current-Predictor Correlations

Create the required square Pearson matrix in the listed order. Correlation is descriptive, not proof that a predictor improves forecasts.

```python
CORRELATION_FEATURES = [
    "air_temperature_c_t", "relative_humidity_pct_t", "interval_rain_mm_t",
    "wind_speed_mps_t", "maximum_wind_speed_mps_t",
    "barometric_pressure_hpa_t", "solar_radiation_w_m2_t",
]

# TODO: Calculate the training-only square matrix and save
# output/q5_correlations.csv with row labels as the first column.
```

## Pattern Figure

Make one labeled figure that shows both monthly and local-hour temperature patterns from training rows only.

```python
# TODO: Save the combined figure as output/q5_patterns.png.
```

## Checkpoint

- [ ] Every calculation uses only targets before local 2024.
- [ ] Monthly counts count observed targets, not all panel rows.
- [ ] Correlation rows and columns have the exact required names and order.
- [ ] The figure includes monthly and local-hour patterns.

Next: [`q6_modeling_preparation.ipynb`](q6_modeling_preparation.ipynb)
