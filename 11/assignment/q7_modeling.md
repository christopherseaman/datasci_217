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

# Q7: Modeling

**14 points** | Phase 8

Choose one regressor from pinned scikit-learn. Fit candidates on training rows only and use validation performance to freeze your choice. A simple model is enough, and it does not need to beat persistence. Do not read any Q6 test file in this notebook.

Prerequisite refresher: [Lecture 10](../../10/README.md) and [Lecture 10 Demo 2](../../10/demo/demo2_ml_boosting.ipynb) cover train-fitted pipelines and validation-based model comparison.

## Setup

```python
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

X_train = pd.read_csv("output/q6_X_train.csv", parse_dates=["cutoff_timestamp_utc", "target_timestamp_utc"])
X_validation = pd.read_csv("output/q6_X_validation.csv", parse_dates=["cutoff_timestamp_utc", "target_timestamp_utc"])
y_train = pd.read_csv("output/q6_y_train.csv")
y_validation = pd.read_csv("output/q6_y_validation.csv")

FEATURE_COLUMNS = [
    "station_name", "air_temperature_c_t", "relative_humidity_pct_t",
    "interval_rain_mm_t", "wind_speed_mps_t", "maximum_wind_speed_mps_t",
    "barometric_pressure_hpa_t", "solar_radiation_w_m2_t",
    "wind_direction_sin_t", "wind_direction_cos_t",
    "air_temperature_lag_1h_c", "air_temperature_lag_24h_c",
    "air_temperature_lag_168h_c", "air_temperature_mean_past_24h_c",
    "air_temperature_change_1h_c", "target_hour_sin", "target_hour_cos",
    "target_day_of_year_sin", "target_day_of_year_cos",
]
CATEGORICAL_FEATURES = ["station_name"]
NUMERIC_FEATURES = FEATURE_COLUMNS[1:]
```

## Pipeline and Model Choice

Build a `ColumnTransformer` that uses `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for station and `SimpleImputer(strategy="median")` for numeric features. Put it and one chosen sklearn regressor in a `Pipeline`. Set `random_state=217` and `n_jobs=1` when supported. Candidate fitting uses training only.

```python
# TODO: Import and configure one sklearn regressor.
# TODO: Build the required preprocessing and Pipeline, fit training candidates,
# and use validation results to freeze one final pipeline.
```

## Validation Predictions and Metrics

The persistence prediction is current `air_temperature_c_t`. Calculate MAE, RMSE, and R2 from identical unrounded rows for both models.

```python
PREDICTION_COLUMNS = [
    "row_id", "station_name", "target_timestamp_utc", "actual",
    "persistence_prediction", "model_prediction",
]
METRIC_COLUMNS = ["model", "mae", "rmse", "r2", "n"]

# TODO: Save q7_validation_predictions.csv and q7_validation_metrics.csv.
```

## Frozen Specification and Permutation Importance

Record the selected regressor class and shallow parameters, not the entire pipeline parameter tree. Compute validation permutation importance through the fitted pipeline with `scoring="neg_mean_absolute_error"`, 10 repeats, and seed 217. Save `result.importances_mean` as `mean_mae_increase` and `result.importances_std` as `std_mae_increase`; a positive value means that permutation increased MAE.

```python
SPEC_COLUMNS = [
    "estimator_module", "estimator_class", "parameters_json",
    "feature_columns", "random_state",
]
IMPORTANCE_COLUMNS = ["feature", "mean_mae_increase", "std_mae_increase"]

# TODO: Save the one-row q7_model_spec.csv.
# TODO: Save q7_permutation_importance.csv in fixed feature order.
```

## Checkpoint

- [ ] No test file or outcome was accessed.
- [ ] All preprocessing was fit on training rows through a pipeline.
- [ ] The model is from pinned sklearn, with required seed/jobs when supported.
- [ ] Baseline and model metrics use identical validation rows.
- [ ] Permutation importance uses validation, MAE scoring, 10 repeats, seed 217.

Next: [`q8_results.ipynb`](q8_results.ipynb)
