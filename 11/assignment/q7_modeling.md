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

**12 points**

Choose one regressor from the pinned scikit-learn. Fit candidates on the training rows only and use validation performance to freeze your choice. A simple model is enough, and it does not need to beat persistence. Do not open any Q6 test file in this notebook.

[Lecture 10](https://github.com/christopherseaman/datasci_217/blob/main/10/README.md) and its [Demo 2](https://github.com/christopherseaman/datasci_217/blob/main/10/demo/demo2_sklearn_prediction.md) build the same kind of pipeline and compare it on a validation set.

## 7.1 Setup

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

X_train = pd.read_csv("output/q6_X_train.csv")
X_validation = pd.read_csv("output/q6_X_validation.csv")
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

## 7.2 Pipeline and Model Choice

Build a `ColumnTransformer` that applies `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to the station and `SimpleImputer(strategy="median")` to the numeric predictors, and put it in a `Pipeline` with one scikit-learn regressor. Set `random_state=217` and `n_jobs=1` when the regressor has them (check with `get_params()`). Fit every candidate on the training rows only, compare them on validation, and say in a Markdown cell which one you keep and why.

```python
# TODO: Import and configure the candidate regressor(s).
# TODO: Build the preprocessing and Pipeline, fit on training rows, and compare on validation.
```

## 7.3 Validation Predictions and Metrics

The persistence prediction is the current `air_temperature_c_t`. Compute MAE, RMSE, and R2 for both models from the same unrounded validation rows.

```python
PREDICTION_COLUMNS = [
    "row_id", "station_name", "target_timestamp_utc", "actual",
    "persistence_prediction", "model_prediction",
]
METRIC_COLUMNS = ["model", "mae", "rmse", "r2", "n"]

# TODO: Save output/q7_validation_predictions.csv and output/q7_validation_metrics.csv
# (model values persistence_baseline and student_model).
```

> **Checkpoint: `output/q7_validation_predictions.csv`**
> First line `row_id,station_name,target_timestamp_utc,actual,persistence_prediction,model_prediction`; the same line count as `q6_X_validation.csv`.

> **Checkpoint: `output/q7_validation_metrics.csv`**
> First line `model,mae,rmse,r2,n`; 3 lines.

## 7.4 Frozen Specification and Permutation Importance

Record the chosen regressor, not the whole pipeline: the module and class from your import line (`sklearn.linear_model` and `Ridge` for `from sklearn.linear_model import Ridge`), and its settings as `json.dumps(model.get_params(deep=False))` (Lecture 07). Then run `permutation_importance` on the fitted pipeline with the validation rows, `scoring="neg_mean_absolute_error"`, `n_repeats=10`, and `random_state=217`, and save `result.importances_mean` as `mean_mae_increase` and `result.importances_std` as `std_mae_increase`. A positive value means shuffling that feature made MAE worse.

```python
SPEC_COLUMNS = [
    "estimator_module", "estimator_class", "parameters_json",
    "feature_columns", "random_state",
]
IMPORTANCE_COLUMNS = ["feature", "mean_mae_increase", "std_mae_increase"]

# TODO: Save the one-row output/q7_model_spec.csv (feature_columns joined with "|", random_state 217).
# TODO: Save output/q7_permutation_importance.csv, one row per fixed predictor.
```

> **Checkpoint: `output/q7_model_spec.csv`**
> First line `estimator_module,estimator_class,parameters_json,feature_columns,random_state`; 2 lines.

> **Checkpoint: `output/q7_permutation_importance.csv`**
> First line `feature,mean_mae_increase,std_mae_increase`; 20 lines.

## Check Your Work

- [ ] No test file or result was opened.
- [ ] All preprocessing was fit on training rows, inside the pipeline.
- [ ] The regressor is from the pinned scikit-learn, with `random_state=217` and `n_jobs=1` where it has them.
- [ ] Both models' metrics use the same validation rows.
- [ ] Permutation importance uses validation rows, MAE scoring, 10 repeats, and seed 217.

Next: [`q8_results.ipynb`](q8_results.ipynb)
