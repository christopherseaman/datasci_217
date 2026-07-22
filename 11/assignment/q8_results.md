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

# Q8: Results

**14 points** | Phase 9

Your Q7 estimator, parameters, and feature order are frozen. Recreate the same pipeline, refit on training plus validation, and evaluate the July-December 2024 test period once. Test diagnostics describe the final result; they are not another tuning opportunity.

## Setup

```python
from pathlib import Path
import importlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

model_spec = pd.read_csv("output/q7_model_spec.csv").iloc[0]
validation_predictions = pd.read_csv("output/q7_validation_predictions.csv")

# TODO: Load Q6 train, validation, and test X/y files and verify ID alignment.
```

## Refit and Test Once

Import the frozen estimator class from the model specification, restore its recorded parameters, recreate the required preprocessing pipeline, and fit train plus validation. Then predict test once.

```python
TEST_PREDICTION_COLUMNS = [
    "row_id", "station_name", "target_timestamp_utc", "actual",
    "persistence_prediction", "model_prediction", "model_error",
    "model_absolute_error",
]
METRIC_COLUMNS = ["model", "mae", "rmse", "r2", "n"]

# TODO: Recreate and refit the frozen pipeline without changing the Q7 choice.
# TODO: Save q8_test_predictions.csv and q8_test_metrics.csv.
```

## Station Metrics

Calculate both models over identical test observations within each station.

```python
STATION_METRIC_COLUMNS = ["model", "station_name", "n", "mae", "rmse", "r2"]

# TODO: Save output/q8_station_metrics.csv ordered by model then station.
```

## Final Visualizations

Create one readable multi-panel figure containing a validation comparison, a test actual-versus-predicted view, and residual panels.

```python
# TODO: Save output/q8_final_visualizations.png.
```

## Checkpoint

- [ ] The Q7 estimator, parameters, and feature order stayed fixed.
- [ ] Final preprocessing and fitting used train plus validation only.
- [ ] Test was evaluated once on identical rows for both models.
- [ ] Overall and station metrics include MAE, RMSE, and R2.
- [ ] The final figure contains all required diagnostic content.

Next: [`q9_writeup.ipynb`](q9_writeup.ipynb)
