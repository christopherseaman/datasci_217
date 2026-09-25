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

**13 points**

Your Q7 regressor, its settings, and the feature list are frozen. Rebuild the same pipeline, fit it on train plus validation, and evaluate the July to December 2024 test period once. The test results describe the final model; they are not another chance to tune it.

## 8.1 Setup

```python
from pathlib import Path
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
parameters = json.loads(model_spec["parameters_json"])
validation_predictions = pd.read_csv("output/q7_validation_predictions.csv")
print(model_spec["estimator_class"], parameters)

# TODO: Load the Q6 train, validation, and test X and y files and check that their row_id values line up.
```

## 8.2 Refit and Test Once

Import the same regressor class as in Q7 and create it with the same arguments, which `parameters` records, such as `Ridge(alpha=parameters["alpha"], random_state=217)`. Put it in the same preprocessing pipeline, fit it on train plus validation together (`pd.concat`), and predict the test rows once.

```python
TEST_PREDICTION_COLUMNS = [
    "row_id", "station_name", "target_timestamp_utc", "actual",
    "persistence_prediction", "model_prediction", "model_error",
    "model_absolute_error",
]
METRIC_COLUMNS = ["model", "mae", "rmse", "r2", "n"]

# TODO: Rebuild and refit the frozen pipeline without changing the Q7 choice.
# TODO: Save output/q8_test_predictions.csv and output/q8_test_metrics.csv.
```

> **Checkpoint: `output/q8_test_predictions.csv`**
> First line `row_id,station_name,target_timestamp_utc,actual,persistence_prediction,model_prediction,model_error,model_absolute_error`; the same line count as `q6_X_test.csv`.

> **Checkpoint: `output/q8_test_metrics.csv`**
> First line `model,mae,rmse,r2,n`; 3 lines.

## 8.3 Station Metrics

Compute both models' metrics within each station over the same test rows.

```python
STATION_METRIC_COLUMNS = ["model", "station_name", "n", "mae", "rmse", "r2"]

# TODO: Save output/q8_station_metrics.csv, one row per model and station.
```

> **Checkpoint: `output/q8_station_metrics.csv`**
> First line `model,station_name,n,mae,rmse,r2`; 5 lines.

## 8.4 Final Visualizations

Create one readable multi-panel figure: a validation comparison of the baseline and the model, a test actual-versus-predicted view, and residual panels.

```python
# TODO: Save the figure with plt.savefig("output/q8_final_visualizations.png").
```

> **Checkpoint: `output/q8_final_visualizations.png`**

## Check Your Work

- [ ] The Q7 regressor, settings, and feature list stayed the same.
- [ ] The final fit used train plus validation only.
- [ ] The test rows were predicted once, and both models' metrics use the same rows.
- [ ] Overall and station metrics include MAE, RMSE, and R2.
- [ ] The figure has all three views, with titles and labeled axes.

Next: [`q9_writeup.ipynb`](q9_writeup.ipynb)
