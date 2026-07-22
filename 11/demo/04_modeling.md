---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Demo 4: Compare, freeze, and report

**Assignment patterns:** Q7 modeling, Q8 results, and Q9 reporting.

We compare a fixed weekly baseline (`lag_168`) with one transparent scikit-learn
pipeline. Validation MAE freezes the choice. Only then do we combine train and
validation, refit the pipeline if selected, and evaluate June exactly once.

There is no performance threshold and no feature-importance requirement. Honest
evaluation and clear evidence are the goals.

## One setup cell

```python
import importlib.metadata as metadata
import importlib.util
import subprocess
import sys

REQUIRED = {
    "numpy": "2.0.2", "pandas": "3.0.3", "pyarrow": "25.0.0",
    "scikit-learn": "1.9.0", "matplotlib": "3.11.1",
}
missing = []
for package, version in REQUIRED.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        missing.append(f"{package}=={version}")
if missing:
    if importlib.util.find_spec("pip") is None:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

from pathlib import Path
from urllib.request import urlretrieve

import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 217  # Ridge is deterministic and has no random_state parameter.
print(f"Python {sys.version.split()[0]} | scikit-learn {metadata.version('scikit-learn')}")
```

## Independently rebuild the model table and split

Fresh Colab downloads only the compact frozen panel. Development URLs use `main`;
immutable annual-tag replacement is pending release freeze.

```python
REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/data"

def acquire_authenticated_panel():
    filenames = ["demo_release_manifest.json", "yellow_taxi_2023_h1_zone_hour_counts.parquet"]
    for directory in (Path("data"), Path("11/demo/data")):
        if all((directory / filename).exists() for filename in filenames):
            break
    else:
        directory = Path("data")
        directory.mkdir(exist_ok=True)
        for filename in filenames:
            path = directory / filename
            if not path.exists():
                urlretrieve(f"{REPO_RAW}/{filename}", path)

    manifest_path = directory / filenames[0]
    panel_path = directory / filenames[1]
    expected_manifest_sha256 = "9d805f0759b8a5b0b17299cacc19038927de63d9d229bef88ccf22764a0af368"
    expected_panel_sha256 = "6c5658bd1d076930a9c552372fb3fb3d5dd71efbc4e4a736b5695e14f5d7b574"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected_manifest_sha256, (
        "Manifest hash mismatch"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifacts"]["panel"]["sha256"] == expected_panel_sha256
    assert hashlib.sha256(panel_path.read_bytes()).hexdigest() == expected_panel_sha256, (
        "Panel hash mismatch"
    )
    print("Authenticated manifest and panel SHA-256 digests.")
    return pd.read_parquet(panel_path)

def build_model_table(panel):
    table = panel.sort_values(["pickup_zone_id", "target_hour_utc"]).copy()
    local = table["target_hour_utc"].dt.tz_convert("America/New_York")
    table["target_hour_local"] = local
    table["hour_of_day"] = local.dt.hour
    table["day_of_week"] = local.dt.dayofweek
    table["month"] = local.dt.month
    table["is_weekend"] = local.dt.dayofweek.ge(5).astype("int8")
    grouped = table.groupby("pickup_zone_id", sort=False)["pickup_count"]
    for lag in (1, 24, 168):
        table[f"lag_{lag}"] = grouped.shift(lag)
    for window in (24, 168):
        table[f"rolling_mean_{window}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
    history = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168"]
    return table.dropna(subset=history).sort_values(
        ["target_hour_utc", "pickup_zone_id"]
    ).reset_index(drop=True)

table = build_model_table(acquire_authenticated_panel())
local_naive = table["target_hour_local"].dt.tz_localize(None)
train = table.loc[local_naive.lt("2023-05-01")].copy()
validation = table.loc[local_naive.between("2023-05-01", "2023-06-01", inclusive="left")].copy()
test = table.loc[local_naive.ge("2023-06-01")].copy()

assert len(table) == 50_100
assert len(train) + len(validation) + len(test) == len(table)
print({"train": len(train), "validation": len(validation), "test": len(test)})
```

## Define metrics, baseline, and one pipeline

Zone is categorical. Numeric fields are standardized before Ridge regression. The
pipeline keeps preprocessing learned from training attached to the estimator.

```python
CATEGORICAL = ["pickup_zone_id"]
NUMERIC = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "pickup_count"

preprocessor = ColumnTransformer([
    ("zone", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("numeric", StandardScaler(), NUMERIC),
])
ridge_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", Ridge(alpha=10.0)),
])

def clipped_predictions(values):
    return np.clip(np.asarray(values, dtype=float), 0, None)

def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
    }

ridge_pipeline.fit(train[FEATURES], train[TARGET])
validation_predictions = {
    "lag_168_baseline": clipped_predictions(validation["lag_168"]),
    "ridge_pipeline": clipped_predictions(ridge_pipeline.predict(validation[FEATURES])),
}
validation_scores = pd.DataFrame([
    {"candidate": name, **metrics(validation[TARGET], predictions)}
    for name, predictions in validation_predictions.items()
]).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
display(validation_scores)
```

## Freeze the choice, then evaluate test once

The next line is the only selection step. After it runs, validation has done its
job. Test predictions are made once and are not used to revise the decision.

```python
selected_candidate = validation_scores.loc[0, "candidate"]
development = pd.concat([train, validation], ignore_index=True)

if selected_candidate == "ridge_pipeline":
    final_estimator = ridge_pipeline.fit(development[FEATURES], development[TARGET])
    test_prediction = clipped_predictions(final_estimator.predict(test[FEATURES]))
else:
    final_estimator = None
    test_prediction = clipped_predictions(test["lag_168"])

test_scores = metrics(test[TARGET], test_prediction)
metrics_table = pd.concat([
    validation_scores.assign(split="validation", selected=lambda x: x["candidate"].eq(selected_candidate)),
    pd.DataFrame([{
        "candidate": selected_candidate, **test_scores, "split": "test", "selected": True,
    }]),
], ignore_index=True)

print("Frozen candidate:", selected_candidate)
display(metrics_table)
assert selected_candidate in validation_scores["candidate"].tolist()
assert np.isfinite(test_prediction).all() and (test_prediction >= 0).all()
```

## Save predictions and useful error slices

Overall metrics can hide where errors occur. We report test errors by zone and by
local hour. Small CSVs are evidence; the large reusable table remains Parquet.

```python
predictions = test[[
    "pickup_zone_id", "target_hour_utc", "target_hour_local", "hour_of_day", "pickup_count"
]].rename(columns={"pickup_count": "actual"})
predictions["prediction"] = test_prediction
predictions["absolute_error"] = (predictions["actual"] - predictions["prediction"]).abs()
predictions["squared_error"] = (predictions["actual"] - predictions["prediction"]) ** 2

def error_slice(frame, group_column):
    result = frame.groupby(group_column, as_index=False).agg(
        observations=("actual", "size"),
        actual_mean=("actual", "mean"),
        prediction_mean=("prediction", "mean"),
        MAE=("absolute_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
    )
    result["RMSE"] = np.sqrt(result.pop("mean_squared_error"))
    return result

zone_errors = error_slice(predictions, "pickup_zone_id")
hour_errors = error_slice(predictions, "hour_of_day")
display(zone_errors)
display(hour_errors)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
predictions.to_parquet(output_dir / "04_test_predictions.parquet", index=False)
metrics_table.to_csv(output_dir / "04_metrics.csv", index=False)
zone_errors.to_csv(output_dir / "04_zone_error_summary.csv", index=False)
hour_errors.to_csv(output_dir / "04_hour_error_summary.csv", index=False)
```

## Make two ordinary report figures

These figures answer common reporting questions without adding decorative
complexity: how predictions track actual counts over time, and where MAE is larger.

```python
hourly = predictions.groupby("target_hour_utc", as_index=False)[["actual", "prediction"]].sum()
figure, axis = plt.subplots(figsize=(11, 4))
axis.plot(hourly["target_hour_utc"], hourly["actual"], label="Actual", linewidth=1)
axis.plot(hourly["target_hour_utc"], hourly["prediction"], label="Prediction", linewidth=1)
axis.set(title="June hourly pickups across 12 zones", xlabel="Target hour (UTC)", ylabel="Pickups")
axis.legend()
figure.tight_layout()
figure.savefig(output_dir / "04_actual_vs_predicted.png", dpi=150)
plt.show()

figure, axis = plt.subplots(figsize=(10, 4))
axis.bar(hour_errors["hour_of_day"], hour_errors["MAE"], color="#287271")
axis.set(title="Test MAE by local hour", xlabel="Local hour", ylabel="MAE", xticks=range(24))
figure.tight_layout()
figure.savefig(output_dir / "04_mae_by_hour.png", dpi=150)
plt.show()
```

## Final checks

```python
assert len(predictions) == len(test)
assert predictions[["actual", "prediction", "absolute_error"]].notna().all().all()
assert len(zone_errors) == 12 and len(hour_errors) == 24
assert np.isclose(test_scores["MAE"], predictions["absolute_error"].mean())
assert np.isclose(test_scores["RMSE"], np.sqrt(predictions["squared_error"].mean()))
for filename in (
    "04_test_predictions.parquet", "04_metrics.csv", "04_zone_error_summary.csv",
    "04_hour_error_summary.csv", "04_actual_vs_predicted.png", "04_mae_by_hour.png",
):
    assert (output_dir / filename).stat().st_size > 0

print(
    f"Final checks passed: {selected_candidate} test MAE={test_scores['MAE']:.3f}, "
    f"RMSE={test_scores['RMSE']:.3f}; six evidence files saved."
)
```
