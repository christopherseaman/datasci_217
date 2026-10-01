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

Compare the weekly baseline, "same hour last week" (`lag_168`), with one transparent scikit-learn pipeline on the validation rows. The lower validation MAE freezes the choice. Only then refit the pipeline on training plus validation rows and evaluate both candidates on June exactly once, then look at where the errors fall. It uses Lecture 11 up to the demo break, plus pipelines, baselines, and metrics (Lecture 10), aggregation (Lecture 08), and saved figures (Lecture 07). Assignment 11's Q7 to Q9 follow the same pattern; Q7 also runs the permutation-importance check from Lecture 10's Demo 2, which this demo leaves out. There is no performance threshold: honest evaluation and clear evidence are the goals.

**How to run:** in Colab, open this notebook from the lecture page's Colab link. Locally, open the `11-demo` folder from Demo 1 in VS Code, open `04_modeling.ipynb`, and select the `.venv` kernel if VS Code does not show it; without that folder, follow Demo 1's "How to run locally" first. This notebook downloads the panel and rebuilds the model table and split itself, so it does not need the earlier demos' output. Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Tested 2026-09-30 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, scikit-learn 1.9.0, and matplotlib 3.11.1.

## Setup

```python
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.` If Colab asks you to restart the session, choose **Runtime → Restart session**, then continue with the next cell.

```python
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("pandas", pd.__version__)
print("scikit-learn", sklearn.__version__)
```

**Expect:** `pandas 3.0.5` and `scikit-learn 1.9.0` (Colab may show an older scikit-learn; the steps work the same way).

This notebook reads the release manifest and the zone-hour panel. This cell is supplied plumbing, as in Demo 1: it keeps any file already in `data/` and downloads the rest.

```python
REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/data"
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
for filename in ["demo_release_manifest.json", "yellow_taxi_2023_h1_zone_hour_counts.parquet"]:
    path = data_dir / filename
    if not path.exists():
        urlretrieve(f"{REPO_RAW}/{filename}", path)
    print(filename, path.stat().st_size, "bytes")
```

**Expect:** two lines: `demo_release_manifest.json 3585 bytes` and `yellow_taxi_2023_h1_zone_hour_counts.parquet 138733 bytes`.

## 1. Rebuild the model table and the split

This supplied cell repeats Demo 2's hash check and `build_model_table()`, and Demo 3's split, unchanged.

```python
manifest_path = data_dir / "demo_release_manifest.json"
published_manifest_sha256 = "558c28a8ab5a16769ac6ef9d170e7bd7f4ae4ef5d2a9e2b11fd2fb84d79b2c9d"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == published_manifest_sha256, "The manifest changed: rename it in data/ and rerun setup"
with open(manifest_path) as file:
    manifest = json.load(file)
panel_path = data_dir / manifest["artifacts"]["panel"]["filename"]
panel_sha256 = hashlib.sha256(panel_path.read_bytes()).hexdigest()
assert panel_sha256 == manifest["artifacts"]["panel"]["sha256"], "The panel changed: rename it in data/ and rerun"
panel = pd.read_parquet(panel_path)

def build_model_table(panel):
    table = panel.sort_values(["pickup_zone_id", "target_hour_utc"]).copy()
    local = table["target_hour_utc"].dt.tz_convert("America/New_York")
    table["target_hour_local"] = local
    table["hour_of_day"] = local.dt.hour
    table["day_of_week"] = local.dt.dayofweek
    table["month"] = local.dt.month
    table["is_weekend"] = (local.dt.dayofweek >= 5).astype("int8")

    grouped = table.groupby("pickup_zone_id", sort=False)["pickup_count"]
    for lag in (1, 24, 168):
        table[f"lag_{lag}"] = grouped.shift(lag)
    for window in (24, 168):
        table[f"rolling_mean_{window}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )

    history_columns = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168"]
    return table.dropna(subset=history_columns).sort_values(
        ["target_hour_utc", "pickup_zone_id"]
    ).reset_index(drop=True)

table = build_model_table(panel)
validation_start = pd.Timestamp("2023-05-01", tz="America/New_York")
test_start = pd.Timestamp("2023-06-01", tz="America/New_York")
target_time = table["target_hour_utc"]
train = table[target_time < validation_start].copy()
validation = table[(target_time >= validation_start) & (target_time < test_start)].copy()
test = table[target_time >= test_start].copy()

print({"train": len(train), "validation": len(validation), "test": len(test)})
```

**Expect:** `{'train': 32532, 'validation': 8928, 'test': 8640}`, the split from Demo 3.

## 2. One pipeline and one baseline, judged on validation

The features mix types, so a `ColumnTransformer` (Lecture 10) sends the zone ID through a categorical branch and the other nine features through a numeric branch:

- **Zone:** `OneHotEncoder(handle_unknown="ignore")` gives each zone its own 0/1 column, so zone 239 is not treated as larger than zone 132; a zone the model never saw would get all zeros instead of an error. `sparse_output=False` returns an ordinary table, which is fine at this size.
- **Numbers:** `StandardScaler` puts the features on one scale, which Ridge's penalty needs.
- **Imputers:** each branch first fills missing values from the training rows. This table has none, so they change nothing here; Assignment 11's sensor data does have gaps.

Every learned value (the zone list, the medians, and the scaling statistics) comes from the rows passed to `fit`, here the training rows, and is reused unchanged on validation and test. `Ridge(alpha=10.0)` is linear regression with a penalty that keeps coefficients small. It accepts `random_state`, which only its `sag` and `saga` solvers use, so setting it changes nothing here but records the seed, as the final exam asks. A pickup count cannot be negative, so `np.maximum(values, 0)` (Lecture 03) raises any negative prediction to 0.

```python
CATEGORICAL = ["pickup_zone_id"]
NUMERIC = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "pickup_count"

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("zone", categorical_pipeline, CATEGORICAL),
    ("numeric", numeric_pipeline, NUMERIC),
])
ridge_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", Ridge(alpha=10.0, random_state=217)),
])

def score(split, candidate, y_true, y_pred):
    return {
        "split": split,
        "candidate": candidate,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

ridge_pipeline.fit(train[FEATURES], train[TARGET])
validation_predictions = {
    "lag_168_baseline": np.maximum(validation["lag_168"], 0),
    "ridge_pipeline": np.maximum(ridge_pipeline.predict(validation[FEATURES]), 0),
}
validation_scores = pd.DataFrame([
    score("validation", name, validation[TARGET], predictions)
    for name, predictions in validation_predictions.items()
]).sort_values("MAE").reset_index(drop=True)
display(validation_scores.round(1))
```

**Expect:** two rows, sorted by MAE: `ridge_pipeline` with MAE 25.0 and RMSE 36.5, then `lag_168_baseline` with MAE 29.4 and RMSE 46.4. On May's hours the pipeline misses by about 25 pickups per zone-hour on average, against about 29 for "same hour last week".

## 3. Freeze the choice, then evaluate test once

The next line is the only selection step: the candidate with the lower validation MAE. After it, validation has done its job. The pipeline is refit on training plus validation rows with the same settings, and both candidates are scored on June once; nothing afterwards goes back to change the choice.

```python
selected_candidate = validation_scores.loc[0, "candidate"]
print("Frozen candidate:", selected_candidate)

development = pd.concat([train, validation], ignore_index=True)
ridge_pipeline.fit(development[FEATURES], development[TARGET])
test_predictions = {
    "lag_168_baseline": np.maximum(test["lag_168"], 0),
    "ridge_pipeline": np.maximum(ridge_pipeline.predict(test[FEATURES]), 0),
}
test_scores = pd.DataFrame([
    score("test", name, test[TARGET], predictions)
    for name, predictions in test_predictions.items()
])

metrics_table = pd.concat([validation_scores, test_scores], ignore_index=True)
metrics_table["selected"] = metrics_table["candidate"] == selected_candidate
display(metrics_table.round(1))
```

**Expect:** `Frozen candidate: ridge_pipeline`, then the four rows of Lecture 11's results table: validation 25.0 and 36.5 for the pipeline and 29.4 and 46.4 for the baseline; test 32.3 and 50.2 for the baseline and 25.6 and 37.8 for the frozen pipeline, with `selected` True on the pipeline's rows. The test result supports the candidate claim for June 2023: the pipeline beat "same hour last week" by about 7 pickups per zone-hour.

## 4. Save the predictions and look at error slices

An overall MAE can hide where the errors are. Group the frozen candidate's test errors by zone and by local hour (Lecture 08), and save them: small summaries as CSV for people, the full prediction table as Parquet for later code.

```python
predictions = test[["pickup_zone_id", "target_hour_utc", "target_hour_local", "hour_of_day", TARGET]]
predictions = predictions.rename(columns={TARGET: "actual"})
predictions["prediction"] = test_predictions[selected_candidate]
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
    result["RMSE"] = np.sqrt(result["mean_squared_error"])
    return result.drop(columns="mean_squared_error")

zone_errors = error_slice(predictions, "pickup_zone_id")
hour_errors = error_slice(predictions, "hour_of_day")
display(zone_errors.sort_values("MAE", ascending=False).round(1))
display(hour_errors.round(1))

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
predictions.to_parquet(output_dir / "04_test_predictions.parquet", index=False)
metrics_table.to_csv(output_dir / "04_metrics.csv", index=False)
zone_errors.to_csv(output_dir / "04_zone_error_summary.csv", index=False)
hour_errors.to_csv(output_dir / "04_hour_error_summary.csv", index=False)
```

**Expect:** a 12-row zone table with 720 test hours per zone, led by the two airports, zone 132 (JFK) at MAE 36.5 and zone 138 (LaGuardia) at 35.5, and ending with zones 170 and 239 at about 17.5; and a 24-row hour table whose MAE is smallest overnight (5.2 at 04:00) and largest in the evening (40.8 at 18:00). The airports have the largest misses even though LaGuardia averages fewer pickups than several Manhattan zones, which makes them the first place to look for a missing feature.

## 5. Make two report figures

Two ordinary figures (Lecture 07) answer the common report questions: do the predictions track the actual counts over time, and at which hours are the misses largest? Turning the date labels with `tick_params` keeps them from running into each other.

```python
hourly = predictions.groupby("target_hour_utc", as_index=False)[["actual", "prediction"]].sum()
figure, axis = plt.subplots(figsize=(11, 4))
axis.plot(hourly["target_hour_utc"], hourly["actual"], label="Actual", linewidth=1)
axis.plot(hourly["target_hour_utc"], hourly["prediction"], label="Prediction", linewidth=1)
axis.set(title="June hourly pickups across 12 zones", xlabel="Target hour (UTC)", ylabel="Pickups")
axis.tick_params(axis="x", rotation=30)
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

**Expect:** a line chart of June's total hourly pickups, with the prediction line following the daily rises and falls of the actual line closely; and a bar chart whose bars are short overnight and tallest from late afternoon to late evening.

## Final checks

```python
assert len(predictions) == len(test)
assert predictions[["actual", "prediction", "absolute_error"]].notna().all().all()
assert len(zone_errors) == 12 and len(hour_errors) == 24
selected_test = test_scores[test_scores["candidate"] == selected_candidate].iloc[0]
assert round(selected_test["MAE"], 6) == round(predictions["absolute_error"].mean(), 6)
for filename in [
    "04_test_predictions.parquet", "04_metrics.csv", "04_zone_error_summary.csv",
    "04_hour_error_summary.csv", "04_actual_vs_predicted.png", "04_mae_by_hour.png",
]:
    assert (output_dir / filename).stat().st_size > 0

print(f"Final checks passed: {selected_candidate} test MAE={selected_test['MAE']:.3f}, "
      f"RMSE={selected_test['RMSE']:.3f}; six evidence files saved.")
```

**Expect:** `Final checks passed: ridge_pipeline test MAE=25.588, RMSE=37.801; six evidence files saved.` The `output/` folder now holds the six files, including `04_zone_error_summary.csv`, which the optional Demo 5 maps.
