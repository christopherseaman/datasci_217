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

# Demo 3: Analyze training patterns and freeze the split

Before any model, fix which target hours each part of the data may be used for, then explore patterns in the training rows only. The split follows the target's local time: training before May 2023, validation in May, and test in June. The test rows stay unopened here; Demo 4 uses validation to choose a model and opens test exactly once. It uses Lecture 11 up to the demo break, plus aggregation (Lecture 08), zoned split boundaries and leakage (Lecture 10), and JSON (Lecture 07). Assignment 11's Q5 and Q6 follow the same pattern with sensor data.

**How to run:** in Colab, open this notebook from the lecture page's Colab link. Locally, open the `11-demo` folder from Demo 1 in VS Code, open `03_model_prep.ipynb`, and select the `.venv` kernel if VS Code does not show it; without that folder, follow Demo 1's "How to run locally" first. This notebook downloads the panel and rebuilds Demo 2's model table itself, so it does not need Demo 2's output. Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Tested 2026-09-30 with Python 3.13, pandas 3.0.5, and pyarrow 25.0.0.

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

import pandas as pd

print("pandas", pd.__version__)
```

**Expect:** `pandas 3.0.5`.

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

## 1. Rebuild the model table

This supplied cell repeats Demo 2's first step and its `build_model_table()` function unchanged: verify the panel against the manifest, then add the calendar and past-only history features.

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

model_table = build_model_table(panel)
print("Model table rows:", f"{len(model_table):,}")
```

**Expect:** `Model table rows: 50,100`, the same table Demo 2 saved.

## 2. Split by target time

The boundaries are local midnights in New York, written as aware timestamps (Lecture 10's "Splitting on Target Time" card). The target times stay in UTC, the table's key, and pandas compares the instants, so nothing needs converting: local midnight on 1 May is 04:00 UTC.

```python
validation_start = pd.Timestamp("2023-05-01", tz="America/New_York")
test_start = pd.Timestamp("2023-06-01", tz="America/New_York")
target_time = model_table["target_hour_utc"]

train = model_table[target_time < validation_start].copy()
validation = model_table[(target_time >= validation_start) & (target_time < test_start)].copy()
test = model_table[target_time >= test_start].copy()

split_summary = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "rows": [len(train), len(validation), len(test)],
    "first_local": [part["target_hour_local"].min() for part in (train, validation, test)],
    "last_local": [part["target_hour_local"].max() for part in (train, validation, test)],
})
display(split_summary)

assert len(train) + len(validation) + len(test) == len(model_table)
assert train["target_hour_utc"].max() < validation["target_hour_utc"].min()
assert validation["target_hour_utc"].max() < test["target_hour_utc"].min()
```

**Expect:** the split table from Lecture 11: `train` 32,532 rows from `2023-01-08 00:00:00-05:00` to `2023-04-30 23:00:00-04:00`, `validation` 8,928 rows for May, and `test` 8,640 rows for June, each starting at local midnight. Every row lands in exactly one split, and the three periods do not overlap.

## 3. Look for patterns without peeking at test

These summaries use `train` only (Lecture 08's named aggregation). Choosing features from validation or test rows would leak those periods into decisions they are meant to judge.

```python
train_hour_pattern = train.groupby("hour_of_day", as_index=False).agg(
    mean_pickups=("pickup_count", "mean"),
    median_pickups=("pickup_count", "median"),
)
train_zone_pattern = train.groupby("pickup_zone_id", as_index=False).agg(
    mean_pickups=("pickup_count", "mean"),
    total_pickups=("pickup_count", "sum"),
)

display(train_hour_pattern.round(1))
display(train_zone_pattern.sort_values("mean_pickups", ascending=False).round(1))
print("Pattern-analysis rows used:", f"{len(train):,} train, 0 validation, 0 test")
```

**Expect:** a 24-row hour table whose mean falls from 75.1 pickups per zone-hour at local hour 0 to about 10.0 at 04:00, then climbs to its peak of 293.2 at 18:00; and a 12-row zone table led by zone 132 (JFK Airport) at 219.4 and ending with zone 239 at 124.6. The strong daily cycle is why hour of day is a feature, and the gap between zones is why each zone gets its own category rather than being treated as a number.

## 4. Check feature availability and leakage

At the start of target hour _t_, the zone and the calendar fields are known, and every lag and rolling feature reads only hours before _t_. This teaching forecast assumes each completed hour's count is available immediately; a real reporting delay would require older lags and a label-availability check (Lecture 10). The current `pickup_count` is the target, so it must never appear among the features.

```python
FEATURES = [
    "pickup_zone_id", "hour_of_day", "day_of_week", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168",
]
TARGET = "pickup_count"

availability = pd.DataFrame({
    "feature": FEATURES,
    "available_before_target_hour": True,
    "reason": [
        "known location", "calendar", "calendar", "calendar", "calendar",
        "past count", "past count", "past count", "past-only window", "past-only window",
    ],
})
display(availability)

assert TARGET not in FEATURES
assert "target_hour_utc" not in FEATURES and "target_hour_local" not in FEATURES
assert model_table[FEATURES].notna().all().all()
```

**Expect:** a 10-row table with `True` in every `available_before_target_hour` cell, and no error: the target and the raw timestamps are not features, and no feature is missing in any row.

## 5. Save a compact split manifest

A **split manifest** records the policy (grain, target, metrics, boundaries, and features) and the resulting row counts, rather than saving three large copies of the rows. `json.dump()` writes it as readable JSON (Lecture 07).

```python
split_manifest = {
    "grain": ["pickup_zone_id", "target_hour_utc"],
    "target": TARGET,
    "primary_metric": "MAE",
    "secondary_metric": "RMSE",
    "timezone": "America/New_York",
    "boundaries": {"validation_start": "2023-05-01", "test_start": "2023-06-01"},
    "features": FEATURES,
    "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
}
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
manifest_path = output_dir / "03_split_manifest.json"
with open(manifest_path, "w") as file:
    json.dump(split_manifest, file, indent=2)

assert sum(split_manifest["rows"].values()) == len(model_table)
print("Saved", manifest_path)
print(json.dumps(split_manifest["rows"]))
print("Final checks passed: training-only analysis, fixed split, and leakage audit.")
```

**Expect:** `Saved output/03_split_manifest.json`, then `{"train": 32532, "validation": 8928, "test": 8640}` and `Final checks passed: training-only analysis, fixed split, and leakage audit.` Open the JSON file to see the whole policy.

Demo 4 uses these boundaries and features to compare a baseline with one model.
