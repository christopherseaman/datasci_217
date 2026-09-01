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

**Assignment patterns:** Q5 pattern analysis and Q6 modeling preparation.

We will explore patterns using training rows only. The fixed local-time split is:

- training: before May 2023
- validation: May 2023
- test: June 2023

The test partition stays unopened here. Demo 4 will use validation to freeze the
model choice, then evaluate test exactly once.

## One setup cell

```python
import importlib.metadata as metadata
import importlib.util
import subprocess
import sys

REQUIRED = {
    "numpy": "2.0.2", "pandas": "3.0.5", "pyarrow": "25.0.0",
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

import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from IPython.display import display

print(f"Python {sys.version.split()[0]} | pandas {pd.__version__}")
```

## Rebuild the prerequisite deterministically

If Demo 2's compact Parquet output exists locally, we load it. Otherwise, fresh
Colab acquires the released panel and applies the same deterministic transformation.
Release URLs use `main` for the published course materials.

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
    expected_manifest_sha256 = "553a1d732c0e0bdee9b8d79d7262a3f361109c23af6c33776f79ae661bca5fc6"
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

authenticated_panel = acquire_authenticated_panel()
model_table = build_model_table(authenticated_panel)

assert len(model_table) == 50_100
print("Model table source: rebuilt from authenticated frozen panel")
print("Model table rows:", f"{len(model_table):,}")
```

## Split by target local time

UTC remains the unique key. Local timestamps express the project policy clearly:
May is validation and June is test.

```python
local_naive = model_table["target_hour_local"].dt.tz_localize(None)
train_mask = local_naive.lt(pd.Timestamp("2023-05-01"))
validation_mask = local_naive.between(
    pd.Timestamp("2023-05-01"), pd.Timestamp("2023-06-01"), inclusive="left"
)
test_mask = local_naive.ge(pd.Timestamp("2023-06-01"))

train = model_table.loc[train_mask].copy()
validation = model_table.loc[validation_mask].copy()
test = model_table.loc[test_mask].copy()

assert (train_mask.astype(int) + validation_mask.astype(int) + test_mask.astype(int)).eq(1).all()
assert train["target_hour_local"].max() < validation["target_hour_local"].min()
assert validation["target_hour_local"].max() < test["target_hour_local"].min()

split_summary = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "rows": [len(train), len(validation), len(test)],
    "first_local": [x["target_hour_local"].min().isoformat() for x in (train, validation, test)],
    "last_local": [x["target_hour_local"].max().isoformat() for x in (train, validation, test)],
})
display(split_summary)
```

## Look for patterns without peeking at test

These summaries use `train` only. Hour-of-day behavior supports calendar features;
zone variation supports treating zone ID as a category rather than a number.

```python
train_hour_pattern = train.groupby("hour_of_day", as_index=False).agg(
    mean_pickups=("pickup_count", "mean"),
    median_pickups=("pickup_count", "median"),
)
train_zone_pattern = train.groupby("pickup_zone_id", as_index=False).agg(
    mean_pickups=("pickup_count", "mean"),
    total_pickups=("pickup_count", "sum"),
)

display(train_hour_pattern)
display(train_zone_pattern.sort_values("mean_pickups", ascending=False))
print("Pattern-analysis rows used:", f"{len(train):,} train, 0 validation, 0 test")
```

## Check feature availability and leakage

At the start of target hour *t*, zone and calendar fields are known. Lag and rolling
features use timestamps before *t*. The current `pickup_count` is the target and
must never appear in `X`.

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
assert "target_hour_local" not in FEATURES and "target_hour_utc" not in FEATURES
assert model_table[FEATURES].notna().all().all()
assert set(model_table.columns).isdisjoint({"fare_amount", "trip_distance", "tip_amount"})
```

## Save a compact split manifest

This JSON records policy and boundaries, not large duplicate CSV partitions.

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
manifest_path.write_text(json.dumps(split_manifest, indent=2) + "\n")

assert sum(split_manifest["rows"].values()) == len(model_table)
assert manifest_path.exists()
print(f"Saved {manifest_path}")
print("Final checks passed: training-only analysis, fixed split, and leakage audit.")
```
