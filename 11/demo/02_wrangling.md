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

# Demo 2: Build a past-only model table

**Assignment patterns:** Q3 wrangling and Q4 feature engineering.

The target is `pickup_count` at one `(pickup_zone_id, target_hour_utc)`. We verify
that grain before making calendar, lag, and rolling features. This notebook is
independent: in fresh Colab it acquires the committed panel directly.

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

import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from IPython.display import display

print(f"Python {sys.version.split()[0]} | pandas {pd.__version__}")
```

## Load the frozen full panel

Development URLs use `main`; immutable annual-tag replacement is pending release
freeze.

```python
REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/data"
FILES = ["demo_release_manifest.json", "yellow_taxi_2023_h1_zone_hour_counts.parquet"]

def data_directory():
    for candidate in (Path("data"), Path("11/demo/data")):
        if (candidate / FILES[0]).exists():
            return candidate
    destination = Path("data")
    destination.mkdir(exist_ok=True)
    for filename in FILES:
        path = destination / filename
        if not path.exists():
            urlretrieve(f"{REPO_RAW}/{filename}", path)
    return destination

data_dir = data_directory()
manifest_path = data_dir / FILES[0]
panel_path = data_dir / FILES[1]
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

panel = pd.read_parquet(panel_path).sort_values(
    ["target_hour_utc", "pickup_zone_id"]
).reset_index(drop=True)

display(panel.head())
print("Shape:", panel.shape)
print("Full-panel pickups in selected zones:", f"{panel['pickup_count'].sum():,}")
print("Authenticated manifest and panel SHA-256 digests.")
```

## Verify key, completeness, and conservation facts

A complete panel has every selected zone at every UTC hour. UTC is the ordering
key because it remains unique through the spring daylight-saving transition;
local time is for interpretable calendar features.

```python
zone_ids = manifest["top_zone_selection"]["zone_ids"]
hours = pd.date_range(
    panel["target_hour_utc"].min(), panel["target_hour_utc"].max(), freq="h"
)
expected_keys = pd.MultiIndex.from_product(
    [hours, zone_ids], names=["target_hour_utc", "pickup_zone_id"]
)
observed_keys = pd.MultiIndex.from_frame(panel[["target_hour_utc", "pickup_zone_id"]])

assert len(panel) == manifest["panel_rows"] == len(expected_keys) == 52_116
assert not observed_keys.has_duplicates
assert len(expected_keys.difference(observed_keys)) == 0
assert panel["pickup_count"].ge(0).all()
assert panel["pickup_count"].sum() == panel.groupby("pickup_zone_id")["pickup_count"].sum().sum()
assert sum(item["row_count"] for item in manifest["source_files"]) == 19_493_620

facts = pd.Series({
    "official source rows documented": sum(x["row_count"] for x in manifest["source_files"]),
    "panel hours": len(hours),
    "selected zones": len(zone_ids),
    "complete zone-hours": len(panel),
    "selected-zone pickups": int(panel["pickup_count"].sum()),
})
display(facts.rename("value").to_frame())
```

The official source-row count and selected-zone pickup count describe different
grains, so they should not be equal. The conservation check we can make here is
that the panel total equals the sum of its zone totals without losing or duplicating
zone-hours.

## Add local calendar fields and past-only history

At prediction time for hour *t*, counts from hour *t* are not available. Every
history feature therefore starts with `shift`, including rolling means.

```python
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

    history_columns = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168"]
    return table.dropna(subset=history_columns).sort_values(
        ["target_hour_utc", "pickup_zone_id"]
    ).reset_index(drop=True)

model_table = build_model_table(panel)
display(model_table.head())
print("Rows after dropping incomplete history:", f"{len(model_table):,}")
```

We can inspect one row directly: `lag_1` must equal that zone's previous UTC
hour, while `rolling_mean_24` must average hours *t-24* through *t-1*.

```python
zone = zone_ids[0]
zone_panel = panel.loc[panel["pickup_zone_id"].eq(zone)].set_index("target_hour_utc")
example = model_table.loc[model_table["pickup_zone_id"].eq(zone)].iloc[0]
target_hour = example["target_hour_utc"]

assert example["lag_1"] == zone_panel.loc[target_hour - pd.Timedelta(hours=1), "pickup_count"]
expected_mean = zone_panel.loc[
    target_hour - pd.Timedelta(hours=24): target_hour - pd.Timedelta(hours=1),
    "pickup_count",
].mean()
assert np.isclose(example["rolling_mean_24"], expected_mean)
assert len(model_table) == len(panel) - 168 * len(zone_ids) == 50_100

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
model_path = output_dir / "02_model_table.parquet"
model_table.to_parquet(model_path, index=False)

print(f"Saved {model_path} ({model_path.stat().st_size:,} bytes)")
print("Final checks passed: complete grain and strictly past-only features.")
```
