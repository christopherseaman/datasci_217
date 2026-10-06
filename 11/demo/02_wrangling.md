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

The model table has one row per zone and target hour: the `pickup_count` to predict, plus features known before that hour starts. You rebuild the expected zone-hour grid and confirm that the release's panel fills it, then add local calendar fields and past-only lags and rolling means. It uses Lecture 11 up to the demo break, plus joins and the expected grid (Lecture 06), time zones, grouped shifts, and rolling windows (Lecture 09), and Parquet (Lecture 04). Assignment 11's Q3 and Q4 build the same kind of table from sensor data.

Run the cells from top to bottom; after each step, an **Expect** line says what you should see.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

This notebook reads two of Demo 1's release files: the manifest and the zone-hour panel. This cell is supplied plumbing, as in Demo 1: it keeps any file already in `data/` and downloads the rest.

```python
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

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

## 1. Load the verified panel

Check the panel against the hash the manifest records before trusting it, as Demo 1 did for every release file.

```python
manifest_path = data_dir / "demo_release_manifest.json"
published_manifest_sha256 = "558c28a8ab5a16769ac6ef9d170e7bd7f4ae4ef5d2a9e2b11fd2fb84d79b2c9d"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == published_manifest_sha256, "The manifest changed: rename it in data/ and rerun from the top"
with open(manifest_path) as file:
    manifest = json.load(file)
panel_path = data_dir / manifest["artifacts"]["panel"]["filename"]
panel_sha256 = hashlib.sha256(panel_path.read_bytes()).hexdigest()
assert panel_sha256 == manifest["artifacts"]["panel"]["sha256"], "The panel changed: rename it in data/ and rerun"

panel = pd.read_parquet(panel_path).sort_values(["target_hour_utc", "pickup_zone_id"]).reset_index(drop=True)
display(panel.head())
print("Shape:", panel.shape)
print("Pickups in the selected zones:", f"{panel['pickup_count'].sum():,}")
```

**Expect:** five rows for the first hour, `2023-01-01 05:00:00+00:00` in `target_hour_utc`, which is local midnight (`2023-01-01 00:00:00-05:00`) in `target_hour_local`; zone 132 had 258 pickups in that hour. Then `Shape: (52116, 4)` and `Pickups in the selected zones: 8,607,337`.

## 2. Confirm that the panel is complete

A **complete panel** has a row for every zone at every hour (Lecture 11). Build the expected grid the way Lecture 06's cross-join snippet does: every elapsed UTC hour from local 2023-01-01 00:00 up to local 2023-07-01 00:00, crossed with the 12 selected zones. Then left-merge the panel onto it with `indicator=True`: a zone-hour missing from the panel would come back `left_only`.

```python
start = pd.Timestamp("2023-01-01", tz="America/New_York").tz_convert("UTC")
end = pd.Timestamp("2023-07-01", tz="America/New_York").tz_convert("UTC")
hours = pd.DataFrame({"target_hour_utc": pd.date_range(start, end, freq="h", inclusive="left")})
zones = pd.DataFrame({"pickup_zone_id": manifest["top_zone_selection"]["zone_ids"]})
expected = zones.merge(hours, how="cross")

coverage = expected.merge(panel, on=["pickup_zone_id", "target_hour_utc"], how="left",
                          validate="one_to_one", indicator=True)
print("Elapsed hours in the window:", len(hours))
print("Expected zone-hours:", len(expected))
display(coverage["_merge"].value_counts())
print("Zone-hours with 0 pickups:", (panel["pickup_count"] == 0).sum())

assert (coverage["_merge"] == "both").all() and len(panel) == len(expected)
```

**Expect:** `Elapsed hours in the window: 4343`, one fewer than 181 days × 24, because the clocks sprang forward on 12 March; `Expected zone-hours: 52116`; all 52,116 rows `both`, with 0 `left_only` and 0 `right_only`; and `Zone-hours with 0 pickups: 375`.

Every expected zone-hour is present, and the panel has no extra rows, so its grain is exactly one zone at one UTC hour. The release builder treated the trip feed as complete and filled its 375 empty zone-hours with zero counts; grid completeness alone cannot prove complete event capture. Assignment 11's sensor readings are measurements, so there an hour without a reading stays missing, and `source_observed` records which rows came from the source.

## 3. Add local calendar fields and past-only history

At prediction time for hour _t_, the count for hour _t_ is not yet known, so every history feature starts with `shift` (Lecture 09): `lag_1` is the previous elapsed hour, `lag_24` is 24 elapsed hours ago, and `lag_168` is 168 elapsed hours ago, each shifted within its own zone. Across a daylight-saving change, those 24- and 168-hour lags can differ from the same local clock hour yesterday or last week. The rolling means shift by one hour before averaging, so the 24-hour mean covers hours _t−24_ through _t−1_. Calendar fields come from local New York time, because rush hour happens at local 08:00, whatever the UTC clock says. A zone's first 168 hours have no week-old lag, so those rows are dropped.

```python
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
display(model_table.head())
print("Rows after dropping incomplete history:", f"{len(model_table):,}")
```

**Expect:** the first rows are local midnight on Sunday 8 January (`hour_of_day` 0, `day_of_week` 6, `is_weekend` 1); zone 132's row has `pickup_count` 194, `lag_1` 297.0, `lag_168` 258.0 (the 258 pickups of the panel's first hour, one week earlier), and `rolling_mean_24` about 214.92. Then `Rows after dropping incomplete history: 50,100`, which is 52,116 minus 168 hours for each of the 12 zones.

## 4. Check one row by hand

A feature that silently read the target hour would make the model look better than it can be. Check the first row for zone 132 against the panel itself: `lag_1` must equal the count one hour earlier, and `rolling_mean_24` must average the 24 hours before the target, not including it.

```python
zone_panel = panel[panel["pickup_zone_id"] == 132].set_index("target_hour_utc")
example = model_table[model_table["pickup_zone_id"] == 132].iloc[0]
target_hour = example["target_hour_utc"]
one_hour = pd.Timedelta(hours=1)

previous_count = zone_panel.loc[target_hour - one_hour, "pickup_count"]
past_24 = zone_panel.loc[target_hour - 24 * one_hour: target_hour - one_hour, "pickup_count"]
print("Target hour:", target_hour)
print("lag_1:", example["lag_1"], "| panel count one hour earlier:", previous_count)
print("rolling_mean_24:", round(example["rolling_mean_24"], 3),
      "| mean of the", len(past_24), "earlier hours:", round(past_24.mean(), 3))

assert example["lag_1"] == previous_count
assert round(example["rolling_mean_24"], 6) == round(past_24.mean(), 6)
```

**Expect:** `Target hour: 2023-01-08 05:00:00+00:00`, `lag_1: 297.0 | panel count one hour earlier: 297`, and `rolling_mean_24: 214.917 | mean of the 24 earlier hours: 214.917`. Label slicing with `.loc` includes both ends, so the slice holds exactly the 24 hours from _t−24_ to _t−1_.

## 5. Save the model table

Save the table as Parquet (Lecture 04), which keeps every dtype, including the time zones, for the next step.

```python
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
model_path = output_dir / "02_model_table.parquet"
model_table.to_parquet(model_path, index=False)
print("Saved", model_path, f"({model_path.stat().st_size:,} bytes)")
print("Final checks passed: complete grain and strictly past-only features.")
```

**Expect:** `Saved output/02_model_table.parquet (631,133 bytes)`; the size can differ slightly with another pyarrow version. Then `Final checks passed: complete grain and strictly past-only features.`

Demo 3 rebuilds this table with the same `build_model_table()` function, so it runs on its own; in Colab, each notebook gets a fresh runtime and would not see this file.
