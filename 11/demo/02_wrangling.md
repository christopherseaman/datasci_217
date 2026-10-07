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

The model table has one row per zone and target hour: the `pickup_count` to predict, plus features known before that hour starts.

- Rebuild the expected zone-hour grid and confirm the panel fills it.
- Add local calendar fields.
- Add past-only lags and rolling means.

Run the cells from top to bottom; an **Expect** line after each step says what you should see.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

This notebook reads two of Demo 1's release files: the manifest and the zone-hour panel. This supplied cell keeps any file already in `data/` and downloads the rest.

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

Check the panel against the hash the manifest records, as Demo 1 did.

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

**Expect:**

- Five rows for the first hour: `2023-01-01 05:00:00+00:00` in `target_hour_utc`, which is local midnight (`2023-01-01 00:00:00-05:00`) in `target_hour_local`; zone 132 had 258 pickups.
- `Shape: (52116, 4)` and `Pickups in the selected zones: 8,607,337`.

## 2. Confirm that the panel is complete

A **complete panel** has a row for every zone at every hour.

1. Build the expected grid: every elapsed UTC hour from local 2023-01-01 00:00 up to local 2023-07-01 00:00, cross-joined with the 12 selected zones.
2. Left-merge the panel onto it with `indicator=True`; a zone-hour missing from the panel comes back `left_only`.

```python
start = pd.Timestamp("2023-01-01", tz="America/New_York").tz_convert("UTC")
end = pd.Timestamp("2023-07-01", tz="America/New_York").tz_convert("UTC")
hours = pd.DataFrame({"target_hour_utc": pd.date_range(start, end, freq="h", inclusive="left")})
zones = pd.DataFrame({"pickup_zone_id": manifest["top_zone_selection"]["zone_ids"]})
expected = zones.merge(hours, how="cross")

coverage = expected.merge(panel, on=["pickup_zone_id", "target_hour_utc"], how="left",
                          validate="one_to_one", indicator=True)
display(pd.DataFrame({
    "count": {
        "elapsed hours in the window": len(hours),
        "expected zone-hours": len(expected),
        "panel zone-hours with 0 pickups": int((panel["pickup_count"] == 0).sum()),
    }
}))
display(coverage["_merge"].value_counts().to_frame())

assert (coverage["_merge"] == "both").all() and len(panel) == len(expected)
```

**Expect:**

- A three-row table: `elapsed hours in the window` 4343 (one fewer than 181 days × 24, because the clocks sprang forward on 12 March), `expected zone-hours` 52116, and `panel zone-hours with 0 pickups` 375.
- A merge table with all 52,116 rows `both`, and 0 `left_only` and 0 `right_only`.

Every expected zone-hour is present and the panel has no extras, so its grain is exactly one zone at one UTC hour.

- The release builder filled its 375 empty zone-hours with zero counts, treating the trip feed as complete.
- Grid completeness alone cannot prove complete event capture.
- For sensor readings, an hour without a reading stays missing, and `source_observed` records which rows came from the source.

## 3. Add local calendar fields and past-only history

At prediction time for hour _t_, the count for hour _t_ is not yet known, so every history feature starts with `shift`:

- `lag_1`, `lag_24`, `lag_168`: the count 1, 24, and 168 elapsed hours earlier, shifted within each zone. Across a daylight-saving change these can differ from the same local clock hour yesterday or last week.
- `rolling_mean_24`, `rolling_mean_168`: shift by one hour before averaging, so the 24-hour mean covers hours _t−24_ through _t−1_.
- Calendar fields come from local New York time, because rush hour is at local 08:00 whatever the UTC clock says.
- A zone's first 168 hours have no week-old lag, so those rows are dropped.

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

**Expect:**

- The first rows are local midnight on Sunday 8 January (`hour_of_day` 0, `day_of_week` 6, `is_weekend` 1).
- Zone 132's row has `pickup_count` 194, `lag_1` 297.0, `lag_168` 258.0 (the panel's first hour, one week earlier), and `rolling_mean_24` about 214.92.
- `Rows after dropping incomplete history: 50,100`, which is 52,116 minus 168 hours for each of the 12 zones.

## 4. Check one row by hand

A feature that silently read the target hour would make the model look better than it can be. Check zone 132's first row against the panel:

- `lag_1` must equal the count one hour earlier.
- `rolling_mean_24` must average the 24 hours before the target, not including it.

```python
zone_panel = panel[panel["pickup_zone_id"] == 132].set_index("target_hour_utc")
example = model_table[model_table["pickup_zone_id"] == 132].iloc[0]
target_hour = example["target_hour_utc"]
one_hour = pd.Timedelta(hours=1)

previous_count = zone_panel.loc[target_hour - one_hour, "pickup_count"]
past_24 = zone_panel.loc[target_hour - 24 * one_hour: target_hour - one_hour, "pickup_count"]
print("Target hour:", target_hour)
display(pd.DataFrame({
    "model table": [example["lag_1"], round(example["rolling_mean_24"], 3)],
    "recomputed from panel": [previous_count, round(past_24.mean(), 3)],
}, index=["lag_1", "rolling_mean_24"]))

assert example["lag_1"] == previous_count
assert round(example["rolling_mean_24"], 6) == round(past_24.mean(), 6)
```

**Expect:** `Target hour: 2023-01-08 05:00:00+00:00` and a two-row table with matching columns: `lag_1` 297.000 in both columns, `rolling_mean_24` 214.917 in both columns. Label slicing with `.loc` includes both ends, so the slice holds exactly the 24 hours from _t−24_ to _t−1_.

## 5. Save the model table

Parquet keeps every dtype, including the time zones.

```python
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
model_path = output_dir / "02_model_table.parquet"
model_table.to_parquet(model_path, index=False)
print("Saved", model_path, f"({model_path.stat().st_size:,} bytes)")
print("Final checks passed: complete grain and strictly past-only features.")
```

**Expect:** `Saved output/02_model_table.parquet (631,133 bytes)`; the size can differ slightly with another pyarrow version. Then `Final checks passed: complete grain and strictly past-only features.`
