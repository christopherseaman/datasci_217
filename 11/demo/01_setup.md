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

# Demo 1: Trust the release before using it

The worked example predicts the **next-hour pickup count** for each of 12 New York taxi zones, the way a hospital would predict next-hour arrivals at each emergency department. This demo checks the data first:

- Download the frozen data release and check every file against its manifest hashes.
- Audit a sample of trip events with deterministic cleaning rules.
- See why the sample and the hourly panel used in later demos cannot be compared row for row.

Run the cells from top to bottom; an **Expect** line after each step says what you should see.

```python
# Installs the course's pandas in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5
```

## 1. Get the release files

A **release** is a frozen set of data files plus a manifest, so everyone analyzes the same bytes. This supplied cell keeps any file already in `data/` and downloads the rest (`urlretrieve(url, path)` saves the file at a web address to `path`).

```python
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/data"
FILENAMES = [
    "demo_release_manifest.json",
    "yellow_taxi_2023_h1_event_sample.parquet",
    "yellow_taxi_2023_h1_zone_hour_counts.parquet",
    "taxi_zone_lookup.csv",
]

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
for filename in FILENAMES:
    path = data_dir / filename
    if not path.exists():
        urlretrieve(f"{REPO_RAW}/{filename}", path)
    print(filename, path.stat().st_size, "bytes")
```

**Expect:** four lines: `demo_release_manifest.json 3585 bytes`, `yellow_taxi_2023_h1_event_sample.parquet 545541 bytes`, `yellow_taxi_2023_h1_zone_hour_counts.parquet 138733 bytes`, and `taxi_zone_lookup.csv 12331 bytes`.

## 2. Check every file against the manifest

The **manifest** is the release's provenance record: the data source, how the 12 zones were chosen, and each file's SHA-256 hash (file fingerprint) and size in bytes.

1. Check the manifest itself against the hash the course publishes.
2. Read it with `json.load()`.
3. Check each data file against it.

```python
manifest_path = data_dir / "demo_release_manifest.json"
published_manifest_sha256 = "558c28a8ab5a16769ac6ef9d170e7bd7f4ae4ef5d2a9e2b11fd2fb84d79b2c9d"
manifest_matches = hashlib.sha256(manifest_path.read_bytes()).hexdigest() == published_manifest_sha256
print("Manifest hash matches:", manifest_matches)
assert manifest_matches, "The manifest changed: rename it in data/ and rerun section 1"

with open(manifest_path) as file:
    manifest = json.load(file)

checks = []
for artifact_name, artifact in manifest["artifacts"].items():
    path = data_dir / artifact["filename"]
    checks.append({
        "artifact": artifact_name,
        "filename": artifact["filename"],
        "sha256_matches": hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"],
        "size_matches": path.stat().st_size == artifact["byte_size"],
    })
release_check = pd.DataFrame(checks)
display(release_check)

assert release_check["sha256_matches"].all() and release_check["size_matches"].all(), "A release file changed"
print("Verified release:", manifest["release_id"])
print("Source:", manifest["attribution"])
print("Official source files documented:", len(manifest["source_files"]))
```

**Expect:** `Manifest hash matches: True`, a three-row table (`sample`, `panel`, `lookup`) with `True` in both match columns, then `Verified release: yellow-taxi-2023-h1-demo-v1`, `Source: NYC Taxi and Limousine Commission Trip Record Data`, and `Official source files documented: 6`. A `False` means the file on disk is not the published one: rename that file in `data/`, keeping a copy to inspect, and rerun section 1.

The manifest also records the official source URLs, the zone-selection rule, and the tool versions. The Taxi and Limousine Commission (TLC) does not vouch for the accuracy of the provider-supplied trip records, so audit before trusting.

## 3. Explore the event grain

- The sample holds 5,000 trip events per month.
- One row is one sampled pickup event, identified by `course_row_id`; it is **not** one zone-hour.
- It was drawn for audit practice, not for estimating totals.
- The lookup table adds each zone's name; the first table joins it onto the events.

```python
events = pd.read_parquet(data_dir / manifest["artifacts"]["sample"]["filename"])
zones = pd.read_csv(data_dir / manifest["artifacts"]["lookup"]["filename"])

assert len(events) == manifest["sample_rows"] == 30_000
assert events["course_row_id"].is_unique
assert not events.duplicated(subset=["source_month", "source_row_number"]).any()

named = events.head().merge(zones, left_on="pickup_zone_id", right_on="LocationID").drop(columns="LocationID")
display(named)
display(events.dtypes.rename("dtype").to_frame())
display(pd.DataFrame({
    "rows": events["source_month"].value_counts().sort_index(),
    "missing_values": events.isna().sum().sum(),
}))
```

**Expect:**

- A five-row table: the event columns (`source_month`, `source_row_number`, `course_row_id`, `pickup_datetime_local`, `pickup_zone_id`) plus `Borough`, `Zone`, and `service_zone`, such as `2023-01  0  2023-01-00000000  2023-01-01 00:32:10  161  Manhattan  Midtown Center  Yellow Zone`.
- `pickup_datetime_local` is `datetime64[us]`, because Parquet stores dtypes.
- A six-row table, `2023-01` to `2023-06`, with 5,000 `rows` each (30,000 in all) and 0 `missing_values`.

The release selected the 12 zones with the most January pickups:

```python
selected_zones = manifest["top_zone_selection"]["zone_ids"]
display(zones[zones["LocationID"].isin(selected_zones)])
```

**Expect:** 12 rows, one per selected zone, such as `132  Queens  JFK Airport  Airports` and `161  Manhattan  Midtown Center  Yellow Zone`; all but the two Queens airports (JFK and LaGuardia) are in Manhattan.

## 4. Apply deterministic cleaning rules

Parse the timestamps rather than trusting how they display, then keep events that:

- have a timestamp inside the stated source month, and
- have a pickup zone among the 12 selected zones.

`np.select()` gives each excluded row its first failed rule as a reason, so every exclusion is auditable and no row is silently changed.

```python
audit = events.copy()
audit["pickup_datetime_local"] = pd.to_datetime(audit["pickup_datetime_local"], errors="coerce")
audit["observed_month"] = audit["pickup_datetime_local"].dt.strftime("%Y-%m")

valid_timestamp = audit["pickup_datetime_local"].notna()
month_matches_source = audit["observed_month"] == audit["source_month"]
zone_is_selected = audit["pickup_zone_id"].isin(selected_zones)

audit["exclusion_reason"] = np.select(
    [~valid_timestamp, ~month_matches_source, ~zone_is_selected],
    ["invalid timestamp", "timestamp outside source month", "zone outside selected set"],
    default="keep",
)
clean_events = audit[audit["exclusion_reason"] == "keep"].copy()

display(audit["exclusion_reason"].value_counts().rename("rows").to_frame())
print("Retained events:", f"{len(clean_events):,}")

assert clean_events["pickup_zone_id"].isin(selected_zones).all()
assert (clean_events["observed_month"] == clean_events["source_month"]).all()
assert len(clean_events) + (audit["exclusion_reason"] != "keep").sum() == len(events)
```

**Expect:** a two-row table, `zone outside selected set` with 16,658 rows and `keep` with 13,342, then `Retained events: 13,342`. The timestamp and month rules excluded nothing here; they stay because a new release could break them.

## 5. Raw events versus a derived panel

The frozen **panel** was derived from **all** official January to June rows (about 19.5 million): filtered to the selected zones, counted per zone and UTC hour, and completed with zero-count hours. The 13,342 cleaned events are a tiny slice of those trips, so aggregating them cannot reproduce the panel's counts.

```python
panel = pd.read_parquet(data_dir / manifest["artifacts"]["panel"]["filename"])
full_panel_pickups = int(panel["pickup_count"].sum())

comparison = pd.DataFrame({
    "object": ["cleaned teaching sample", "derived full panel"],
    "row_grain": ["one sampled event", "one zone-hour"],
    "rows": [len(clean_events), len(panel)],
    "pickup_total": [len(clean_events), full_panel_pickups],
})
display(comparison)

assert len(panel) == manifest["panel_rows"] == 52_116
assert full_panel_pickups == 8_607_337
print("Final checks passed: release, event grain, cleaning audit, and grain distinction.")
```

**Expect:** a two-row table: the sample has 13,342 rows standing for 13,342 pickups, while the panel has 52,116 zone-hour rows standing for 8,607,337 pickups. The last line reads `Final checks passed: release, event grain, cleaning audit, and grain distinction.`
