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

This demo starts Lecture 11's worked example: predict the **next-hour pickup count** for each of 12 New York taxi zones, the way a hospital would predict next-hour arrivals at each emergency department. Before any analysis, you get the course's frozen data release, check every file against the hashes its manifest records, audit a sample of trip events, and see why the sample and the hourly panel the later demos use cannot be compared row for row. It uses Lecture 11 up to the demo break, plus Parquet (Lecture 04), file hashes and cleaning rules (Lecture 05), JSON (Lecture 07), and times written as text (Lecture 09). Assignment 11's Q1 and Q2 follow the same pattern with sensor data.

## How to run

Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-30 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, and pyarrow 25.0.0.

- **In Colab:** run the install cell below first.
- **Locally:** run these commands in a terminal.

<!-- #region -->
```shell
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/setup_demo.sh | sh
cd ~/11-demo
uv venv --seed
source .venv/bin/activate
uv sync
```
<!-- #endregion -->

Then open the `11-demo` folder in VS Code and choose its `.venv` as the notebook kernel.

## Setup

```python
%pip install -q --no-warn-conflicts pandas==3.0.5
```

**Expect:** nothing, or a note to restart the kernel. If Colab asks you to restart the session, do it and rerun from the top.

```python
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

print("pandas", pd.__version__)
```

**Expect:** `pandas 3.0.5`.

## 1. Get the release files

The course froze its taxi data as a **release**: three data files and a manifest, kept in the course repository so everyone analyzes the same bytes. This cell is supplied plumbing: it keeps any file already in `data/` and downloads the rest (`urlretrieve(url, path)` saves the file at a web address to `path`).

```python
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

The **manifest** is the release's provenance record: where the data came from, how the 12 zones were chosen, and each file's SHA-256 hash and size in bytes (Lecture 05's file fingerprint). First check the manifest itself against the hash the course publishes, then read it with `json.load()` (Lecture 07) and check each data file against it.

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

The manifest also records the official source URLs, the rule that picked the 12 zones, and the versions of the tools that built the release. The Taxi and Limousine Commission (TLC) cautions that technology providers supplied the trip records and that it does not vouch for their accuracy, which is one more reason to audit before trusting.

## 3. Explore the event grain

The sample holds 5,000 trip events per month. One row is one sampled pickup event, identified by `course_row_id`; it is **not** one zone-hour, and it was drawn for audit practice, not for estimating totals.

```python
events = pd.read_parquet(data_dir / manifest["artifacts"]["sample"]["filename"])

assert len(events) == manifest["sample_rows"] == 30_000
assert events["course_row_id"].is_unique
assert not events.duplicated(subset=["source_month", "source_row_number"]).any()

display(events.head())
display(events.dtypes.rename("dtype").to_frame())
print("Event rows:", f"{len(events):,}")
print("Months represented:", events["source_month"].value_counts().sort_index().to_dict())
print("Missing values:", events.isna().sum().to_dict())
```

**Expect:** five rows of `source_month`, `source_row_number`, `course_row_id`, `pickup_datetime_local`, and `pickup_zone_id`, such as `2023-01  0  2023-01-00000000  2023-01-01 00:32:10  161`; `pickup_datetime_local` already has the `datetime64[us]` dtype, because Parquet stores dtypes; `Event rows: 30,000`; 5,000 rows in each month from `2023-01` to `2023-06`; and 0 missing values in every column.

The release selected the 12 zones with the most January pickups. The lookup table names them:

```python
zones = pd.read_csv(data_dir / manifest["artifacts"]["lookup"]["filename"])
selected_zones = manifest["top_zone_selection"]["zone_ids"]
display(zones[zones["LocationID"].isin(selected_zones)])
```

**Expect:** 12 rows, one per selected zone, such as `132  Queens  JFK Airport  Airports` and `161  Manhattan  Midtown Center  Yellow Zone`; all but the two airports, JFK and LaGuardia in Queens, are in Manhattan.

## 4. Apply deterministic cleaning rules

Parse the timestamps rather than trusting how they display, then keep the events whose timestamp falls in its stated source month and whose pickup zone is one of the 12 selected zones. `np.select()` (Lecture 03) gives each excluded row its first failed rule as a reason, so every exclusion is auditable and no row is silently changed.

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

**Expect:** two reasons, `zone outside selected set` with 16,658 rows and `keep` with 13,342, then `Retained events: 13,342`. Every timestamp parsed and matched its month, so those two rules excluded nothing here; they stay in the audit because a new release could break them.

## 5. Raw events versus a derived panel

The frozen **panel** was derived from **all** official January to June source rows, about 19.5 million of them: filtered to the selected zones, counted per zone and UTC hour, and completed with zero-count hours. The 13,342 cleaned sample events are a tiny slice of the same trips, so aggregating them can never reproduce the panel's counts.

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

**Expect:** the two-row table from Lecture 11's grain section: the sample has 13,342 rows standing for 13,342 pickups, while the panel has 52,116 zone-hour rows standing for 8,607,337 pickups. The last line reads `Final checks passed: release, event grain, cleaning audit, and grain distinction.`

Demo 2 starts from the full panel, not from this sample. Each notebook runs on its own, so you can close this one first.
