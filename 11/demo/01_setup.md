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

**Assignment patterns:** Q1 setup/exploration and Q2 cleaning.

Our project contract is precise: predict the **next-hour pickup count** for one
`(pickup_zone_id, target_hour_utc)` pair. This notebook starts one level earlier,
with sampled trip events, so we can practice auditing raw-ish records before using
the derived hourly panel.

The download URLs below use the public `main` branch for the released course
materials.

## One setup cell

This is the only environment setup cell. It installs a package only when its exact
required version is missing, then performs all imports.

```python
import importlib.metadata as metadata
import importlib.util
import subprocess
import sys

REQUIRED = {
    "numpy": "2.0.2",
    "pandas": "3.0.5",
    "pyarrow": "25.0.0",
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

print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")
```

## Acquire and verify the frozen release

Local Jupyter finds the committed files in `data/`. A fresh Colab runtime downloads
the same files from raw GitHub URLs. No Drive mount or manual upload is needed.

```python
REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/data"
FILENAMES = [
    "demo_release_manifest.json",
    "yellow_taxi_2023_h1_event_sample.parquet",
    "yellow_taxi_2023_h1_zone_hour_counts.parquet",
    "taxi_zone_lookup.csv",
]

def data_directory():
    for candidate in (Path("data"), Path("11/demo/data")):
        if (candidate / "demo_release_manifest.json").exists():
            return candidate
    destination = Path("data")
    destination.mkdir(exist_ok=True)
    for filename in FILENAMES:
        path = destination / filename
        if not path.exists():
            urlretrieve(f"{REPO_RAW}/{filename}", path)
    return destination

data_dir = data_directory()
manifest_path = data_dir / "demo_release_manifest.json"
expected_manifest_sha256 = "553a1d732c0e0bdee9b8d79d7262a3f361109c23af6c33776f79ae661bca5fc6"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected_manifest_sha256, (
    "Manifest hash mismatch"
)
manifest = json.loads(manifest_path.read_text())

for artifact in manifest["artifacts"].values():
    path = data_dir / artifact["filename"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == artifact["sha256"], f"Hash mismatch: {path}"
    assert path.stat().st_size == artifact["byte_size"]

assert manifest["artifacts"]["sample"]["sha256"] == (
    "750bcc85f0267f9189dc9842ef44827168c384d4a7e5a8678e9a996348fc4b7d"
)

print(f"Verified release: {manifest['release_id']}")
print(f"Source: {manifest['attribution']}")
print(f"Official source files documented: {len(manifest['source_files'])}")
```

The manifest is evidence, not decoration. It records official source URLs and
hashes, the deterministic top-zone rule, artifact hashes, and builder versions.
The TLC also cautions that its technology providers supplied the trip records and
that TLC does not represent their accuracy.

## Explore the event grain

One row is one sampled source event, identified by `course_row_id`. It is **not**
one zone-hour and it is not a random sample intended for estimating totals.

```python
events = pd.read_parquet(data_dir / manifest["artifacts"]["sample"]["filename"])

assert len(events) == manifest["sample_rows"] == 30_000
assert events["course_row_id"].is_unique
assert events[["source_month", "source_row_number"]].duplicated().sum() == 0

display(events.head())
display(events.dtypes.rename("dtype").to_frame())
print("Event rows:", f"{len(events):,}")
print("Months represented:", events["source_month"].value_counts().sort_index().to_dict())
print("Missing values:", events.isna().sum().to_dict())
```

## Apply deterministic cleaning rules

We parse timestamps rather than trusting a display format. We then retain events
whose timestamp belongs to its stated source month and whose pickup zone is one of
the 12 zones selected by the release rule. Every exclusion receives an auditable
reason; no row is silently changed.

```python
audit = events.copy()
audit["pickup_datetime_local"] = pd.to_datetime(
    audit["pickup_datetime_local"], errors="coerce"
)
audit["expected_month"] = audit["source_month"].astype("string")
audit["observed_month"] = audit["pickup_datetime_local"].dt.strftime("%Y-%m")

selected_zones = set(manifest["top_zone_selection"]["zone_ids"])
valid_timestamp = audit["pickup_datetime_local"].notna()
month_matches_source = audit["observed_month"].eq(audit["expected_month"])
zone_is_selected = audit["pickup_zone_id"].isin(selected_zones)

audit["exclusion_reason"] = np.select(
    [~valid_timestamp, ~month_matches_source, ~zone_is_selected],
    ["invalid timestamp", "timestamp outside source month", "zone outside selected set"],
    default="keep",
)
clean_events = audit.loc[audit["exclusion_reason"].eq("keep")].copy()

display(audit["exclusion_reason"].value_counts().rename("rows").to_frame())
print("Retained events:", f"{len(clean_events):,}")

assert clean_events["pickup_zone_id"].isin(selected_zones).all()
assert clean_events["observed_month"].eq(clean_events["expected_month"]).all()
assert len(clean_events) + audit["exclusion_reason"].ne("keep").sum() == len(events)
```

## Raw events versus a derived panel

The frozen panel was derived from **all official January-June source rows**, then
filtered to the selected zones and completed with zero-count hours. The 30,000-row
event sample exists only for audit and cleaning practice.

**Do not aggregate this sample and expect to reproduce the full panel counts.**
Sampling throws away events, while panel construction also applies documented
selection, hourly aggregation, and zero-filling rules.

```python
panel = pd.read_parquet(data_dir / manifest["artifacts"]["panel"]["filename"])
sample_selected_events = len(clean_events)
full_panel_pickups = int(panel["pickup_count"].sum())

comparison = pd.DataFrame({
    "object": ["cleaned teaching sample", "derived full panel"],
    "row_grain": ["one sampled event", "one zone-hour"],
    "rows": [len(clean_events), len(panel)],
    "pickup_total": [sample_selected_events, full_panel_pickups],
})
display(comparison)

assert len(panel) == manifest["panel_rows"] == 52_116
assert full_panel_pickups == 8_607_337
assert sample_selected_events != full_panel_pickups
print("Final checks passed: release, event grain, cleaning audit, and grain distinction.")
```

The next demo begins with the frozen full panel, not with this event sample.
