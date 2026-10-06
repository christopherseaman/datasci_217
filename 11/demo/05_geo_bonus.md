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

# Optional Demo 5: Map zone-level model error

This notebook is **demo-only and non-graded**: nothing in it is assumed prior knowledge, and the assignment never needs it. It maps Demo 4's test error for each of the 12 taxi zones. The course concepts are reading a results table, validating a join, and making an honest labeled figure; the geospatial machinery (the geo packages, the zone boundaries, the shapefile, and drawing polygons) is supplied.

## How to run

Run the cells from top to bottom; after each step, an **Expect** line says what you should see. The notebook reads Demo 4's `output/04_zone_error_summary.csv` when you ran Demo 4 in the same folder, and otherwise downloads a copy; the zone boundaries need an internet connection. Tested 2026-09-30 with Python 3.13, pandas 3.0.5, geopandas 1.1.1, and matplotlib 3.11.1.

- **In Colab:** run the install cell below first.
- **Locally:** a `~/11-demo` folder already set up for Demo 1 needs only the `uv add` line below and then its `.venv` chosen as the notebook kernel. Otherwise, run these commands in a terminal, then the `uv add` line.

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

The geo packages are not in `pyproject.toml`: locally, with the environment active, run `uv add geopandas==1.1.1 shapely==2.1.1 pyogrio==0.11.1` once.

## Setup

```python
%pip install -q --no-warn-conflicts pandas==3.0.5 geopandas==1.1.1 shapely==2.1.1 pyogrio==0.11.1
```

**Expect:** nothing, or a note to restart the kernel. If Colab asks you to restart the session, do it and rerun from the top.

```python
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

print("geopandas", gpd.__version__)
```

**Expect:** `geopandas 1.1.1`.

This notebook reads two files. Demo 4's zone error summary comes from `output/` when Demo 4 ran in this folder; otherwise the cell downloads the course's committed copy of it into `data/`. The zone boundaries are the Taxi and Limousine Commission's (TLC) official shapefile, a zipped set of map files that holds each taxi zone's boundary as a polygon. This cell is supplied plumbing: it keeps any file already present and downloads the rest.

```python
REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/data"
GEO_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

summary_path = Path("output/04_zone_error_summary.csv")
if not summary_path.exists():
    summary_path = Path("data/04_zone_error_summary.csv")
    if not summary_path.exists():
        summary_path.parent.mkdir(exist_ok=True)
        urlretrieve(f"{REPO_RAW}/04_zone_error_summary.csv", summary_path)

geo_dir = Path("output/geo")
archive = geo_dir / "taxi_zones.zip"
if not archive.exists():
    geo_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(GEO_URL, archive)

for path in (summary_path, archive):
    print(path, path.stat().st_size, "bytes")
```

**Expect:** `output/04_zone_error_summary.csv 1037 bytes` when Demo 4 ran in this folder, or `data/04_zone_error_summary.csv 1037 bytes` for the downloaded copy, then `output/geo/taxi_zones.zip 1022574 bytes`.

## 1. Load Demo 4's zone errors

```python
zone_errors = pd.read_csv(summary_path)
assert zone_errors["pickup_zone_id"].is_unique
assert len(zone_errors) == 12
print("Read", summary_path)
zone_errors.round(1).head()
```

**Expect:** `Read output/04_zone_error_summary.csv` (or `data/04_zone_error_summary.csv` for the downloaded copy) and the first five zones, starting with zone 132 at MAE 36.5, the same values as Demo 4's zone table.

## 2. Check the zone boundaries and join them

The supplied cell checks the shapefile archive's SHA-256 hash, as Demo 1 did for the release, then unzips it and reads it with geopandas. The join is the familiar part: a one-to-one inner merge on the zone ID (Lecture 06), which must keep all 12 zones.

```python
shapefile = geo_dir / "taxi_zones" / "taxi_zones.shp"
expected_geo_sha256 = "f6d711917bb4340f8f644d5366c51665489eb2d426dd1a4a55677721ae5adf17"
actual_geo_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual_geo_sha256 == expected_geo_sha256, "The zone archive changed: rename output/geo to preserve it, then rerun"
if not shapefile.exists():
    with ZipFile(archive) as zipped:
        zipped.extractall(geo_dir)

zones = gpd.read_file(shapefile)[["LocationID", "zone", "borough", "geometry"]]
zones["LocationID"] = zones["LocationID"].astype("int64")
mapped = zones.merge(zone_errors, left_on="LocationID", right_on="pickup_zone_id",
                     how="inner", validate="one_to_one")

assert len(mapped) == 12
assert mapped.geometry.notna().all()
mapped[["LocationID", "zone", "borough", "MAE"]].sort_values("MAE", ascending=False).round(1)
```

**Expect:** a 12-row table sorted by MAE, from `JFK Airport` (Queens, 36.5) and `LaGuardia Airport` (Queens, 35.5) down to `Upper West Side South` (Manhattan, 17.4).

## 3. Draw the choropleth

A **choropleth** colors each area by a value, here the zone's June MAE; darker red means larger misses. Only the 12 modeled zones are drawn, with no background map, so the figure needs no map-tile service.

```python
figure, axis = plt.subplots(figsize=(9, 9))
mapped.plot(
    column="MAE",
    cmap="YlOrRd",
    edgecolor="white",
    linewidth=0.7,
    legend=True,
    legend_kwds={"label": "June mean absolute error (pickups per zone-hour)"},
    ax=axis,
)
axis.set_title("Selected taxi-zone forecast error")
axis.set_axis_off()
figure.tight_layout()

map_path = Path("output/05_zone_error_choropleth.png")
figure.savefig(map_path, dpi=150, bbox_inches="tight")
plt.show()

assert map_path.stat().st_size > 0
print(f"Final check passed: mapped {len(mapped)} zones and saved {map_path}.")
```

**Expect:** a map with the ten small Manhattan zones at the upper left, shaded from pale yellow to red, and the two much larger airport zones in Queens, LaGuardia to the east and JFK to the southeast, in the darkest red; then `Final check passed: mapped 12 zones and saved output/05_zone_error_choropleth.png.`
