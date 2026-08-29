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

This notebook is **demo-only and non-graded**. It is not assumed prior knowledge
and is never required by the assignment or grader. Run Demo 4 first so its zone
error summary exists.

The course concepts are reading a results table, validating a join, and making an
honest labeled figure. Package installation, geometry download, shapefile reading,
coordinate handling, and polygon rendering are supplied geospatial machinery.

## Separate optional setup

These geo packages are intentionally absent from core requirements.

```python
import importlib.metadata as metadata
import importlib.util
import subprocess
import sys

REQUIRED = {
    "numpy": "2.0.2",
    "pandas": "3.0.5",
    "matplotlib": "3.11.1",
    "geopandas": "1.1.1",
    "shapely": "2.1.1",
    "pyogrio": "0.11.1",
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

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import hashlib
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

print(f"geopandas {gpd.__version__}")
```

## Load Demo 4 evidence

```python
summary_candidates = [
    Path("output/04_zone_error_summary.csv"),
    Path("11/demo/output/04_zone_error_summary.csv"),
]
summary_path = next((path for path in summary_candidates if path.exists()), None)
if summary_path is None:
    raise FileNotFoundError(
        "Run 04_modeling.ipynb first; it creates output/04_zone_error_summary.csv"
    )

zone_errors = pd.read_csv(summary_path)
assert zone_errors["pickup_zone_id"].is_unique
assert len(zone_errors) == 12
zone_errors.head()
```

## Download official TLC polygons

The geometry is official TLC taxi-zone data. Polygon-only rendering succeeds
without a tile service. An OSM basemap is an optional enhancement, not a dependency.

```python
GEO_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
geo_dir = Path("output/geo")
geo_dir.mkdir(parents=True, exist_ok=True)
archive = geo_dir / "taxi_zones.zip"
shapefile = geo_dir / "taxi_zones" / "taxi_zones.shp"

if not archive.exists():
    urlretrieve(GEO_URL, archive)

expected_geo_sha256 = "f6d711917bb4340f8f644d5366c51665489eb2d426dd1a4a55677721ae5adf17"
actual_geo_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
if actual_geo_sha256 != expected_geo_sha256:
    raise ValueError(
        f"Taxi-zone archive hash mismatch: expected {expected_geo_sha256}, got {actual_geo_sha256}"
    )

if not shapefile.exists():
    with ZipFile(archive) as zipped:
        zipped.extractall(geo_dir)

zones = gpd.read_file(shapefile)[["LocationID", "zone", "borough", "geometry"]]
zones["LocationID"] = zones["LocationID"].astype("int64")
mapped = zones.merge(
    zone_errors,
    left_on="LocationID",
    right_on="pickup_zone_id",
    how="inner",
    validate="one_to_one",
)

assert len(mapped) == 12
assert mapped.geometry.notna().all()
print("Authenticated official taxi_zones.zip SHA-256 digest.")
mapped[["LocationID", "zone", "borough", "MAE"]].sort_values("MAE", ascending=False)
```

## Render the choropleth

```python
figure, axis = plt.subplots(figsize=(9, 9))
mapped.plot(
    column="MAE",
    cmap="YlOrRd",
    edgecolor="white",
    linewidth=0.7,
    legend=True,
    legend_kwds={"label": "June mean absolute error"},
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
