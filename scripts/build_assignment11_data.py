# /// script
# requires-python = "==3.12.13"
# dependencies = ["pandas==3.0.5"]
# ///
"""Build the frozen Chicago Beach Weather release for Assignment 11."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "11" / "assignment" / "data"
SOURCE_URL = (
    "https://data.cityofchicago.org/api/views/"
    "k7hf-8y75/rows.csv?accessType=DOWNLOAD"
)
SOURCE_SHA256 = "0cf9912f0b5782022faef9da275c1099f0e90f46931eb9dc0d60e568202a3de8"
RELEASE_FILENAME = "chicago_beach_sensors_2022_2024.csv"
MANIFEST_FILENAME = "release_manifest.json"
PUBLISHED_AT_UTC = "2026-07-22T21:00:00Z"
STATIONS = ["Foster Weather Station", "Oak Street Weather Station"]
SOURCE_COLUMNS = {
    "Station Name": "station_name",
    "Measurement Timestamp": "measurement_timestamp",
    "Air Temperature": "air_temperature_c",
    "Wet Bulb Temperature": "wet_bulb_temperature_c",
    "Humidity": "relative_humidity_pct",
    "Rain Intensity": "rain_intensity_mm_per_hour",
    "Interval Rain": "interval_rain_mm",
    "Total Rain": "total_rain_mm",
    "Precipitation Type": "precipitation_type_code",
    "Wind Direction": "wind_direction_deg",
    "Wind Speed": "wind_speed_mps",
    "Maximum Wind Speed": "maximum_wind_speed_mps",
    "Barometric Pressure": "barometric_pressure_hpa",
    "Solar Radiation": "solar_radiation_w_m2",
    "Battery Life": "battery_voltage_v",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path) -> None:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "datasci-217-release-builder"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise RuntimeError(f"download failed with HTTP {response.status}")
        destination.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    release_path = output_dir / RELEASE_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    if release_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Assignment 11 release already exists; use an empty output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ds217-a11-") as temporary_directory:
        source_path = args.source_path.resolve() if args.source_path else Path(temporary_directory) / "source.csv"
        if not source_path.exists():
            print(f"Downloading {SOURCE_URL}")
            download(source_path)
        source_hash = sha256(source_path)
        if source_hash != SOURCE_SHA256:
            raise RuntimeError(f"source hash changed: {source_hash}")

        source = pd.read_csv(source_path, usecols=list(SOURCE_COLUMNS))
        timestamp = pd.to_datetime(
            source["Measurement Timestamp"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )
        keep = (
            source["Station Name"].isin(STATIONS)
            & timestamp.ge("2022-01-01 00:00:00")
            & timestamp.lt("2025-01-01 00:00:00")
        )
        release = source.loc[keep, list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS).copy()
        release["measurement_timestamp"] = timestamp.loc[keep]
        release = release.sort_values(
            ["measurement_timestamp", "station_name"],
            kind="stable",
        ).reset_index(drop=True)
        if release.duplicated(["station_name", "measurement_timestamp"]).any():
            raise RuntimeError("release cohort has duplicate station-hour keys")
        release.to_csv(
            release_path,
            index=False,
            lineterminator="\n",
            date_format="%Y-%m-%d %H:%M:%S",
        )

    manifest = {
        "schema": "datasci217/assignment11-release/v1",
        "release_id": "chicago-beach-weather-2022-2024-v1",
        "release_filename": RELEASE_FILENAME,
        "release_sha256": sha256(release_path),
        "release_byte_size": release_path.stat().st_size,
        "row_count": len(release),
        "column_count": len(release.columns),
        "columns": release.columns.tolist(),
        "stations": STATIONS,
        "first_timestamp": release["measurement_timestamp"].min().isoformat(sep=" "),
        "last_timestamp": release["measurement_timestamp"].max().isoformat(sep=" "),
        "source_timezone": "America/Chicago",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_row_count": 206_793,
        "source_page": (
            "https://data.cityofchicago.org/Parks-Recreation/"
            "Beach-Weather-Stations-Automated-Sensors/k7hf-8y75/about_data"
        ),
        "attribution": "City of Chicago Beach Weather Stations - Automated Sensors",
        "published_at_utc": PUBLISHED_AT_UTC,
        "builder": {
            "script": "scripts/build_assignment11_data.py",
            "python": "3.12.13",
            "pandas": pd.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {release_path} ({len(release):,} rows)")
    print(f"SHA-256: {manifest['release_sha256']}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
