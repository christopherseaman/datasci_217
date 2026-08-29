# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "numpy==2.0.2",
#   "pandas==3.0.5",
#   "pyarrow==25.0.0",
# ]
# ///
"""Build the compact Yellow Taxi release used by Lecture 11 demos."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "11" / "demo" / "data"
MONTHS = range(1, 7)
LOCAL_TZ = "America/New_York"
SAMPLE_ROWS_PER_MONTH = 5_000
ZONE_COUNT = 12
PUBLISHED_AT_UTC = "2026-07-22T20:00:00Z"
SOURCE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2023-{month:02d}.parquet"
)
SOURCE_SHA256 = {
    1: "32df6f67578fa86c484a6b5ef23a5281992ff085521082340b0f9e5889e9a572",
    2: "4809e6aaac64f05a62d16a25d55713be1537ad64fc261e895eaf2d2120fe750a",
    3: "e7d44943111b007bf0e7084863511886e0db29f862f9a96239383a1f86c6c26e",
    4: "95c01c53c865e06489179bdecce3a59603697a1557c4d5d183c20ae13a144dc6",
    5: "9bd7d1c557bb9d0413619ba296b4828b884852a70bd07f2230b83f080d9a8591",
    6: "3e60e29b5df45683948b68acbfb503aa459d14162c610622585c6580fdbfc73a",
}
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
LOOKUP_SHA256 = "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "datasci-217-release-builder"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"download failed with HTTP {response.status}: {url}")
        destination.write_bytes(response.read())


def valid_events(frame: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce").dt.tz_localize(
        LOCAL_TZ,
        ambiguous="NaT",
        nonexistent="NaT",
    )
    zone = pd.to_numeric(frame["PULocationID"], errors="coerce")
    valid = (
        local.ge(pd.Timestamp("2023-01-01", tz=LOCAL_TZ))
        & local.lt(pd.Timestamp("2023-07-01", tz=LOCAL_TZ))
        & zone.notna()
        & np.isfinite(zone.astype(float))
        & zone.astype(float).mod(1).eq(0)
        & zone.gt(0)
    )
    return pd.DataFrame(
        {
            "pickup_datetime_utc": local.loc[valid].dt.tz_convert("UTC"),
            "pickup_zone_id": zone.loc[valid].astype("int64"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_paths = {
        "sample": output_dir / "yellow_taxi_2023_h1_event_sample.parquet",
        "panel": output_dir / "yellow_taxi_2023_h1_zone_hour_counts.parquet",
        "lookup": output_dir / "taxi_zone_lookup.csv",
        "manifest": output_dir / "demo_release_manifest.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError("Lecture 11 demo release already exists in the output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ds217-l11-demo-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        source_root = args.source_dir.resolve() if args.source_dir else temporary_root
        sources: list[dict[str, object]] = []
        samples: list[pd.DataFrame] = []
        valid_by_month: dict[int, pd.DataFrame] = {}

        for month in MONTHS:
            url = SOURCE_URL.format(month=month)
            source_path = source_root / Path(url).name
            if not source_path.exists():
                print(f"Downloading {url}")
                download(url, source_path)
            source_hash = sha256(source_path)
            if source_hash != SOURCE_SHA256[month]:
                raise RuntimeError(f"source hash changed for {source_path.name}: {source_hash}")

            raw = pd.read_parquet(
                source_path,
                columns=["tpep_pickup_datetime", "PULocationID"],
            )
            sample_positions = np.linspace(
                0,
                len(raw) - 1,
                num=min(SAMPLE_ROWS_PER_MONTH, len(raw)),
                dtype=np.int64,
            )
            sample = raw.iloc[sample_positions].copy()
            sample.insert(0, "source_month", f"2023-{month:02d}")
            sample.insert(1, "source_row_number", sample_positions)
            sample.insert(
                2,
                "course_row_id",
                [f"2023-{month:02d}-{row_number:08d}" for row_number in sample_positions],
            )
            sample = sample.rename(
                columns={
                    "tpep_pickup_datetime": "pickup_datetime_local",
                    "PULocationID": "pickup_zone_id",
                }
            )
            sample["pickup_zone_id"] = sample["pickup_zone_id"].astype("Int64")
            samples.append(sample)
            valid_by_month[month] = valid_events(raw)
            sources.append(
                {
                    "month": f"2023-{month:02d}",
                    "url": url,
                    "filename": source_path.name,
                    "sha256": source_hash,
                    "byte_size": source_path.stat().st_size,
                    "row_count": len(raw),
                }
            )

        january_counts = valid_by_month[1]["pickup_zone_id"].value_counts()
        top_zones = (
            january_counts.rename_axis("pickup_zone_id")
            .reset_index(name="pickup_count")
            .sort_values(
                ["pickup_count", "pickup_zone_id"],
                ascending=[False, True],
                kind="stable",
            )
            .head(ZONE_COUNT)["pickup_zone_id"]
            .tolist()
        )
        observed = pd.concat(valid_by_month.values(), ignore_index=True)
        observed = observed.loc[observed["pickup_zone_id"].isin(top_zones)].copy()
        observed["target_hour_utc"] = observed["pickup_datetime_utc"].dt.floor("h")
        observed_counts = (
            observed.groupby(["target_hour_utc", "pickup_zone_id"], observed=True, sort=True)
            .size()
            .rename("pickup_count")
            .reset_index()
        )
        hours = pd.date_range(
            pd.Timestamp("2023-01-01", tz=LOCAL_TZ).tz_convert("UTC"),
            pd.Timestamp("2023-07-01", tz=LOCAL_TZ).tz_convert("UTC"),
            freq="h",
            inclusive="left",
        )
        panel = pd.MultiIndex.from_product(
            [hours, sorted(top_zones)],
            names=["target_hour_utc", "pickup_zone_id"],
        ).to_frame(index=False)
        panel = panel.merge(
            observed_counts,
            on=["target_hour_utc", "pickup_zone_id"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        panel["target_hour_local"] = panel["target_hour_utc"].dt.tz_convert(LOCAL_TZ)
        panel["pickup_count"] = panel["pickup_count"].fillna(0).astype("int64")
        panel = panel[
            ["pickup_zone_id", "target_hour_utc", "target_hour_local", "pickup_count"]
        ]

        pd.concat(samples, ignore_index=True).to_parquet(
            output_paths["sample"],
            compression="zstd",
            index=False,
        )
        panel.to_parquet(output_paths["panel"], compression="zstd", index=False)
        download(LOOKUP_URL, output_paths["lookup"])
        lookup_hash = sha256(output_paths["lookup"])
        if lookup_hash != LOOKUP_SHA256:
            raise RuntimeError(f"taxi zone lookup hash changed: {lookup_hash}")

    manifest = {
        "schema": "datasci217/lecture11-demo-release/v1",
        "release_id": "yellow-taxi-2023-h1-demo-v1",
        "published_at_utc": PUBLISHED_AT_UTC,
        "source_timezone": LOCAL_TZ,
        "source_page": "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
        "attribution": "NYC Taxi and Limousine Commission Trip Record Data",
        "source_disclaimer": (
            "TLC states that technology providers supplied these records and that TLC "
            "does not represent their accuracy."
        ),
        "source_files": sources,
        "top_zone_selection": {
            "source_period": "2023-01",
            "zone_count": ZONE_COUNT,
            "zone_ids": sorted(top_zones),
            "ranking": "pickup count descending, pickup zone ID ascending tie-break",
        },
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": sha256(path),
                "byte_size": path.stat().st_size,
            }
            for name, path in output_paths.items()
            if name != "manifest"
        },
        "sample_rows": SAMPLE_ROWS_PER_MONTH * len(MONTHS),
        "panel_rows": len(panel),
        "builder": {
            "script": "scripts/build_lecture11_demo_data.py",
            "python": "3.12.13",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_paths['sample']}")
    print(f"Wrote {output_paths['panel']} ({len(panel):,} rows)")
    print(f"Wrote {output_paths['manifest']}")


if __name__ == "__main__":
    main()
