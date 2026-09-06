"""Public, artifact-only checks for Assignment 09."""
from __future__ import annotations
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
PROTECTED={".python-version":"aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b","requirements.txt":"90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21",".gitignore":"835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95","README.md":"1f0ef949d7fd678e00fe58d8bb94d6af0e75e3fd5fd73a5715b6e965d5aed18c","PLATFORM_CHECK.md":"f3aa2d2dc6eff93a637177fec91aded84fad799e1a64b1180744fc36e1d2ad8e","data/fixture.json":"27558bc4da7738775879501a6f11a0a9d874f3948823e54bb5e82ab91a02d703","data/zone_co2_readings.csv":"c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4"}
ARTIFACTS={"prepared_panel.csv":(12,["zone","recorded_at","co2_ppm","source_row"]),"hourly_grid.csv":(16,["zone","recorded_at","co2_ppm","source_row","grid_created_row","source_value_missing"]),"two_hour_summary.csv":(8,["zone","recorded_at","mean_co2_ppm","reading_count"]),"temporal_features.csv":(12,["zone","recorded_at","co2_ppm","co2_lag_1","co2_difference","mean_previous_2_observations","mean_previous_2h"]),"availability_decisions.csv":(4,["candidate","latest_required_timestamp","available_by_prediction_time","decision"]),"chronological_blocks.csv":(12,["zone","recorded_at","co2_ppm","source_row","block"])}
def check_environment_and_files():
    for name,digest in PROTECTED.items():
        path=ROOT/name; assert path.is_file() and sha256(path.read_bytes()).hexdigest()==digest, f"Restore protected {name}."
def check_artifacts():
    output=ROOT/"output"; assert output.is_dir() and not output.is_symlink(),"Missing regular output/ directory."
    assert {p.name for p in output.iterdir() if p.is_file() or p.is_symlink()}==set(ARTIFACTS)|{".gitkeep"},"Keep exactly six required CSVs plus output/.gitkeep."
    for name,(rows,columns) in ARTIFACTS.items():
        path=output/name; assert path.is_file() and not path.is_symlink(),f"output/{name} must be regular."
        with path.open(newline="",encoding="utf-8") as handle: parsed=list(csv.reader(handle))
        assert parsed and parsed[0]==columns and len(parsed)-1==rows,f"Wrong schema or row count in output/{name}."
    source = _fixture(ROOT)
    _task1(ROOT, source)
    _task2(ROOT, source)
    _task3(ROOT, source)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _csv(root: Path, name: str, dates: list[str]) -> pd.DataFrame:
    path = root / "output" / name
    _assert(path.is_file() and not path.is_symlink(), f"missing regular artifact: {name}")
    try:
        frame = pd.read_csv(path, keep_default_na=True)
        for column in dates:
            frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame
    except Exception as error:
        raise AssertionError(f"invalid CSV {name}: {error}") from error


def _same(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], label: str, ordered: bool = False) -> None:
    _assert(actual.columns.tolist() == expected.columns.tolist(), f"{label} columns differ")
    _assert(not actual.duplicated(keys).any(), f"{label} has duplicate row identities")
    if ordered:
        _assert(actual[keys].astype(str).equals(expected[keys].astype(str)), f"{label} chronology/order differs")
    else:
        _assert(set(map(tuple, actual[keys].astype(str).to_numpy())) == set(map(tuple, expected[keys].astype(str).to_numpy())), f"{label} row identities differ")
        actual = actual.sort_values(keys, kind="stable").reset_index(drop=True); expected = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    for column in actual.columns:
        if column in keys: continue
        left, right = actual[column], expected[column]
        if pd.api.types.is_bool_dtype(right):
            _assert(left.astype("boolean").equals(right.astype("boolean")), f"{label} values differ: {column}")
        elif pd.api.types.is_numeric_dtype(right):
            _assert(np.allclose(pd.to_numeric(left), right, rtol=1e-7, atol=1e-8, equal_nan=True), f"{label} values differ: {column}")
        else:
            _assert(left.fillna("<missing>").astype(str).equals(right.fillna("<missing>").astype(str)), f"{label} values differ: {column}")


def _fixture(root: Path) -> pd.DataFrame:
    path = root / "data" / "zone_co2_readings.csv"
    _assert(path.is_file() and sha256(path.read_bytes()).hexdigest() == PROTECTED["data/zone_co2_readings.csv"], "fixture changed")
    source = pd.read_csv(path, dtype={"zone": "string", "recorded_at": "string", "co2_ppm": "float64"})
    source["recorded_at"] = pd.to_datetime(source["recorded_at"], format="%Y-%m-%d %H:%M").dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    source["source_row"] = 1
    return source.sort_values(["zone", "recorded_at"], kind="stable").reset_index(drop=True)


def _task1(root: Path, source: pd.DataFrame) -> None:
    _same(_csv(root, "prepared_panel.csv", ["recorded_at"]), source, ["zone", "recorded_at"], "prepared panel", ordered=True)


def _task2(root: Path, source: pd.DataFrame) -> None:
    hourly = source.set_index("recorded_at").groupby("zone")[["co2_ppm", "source_row"]].resample("h").asfreq().reset_index()
    hourly["grid_created_row"] = hourly["source_row"].isna(); hourly["source_value_missing"] = hourly["source_row"].eq(1) & hourly["co2_ppm"].isna()
    hourly = hourly[["zone", "recorded_at", "co2_ppm", "source_row", "grid_created_row", "source_value_missing"]]
    _same(_csv(root, "hourly_grid.csv", ["recorded_at"]), hourly, ["zone", "recorded_at"], "hourly grid", ordered=True)
    summary = source.set_index("recorded_at").groupby("zone").resample("2h", closed="left", label="left").agg(mean_co2_ppm=("co2_ppm", "mean"), reading_count=("source_row", "sum")).reset_index()
    _same(_csv(root, "two_hour_summary.csv", ["recorded_at"]), summary, ["zone", "recorded_at"], "two-hour summary", ordered=True)


def _task3(root: Path, source: pd.DataFrame) -> None:
    features = source[["zone", "recorded_at", "co2_ppm"]].copy(); group = features.groupby("zone")["co2_ppm"]
    features["co2_lag_1"] = group.shift(); features["co2_difference"] = group.diff(); features["mean_previous_2_observations"] = group.transform(lambda x: x.shift().rolling(2, min_periods=1).mean())
    elapsed = features.set_index("recorded_at").groupby("zone")["co2_ppm"].rolling("2h", closed="left", min_periods=1).mean().rename("mean_previous_2h").reset_index()
    features = features.merge(elapsed, on=["zone", "recorded_at"], validate="one_to_one", sort=False)
    _same(_csv(root, "temporal_features.csv", ["recorded_at"]), features, ["zone", "recorded_at"], "past features", ordered=True)
    availability = pd.DataFrame({"candidate": ["calendar hour", "previous recorded CO2", "centered three-observation mean", "next recorded CO2"], "latest_required_timestamp": pd.to_datetime(["2026-01-20 18:00Z", "2026-01-20 17:00Z", "2026-01-20 19:00Z", "2026-01-20 19:00Z"], utc=True), "available_by_prediction_time": [True, True, False, False], "decision": ["keep", "keep", "reject", "reject"]})
    _same(_csv(root, "availability_decisions.csv", ["latest_required_timestamp"]), availability, ["candidate"], "availability decisions")
    blocks = source.copy(); blocks["block"] = np.where(blocks["recorded_at"] < pd.Timestamp("2026-01-20 18:00", tz="UTC"), "earlier", "later_holdout")
    _same(_csv(root, "chronological_blocks.csv", ["recorded_at"]), blocks, ["zone", "recorded_at"], "chronological blocks", ordered=True)


def main():
    failures=[]
    for label,check in (("environment and protected files",check_environment_and_files),("six generated artifacts",check_artifacts)):
        try: check()
        except Exception as error: failures.append(f"[FIX] {label}: {error}")
    if failures: print("\n".join(failures)); return 1
    print("All public checks passed. The six committed artifacts are ready for grading."); return 0
if __name__=="__main__": raise SystemExit(main())
