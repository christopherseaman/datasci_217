# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==6.29.5",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.5",
# ]
# ///

"""Teacher-controlled central-grader reference for Assignment 09.

Production grading reads the six committed CSV artifacts directly. Notebook
execution and alternate-input checks remain optional release QA.
"""

from __future__ import annotations

import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    "README.md": "4491b423eb0e8f3a67bd6ec195f8726fdec6f17ca6571b98fb0b603b656ee9cc",
    "PLATFORM_CHECK.md": "f3aa2d2dc6eff93a637177fec91aded84fad799e1a64b1180744fc36e1d2ad8e",
    "check_assignment.py": "d8c7a2d0a21a261f7f2c12fe2df5205b80bf4ec212a8f298a13296c3444cc0b8",
    "data/fixture.json": "27558bc4da7738775879501a6f11a0a9d874f3948823e54bb5e82ab91a02d703",
    "data/zone_co2_readings.csv": "c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4",
}
ARTIFACT_NAMES = {"prepared_panel.csv", "hourly_grid.csv", "two_hour_summary.csv", "temporal_features.csv", "availability_decisions.csv", "chronological_blocks.csv"}


class InfrastructureError(RuntimeError):
    """A runner/grader failure for which no student grade is valid."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _context() -> dict[str, str]:
    mapping = {
        "assignment": "ASSIGNMENT",
        "submission": "SUBMISSION_TAG",
        "commit": "COMMIT_URL",
        "release": "RELEASE_URL",
    }
    result = {}
    for field, variable in mapping.items():
        value = os.environ.get(variable, "").strip()
        if not value:
            raise InfrastructureError(f"missing required grading context: {variable}")
        result[field] = value
    result["review"] = os.environ.get("REVIEW_URL", "").strip() or result["commit"]
    result["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def _result_test(name: str, maximum: int, error: Exception | None) -> dict:
    passed = error is None
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {'all automated checks passed' if passed else error}")
    return {"test-name": name, "passed": passed, "score": maximum if passed else 0, "max-score": maximum}


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    result_path = Path("result.json")
    try:
        if result_path.exists():
            if not result_path.is_file():
                raise InfrastructureError("result.json path is not a regular file")
            result_path.unlink()
        result = grade_submission(target)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except Exception as error:
        if result_path.is_file():
            result_path.unlink()
        print(f"Grader infrastructure failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


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
    _assert(path.is_file() and sha256(path.read_bytes()).hexdigest() == PROTECTED_FILE_SHA256["data/zone_co2_readings.csv"], "fixture changed")
    source = pd.read_csv(path, dtype={"zone": "string", "recorded_at": "string", "co2_ppm": "float64"})
    source["recorded_at"] = pd.to_datetime(source["recorded_at"], format="%Y-%m-%d %H:%M").dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    source["source_row"] = 1
    return source.sort_values(["zone", "recorded_at"], kind="stable").reset_index(drop=True)


def _inventory(root: Path) -> None:
    output = root / "output"; _assert(output.is_dir() and not output.is_symlink(), "missing regular output directory")
    _assert(ARTIFACT_NAMES | {".gitkeep"} <= {p.name for p in output.iterdir() if p.is_file() or p.is_symlink()}, "required output artifacts are missing")


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


def grade_submission(submission_root: str | Path) -> dict:
    """Grade committed artifacts only; student notebooks are never executed."""
    context = _context(); root = Path(submission_root).resolve(); source: pd.DataFrame | None = None
    specs = (("Fixture integrity", 10, lambda s: _fixture(root)), ("Task 1 temporal preparation", 25, lambda s: _task1(root, s)), ("Task 2 temporal summaries", 30, lambda s: _task2(root, s)), ("Task 3 past-only chronology", 30, lambda s: _task3(root, s)), ("Visible artifact inventory", 5, lambda s: _inventory(root)))
    tests = []
    for name, maximum, check in specs:
        try:
            if source is None: source = _fixture(root)
            check(source)
        except Exception as error: tests.append(_result_test(name, maximum, error))
        else: tests.append(_result_test(name, maximum, None))
    return {"schema": "datasci217/grading-result/v1", **context, "score": sum(t["score"] for t in tests), "max-score": 100, "tests": tests}


if __name__ == "__main__":
    raise SystemExit(main())
