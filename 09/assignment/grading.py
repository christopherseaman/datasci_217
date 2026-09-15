# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#   "ipykernel==6.29.5",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.3.3",
#   "pandas==3.0.5",
# ]
# ///

"""Teacher-controlled central-grader reference for Assignment 09.

Production grading reads the six committed CSV artifacts directly. Notebook
execution and alternate-input checks remain optional release QA.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


GRADER_ROOT = Path(__file__).resolve().parent


class InfrastructureError(RuntimeError):
    """A runner/grader failure for which no student grade is valid."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _result_test(name: str, maximum: int, error: Exception | None) -> dict:
    passed = error is None
    return {"test-name": name, "passed": passed, "score": maximum if passed else 0,
            "max-score": maximum, "detail": "artifact contract passed" if passed else str(error)}


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


def _fixture() -> pd.DataFrame:
    path = GRADER_ROOT / "data" / "zone_co2_readings.csv"
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


def grade_submission(submission_root: str | Path) -> dict:
    """Grade committed artifacts only; student notebooks are never executed."""
    root = Path(submission_root).resolve(); source: pd.DataFrame | None = None
    specs = (("Task 1 temporal preparation", 30, lambda s: _task1(root, s)), ("Task 2 temporal summaries", 35, lambda s: _task2(root, s)), ("Task 3 past-only chronology", 35, lambda s: _task3(root, s)))
    tests = []
    for name, maximum, check in specs:
        try:
            if source is None: source = _fixture()
            check(source)
        except Exception as error: tests.append(_result_test(name, maximum, error))
        else: tests.append(_result_test(name, maximum, None))
    return {"schema": "datasci217/grading-result/v1", "score": sum(t["score"] for t in tests), "max-score": 100, "tests": tests}




def main(argv: list[str] | None = None) -> int:
    import argparse
    from contextlib import redirect_stdout

    parser = argparse.ArgumentParser(description="Grade saved assignment artifacts.")
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr if args.as_json else sys.stdout):
            result = grade_submission(args.submission_dir)
    except Exception as error:
        if args.as_json:
            print(json.dumps({"schema": "datasci217/grading-result/v1",
                              "error": f"{type(error).__name__}: {error}"}))
        else:
            print(f"[INFRASTRUCTURE] {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        for test in result["tests"]:
            print(f"[{'PASS' if test['passed'] else 'FIX'}] {test['test-name']} "
                  f"({test['score']}/{test['max-score']})")
        print(f"Automated score: {result['score']}/{result['max-score']}")
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
