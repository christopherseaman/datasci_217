"""Independent central-grader reference for Assignment 06.

Production grading reads the five committed CSV artifacts directly and awards milestone credit independently.
"""

from __future__ import annotations

import json
import math
import csv
from pathlib import Path
import argparse

REFERENCE_ROOT = Path(__file__).resolve().parent


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _result_test(name: str, maximum: int, error: Exception | None) -> dict:
    passed = error is None
    detail = "all automated checks passed" if passed else str(error)
    return {
        "test-name": name,
        "passed": passed,
        "score": maximum if passed else 0,
        "max-score": maximum,
        "detail": detail,
    }


def grade_submission(submission_root: str | Path) -> dict:
    """Grade one local submission and return the grading result object."""

    root = Path(submission_root).resolve()
    test_specs = (
        ("Task 1 automated", 45),
        ("Task 2 automated", 30),
        ("Task 3 automated", 25),
    )
    errors: dict[str, Exception | None] = {name: None for name, _ in test_specs}
    for task_name, check in (
        ("Task 1 automated", check_merge),
        ("Task 2 automated", check_concat),
        ("Task 3 automated", check_reshape),
    ):
        try:
            check(root)
        except Exception as error:
            errors[task_name] = error

    tests = [_result_test(name, points, errors[name]) for name, points in test_specs]
    score = sum(test["score"] for test in tests)
    return {
        "schema": "datasci217/grading-result/v1",
        "score": score,
        "max-score": 100,
        "tests": tests,
    }


def _fixture_rows(name: str) -> list[dict[str, str]]:
    path = REFERENCE_ROOT / "data" / name
    _assert(path.is_file() and not path.is_symlink(), f"Missing fixture {name}.")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _expected_rows(name: str) -> list[dict[str, str]]:
    if name == "specimen_merge_audit.csv":
        specimens = _fixture_rows("specimens.csv")
        stations = {row["station_code"]: row for row in _fixture_rows("stations_history.csv") if row["record_status"] == "current"}
        return [{**row, "station_name": stations.get(row["station_code"], {}).get("station_name", ""),
                 "region": stations.get(row["station_code"], {}).get("region", ""),
                 "_merge": "both" if row["station_code"] in stations else "left_only"} for row in specimens]
    if name == "combined_specimens.csv":
        return [{**row, "source_partition": label} for label in ("batch_a", "batch_b")
                for row in _fixture_rows(f"specimens_{label}.csv")]
    if name == "aligned_features.csv":
        masses = {row["specimen_id"]: row["mass_g"] for row in _fixture_rows("specimens.csv")[:3]}
        scores = {row["specimen_id"]: row["review_score"] for row in _fixture_rows("review_scores.csv")}
        return [{"specimen_id": key, "mass_g": masses.get(key, ""), "review_score": scores.get(key, "")}
                for key in dict.fromkeys([*masses, *scores])]
    wide = _fixture_rows("sensor_scores_wide.csv")
    if name == "sensor_scores_round_trip.csv":
        return wide
    return [{"sensor_id": row["sensor_id"], "station_code": row["station_code"],
             "measurement_label": field, "value": row[field]}
            for field in ("baseline_value", "followup_value") for row in wide]


def _check_csv(root: Path, name: str) -> None:
    output = root / "output"
    path = output / name
    _assert(output.is_dir() and not output.is_symlink() and path.is_file() and not path.is_symlink(), f"Missing regular output/{name}.")
    expected = _expected_rows(name)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        actual = list(reader)
        _assert(reader.fieldnames == list(expected[0]), f"{name}: columns differ.")
    _assert(len(actual) == len(expected), f"{name}: row count differs.")
    numeric = {"collection_number", "mass_g", "review_score", "value", "baseline_value", "followup_value"}
    for index, (row, reference) in enumerate(zip(actual, expected), 1):
        for column, wanted in reference.items():
            got = row[column]
            if column in numeric and wanted != "":
                try:
                    equal = math.isclose(float(got), float(wanted), rel_tol=1e-9, abs_tol=1e-9)
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = got == wanted
            _assert(equal, f"{name}: row {index}, {column} differs (check values and the documented source order).")


def check_merge(root: Path) -> None:
    _check_csv(root, "specimen_merge_audit.csv")


def check_concat(root: Path) -> None:
    _check_csv(root, "combined_specimens.csv")
    _check_csv(root, "aligned_features.csv")


def check_reshape(root: Path) -> None:
    _check_csv(root, "sensor_scores_long.csv")
    _check_csv(root, "sensor_scores_round_trip.csv")

def _format_result(result: dict) -> str:
    return "\n".join(
        f"[{'PASS' if test['passed'] else 'FAIL'}] {test['test-name']}: "
        f"{test['score']}/{test['max-score']}"
        + (f" — {test['detail']}" if test.get("detail") else "")
        for test in result["tests"]
    ) + f"\nScore: {result['score']}/{result['max-score']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade committed assignment artifacts without executing submission code.")
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    result = grade_submission(args.submission_dir)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(_format_result(result))
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
