# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "numpy==2.0.2",
#   "pandas==3.0.5",
# ]
# ///

"""Independent central-grader reference for Assignment 06.

Production grading reads the five committed CSV artifacts directly and awards milestone credit independently.
"""

from __future__ import annotations

import datetime
from hashlib import sha256
import json
import math
import csv
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21",
    ".gitignore": "2d857aeb38b492c9cac001ba2bef86d2287357f7f5b3f1203d929ac1e79fa138",
    "README.md": "e7aedbc4f7a83dad34db24209a1490ead92e6a1e8c6dd68763eb235a51c2d573",
    "PLATFORM_CHECK.md": "acfb702816fb89e24daf322dd38b177965010b520b5c305c78343fe5e89790ed",
    "check_assignment.py": "4285d5f12ab499117194105e6b3dcd7004da7f5fcd1651e8f8a5fa4640a25703",
    "data/fixture.json": "12b8d3375e4895b6cb443c156794dc9598f5598e64920d2f2818b50883a99f55",
    "data/specimens.csv": "26eeae8d64a2870dc94195a45f924058b777eb1c97f96d2310e86f06403ba605",
    "data/stations_history.csv": "dc6f75e588183d5291abd69b4d5aa856472a711f6ff546b015dd21610d55708c",
    "data/specimens_batch_a.csv": "1aaa71d01d141bf45dd65ba1ec7c28286536c8ee8aa72834c18bcf0b54af2943",
    "data/specimens_batch_b.csv": "8506512a4cef07d7918817e8d8dc15c7230f2923bd28d531326c997995dd58bc",
    "data/review_scores.csv": "d7a1c9570d463a006cec838a4557581467ffb7459d315f57cbfb3cf73274ad22",
    "data/sensor_scores_wide.csv": "6eb9bfb9561fc7c55708bc0038b77b99e6d383f85843dff3e735c5962abe8701",
}
FIXTURE_NAMES = {
    "specimens.csv", "stations_history.csv", "specimens_batch_a.csv",
    "specimens_batch_b.csv", "review_scores.csv", "sensor_scores_wide.csv",
}
OUTPUT_NAMES = ('specimen_merge_audit.csv', 'combined_specimens.csv', 'aligned_features.csv', 'sensor_scores_long.csv', 'sensor_scores_round_trip.csv')
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", *(f"data/{name}" for name in FIXTURE_NAMES),
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}
REQUIRED_CONTEXT_ENV = {
    "assignment": "ASSIGNMENT",
    "submission": "SUBMISSION_TAG",
    "commit": "COMMIT_URL",
    "release": "RELEASE_URL",
}


class InfrastructureError(RuntimeError):
    """Raised when the runner contract is unavailable or grading cannot finish."""


def _context() -> dict[str, str]:
    context: dict[str, str] = {}
    missing: list[str] = []
    for field, environment_name in REQUIRED_CONTEXT_ENV.items():
        value = os.environ.get(environment_name, "").strip()
        if not value:
            missing.append(environment_name)
        context[field] = value
    if missing:
        raise InfrastructureError(
            "missing required grading context: " + ", ".join(missing)
        )
    context["review"] = os.environ.get("REVIEW_URL", "").strip() or context["commit"]
    context["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return context


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_static_contract(root: Path) -> None:
    _assert(sys.version_info[:3] == (3, 12, 13), "grader must use Python 3.12.13")
    _assert(np.__version__ == "2.0.2", "grader must use NumPy 2.0.2")
    _assert(pd.__version__ == "3.0.5", "grader must use pandas 3.0.5")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(root).parts[0] != ".git"
        and path.relative_to(root).parts[0] != "output"
    }
    _assert(STUDENT_PACKAGE_FILES <= actual_files, "required student package files are missing")
    for relative, expected in PROTECTED_FILE_SHA256.items():
        path = root / relative
        _assert(path.is_file(), f"missing protected file: {relative}")
        _assert(sha256(path.read_bytes()).hexdigest() == expected, f"protected file changed: {relative}")
    _assert((root / ".python-version").read_text() == "3.12.13\n", "wrong Python record")
    _assert((root / "requirements.txt").read_text() == "numpy==2.0.2\npandas==3.0.5\n", "wrong dependency records")
    gitignore = (root / ".gitignore").read_text()
    _assert("output/" not in gitignore and "*.csv" not in gitignore, "required CSV artifacts are ignored")
    actual_fixtures = {path.name for path in (root / "data").glob("*.csv") if path.is_file()}
    _assert(FIXTURE_NAMES <= actual_fixtures, "required fixture files are missing")
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "missing regular output directory")
    actual_outputs = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert({".gitkeep", *OUTPUT_NAMES} <= actual_outputs, "required submission outputs are missing")
    return


def _result_test(name: str, maximum: int, error: Exception | None) -> dict:
    passed = error is None
    detail = "all automated checks passed" if passed else str(error)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return {
        "test-name": name,
        "passed": passed,
        "score": maximum if passed else 0,
        "max-score": maximum,
    }


def grade_submission(submission_root: str | Path) -> dict:
    """Grade one local submission and return the grading result object."""

    context = _context()
    root = Path(submission_root).resolve()
    test_specs = (
        ("Task 1 automated", 45),
        ("Task 2 automated", 30),
        ("Task 3 automated", 25),
    )
    errors: dict[str, Exception | None] = {name: None for name, _ in test_specs}
    package_error = None
    try:
        _check_static_contract(root)
    except Exception as error:
        package_error = error
    for task_name, check in (
        ("Task 1 automated", check_merge),
        ("Task 2 automated", check_concat),
        ("Task 3 automated", check_reshape),
    ):
        try:
            check(root)
        except Exception as error:
            errors[task_name] = error

    tests = [_result_test("Submission package", 0, package_error)] + [
        _result_test(name, points, errors[name]) for name, points in test_specs
    ]
    score = sum(test["score"] for test in tests)
    return {
        "schema": "datasci217/grading-result/v1",
        **context,
        "score": score,
        "max-score": 100,
        "tests": tests,
    }


def _fixture_rows(root: Path, name: str) -> list[dict[str, str]]:
    path = root / "data" / name
    _assert(path.is_file() and not path.is_symlink(), f"Missing fixture {name}.")
    _assert(sha256(path.read_bytes()).hexdigest() == PROTECTED_FILE_SHA256[f"data/{name}"], f"Restore fixture {name}.")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _expected_rows(root: Path, name: str) -> list[dict[str, str]]:
    if name == "specimen_merge_audit.csv":
        specimens = _fixture_rows(root, "specimens.csv")
        stations = {row["station_code"]: row for row in _fixture_rows(root, "stations_history.csv") if row["record_status"] == "current"}
        return [{**row, "station_name": stations.get(row["station_code"], {}).get("station_name", ""),
                 "region": stations.get(row["station_code"], {}).get("region", ""),
                 "_merge": "both" if row["station_code"] in stations else "left_only"} for row in specimens]
    if name == "combined_specimens.csv":
        return [{**row, "source_partition": label} for label in ("batch_a", "batch_b")
                for row in _fixture_rows(root, f"specimens_{label}.csv")]
    if name == "aligned_features.csv":
        masses = {row["specimen_id"]: row["mass_g"] for row in _fixture_rows(root, "specimens.csv")[:3]}
        scores = {row["specimen_id"]: row["review_score"] for row in _fixture_rows(root, "review_scores.csv")}
        return [{"specimen_id": key, "mass_g": masses.get(key, ""), "review_score": scores.get(key, "")}
                for key in dict.fromkeys([*masses, *scores])]
    wide = _fixture_rows(root, "sensor_scores_wide.csv")
    if name == "sensor_scores_round_trip.csv":
        return wide
    return [{"sensor_id": row["sensor_id"], "station_code": row["station_code"],
             "measurement_label": field, "value": row[field]}
            for field in ("baseline_value", "followup_value") for row in wide]


def _check_csv(root: Path, name: str) -> None:
    output = root / "output"
    path = output / name
    _assert(output.is_dir() and not output.is_symlink() and path.is_file() and not path.is_symlink(), f"Missing regular output/{name}.")
    expected = _expected_rows(root, name)
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

def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    try:
        result = grade_submission(target)
        Path("result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except InfrastructureError as error:
        print(f"[INFRASTRUCTURE] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
