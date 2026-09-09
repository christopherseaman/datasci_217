"""Dependency-free public artifact checks for Assignment 06.

Checks compare committed CSV values without inspecting or executing student code.
"""

from __future__ import annotations

import csv
from hashlib import sha256
from importlib import metadata
import json
import math
from pathlib import Path
import sys


ASSIGNMENT_DIR = Path(__file__).resolve().parent
EXPECTED_PYTHON = "3.12.13\n"
EXPECTED_REQUIREMENTS = "numpy==2.0.2\npandas==3.0.5\n"
EXPECTED_GITIGNORE = (
    ".venv/\n"
    ".ipynb_checkpoints/\n"
    "__pycache__/\n"
    "*.py[cod]\n"
    ".pytest_cache/\n"
    "result.json\n"
)
PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21",
    ".gitignore": "2d857aeb38b492c9cac001ba2bef86d2287357f7f5b3f1203d929ac1e79fa138",
    "README.md": "e7aedbc4f7a83dad34db24209a1490ead92e6a1e8c6dd68763eb235a51c2d573",
    "PLATFORM_CHECK.md": "acfb702816fb89e24daf322dd38b177965010b520b5c305c78343fe5e89790ed",
    "data/fixture.json": "12b8d3375e4895b6cb443c156794dc9598f5598e64920d2f2818b50883a99f55",
}
FIXTURE_MANIFEST = {
    "fixture_set_id": "a06-structural-wrangling-v1",
    "provenance": "Course-authored synthetic specimen, station, review, and sensor records; no human-subject data.",
    "files": [
        {
            "path": "specimens.csv",
            "row_grain": "one row per specimen",
            "row_count": 7,
            "columns": ["specimen_id", "collector_id", "collection_number", "station_code", "material", "mass_g"],
            "sha256": "26eeae8d64a2870dc94195a45f924058b777eb1c97f96d2310e86f06403ba605",
        },
        {
            "path": "stations_history.csv",
            "row_grain": "one row per station-history record",
            "row_count": 5,
            "columns": ["station_code", "station_name", "region", "record_status"],
            "sha256": "dc6f75e588183d5291abd69b4d5aa856472a711f6ff546b015dd21610d55708c",
        },
        {
            "path": "specimens_batch_a.csv",
            "row_grain": "one row per specimen in source partition A",
            "row_count": 4,
            "columns": ["specimen_id", "collector_id", "collection_number", "station_code", "material", "mass_g"],
            "sha256": "1aaa71d01d141bf45dd65ba1ec7c28286536c8ee8aa72834c18bcf0b54af2943",
        },
        {
            "path": "specimens_batch_b.csv",
            "row_grain": "one row per specimen in source partition B",
            "row_count": 3,
            "columns": ["specimen_id", "collector_id", "collection_number", "station_code", "material", "mass_g"],
            "sha256": "8506512a4cef07d7918817e8d8dc15c7230f2923bd28d531326c997995dd58bc",
        },
        {
            "path": "review_scores.csv",
            "row_grain": "one row per reviewed specimen",
            "row_count": 3,
            "columns": ["specimen_id", "review_score"],
            "sha256": "d7a1c9570d463a006cec838a4557581467ffb7459d315f57cbfb3cf73274ad22",
        },
        {
            "path": "sensor_scores_wide.csv",
            "row_grain": "one row per sensor and station pair",
            "row_count": 4,
            "columns": ["sensor_id", "station_code", "baseline_value", "followup_value"],
            "sha256": "6eb9bfb9561fc7c55708bc0038b77b99e6d383f85843dff3e735c5962abe8701",
        },
    ],
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
    *(f"data/{record['path']}" for record in FIXTURE_MANIFEST["files"]),
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_bytes(relative: str) -> bytes:
    path = ASSIGNMENT_DIR / relative
    _assert(path.is_file(), f"Missing protected file: {relative}.")
    return path.read_bytes()


def _read_json(path: Path, label: str):
    _assert(path.is_file(), f"Missing {label}.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"{label} must be valid UTF-8 JSON: {error}") from error


def _check_submission_inventory() -> None:
    ignored_roots = {
        ".git", "output", "_grader_selftest", ".venv", "venv",
        "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "result.json",
    }
    actual = {
        path.relative_to(ASSIGNMENT_DIR).as_posix()
        for path in ASSIGNMENT_DIR.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not any(part in ignored_roots for part in path.relative_to(ASSIGNMENT_DIR).parts)
    }
    _assert(STUDENT_PACKAGE_FILES <= actual, "Required submission files are missing.")


def check_environment_and_protected_files() -> None:
    _check_submission_inventory()
    _assert(
        sys.version_info[:3] == (3, 12, 13),
        "Run this checker with the Assignment 06 CPython 3.12.13 interpreter.",
    )
    for package, expected in (("numpy", "2.0.2"), ("pandas", "3.0.5")):
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise AssertionError(f"Install {package}=={expected} in this environment.") from error
        _assert(observed == expected, f"Expected {package}=={expected}; found {observed}.")
    _assert(_read_bytes(".python-version").decode() == EXPECTED_PYTHON, "Restore .python-version.")
    _assert(_read_bytes("requirements.txt").decode() == EXPECTED_REQUIREMENTS, "Restore requirements.txt.")
    gitignore = _read_bytes(".gitignore").decode()
    _assert(gitignore == EXPECTED_GITIGNORE, "Restore the supplied .gitignore.")
    _assert("output/" not in gitignore and "*.csv" not in gitignore, "Required CSV outputs must remain visible to Git.")
    for relative, expected in PROTECTED_FILE_SHA256.items():
        observed = sha256(_read_bytes(relative)).hexdigest()
        _assert(observed == expected, f"Restore the protected {relative} file.")


def check_fixtures() -> None:
    manifest = _read_json(ASSIGNMENT_DIR / "data" / "fixture.json", "data/fixture.json")
    _assert(manifest == FIXTURE_MANIFEST, "Restore the exact fixture manifest.")
    expected_names = {record["path"] for record in FIXTURE_MANIFEST["files"]}
    actual_names = {path.name for path in (ASSIGNMENT_DIR / "data").glob("*.csv") if path.is_file()}
    _assert(expected_names <= actual_names, "Required fixture files are missing.")
    for record in FIXTURE_MANIFEST["files"]:
        path = ASSIGNMENT_DIR / "data" / record["path"]
        data = path.read_bytes()
        _assert(data.endswith(b"\n") and b"\r" not in data, f"Restore LF/final-newline bytes in data/{path.name}.")
        _assert(sha256(data).hexdigest() == record["sha256"], f"Restore immutable data/{path.name}.")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        _assert(rows and rows[0] == record["columns"], f"Wrong columns in data/{path.name}.")
        _assert(len(rows) - 1 == record["row_count"], f"Wrong row count in data/{path.name}.")


def _fixture_rows(root: Path, name: str) -> list[dict[str, str]]:
    path = root / "data" / name
    _assert(path.is_file() and not path.is_symlink(), f"Missing fixture {name}.")
    expected = next(record["sha256"] for record in FIXTURE_MANIFEST["files"] if record["path"] == name)
    _assert(sha256(path.read_bytes()).hexdigest() == expected, f"Restore fixture {name}.")
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
    checks = (
        ("environment and protected files", check_environment_and_protected_files),
        ("fixture integrity", check_fixtures),
        ("committed merge audit", lambda: check_merge(ASSIGNMENT_DIR)),
        ("committed concatenation and alignment", lambda: check_concat(ASSIGNMENT_DIR)),
        ("committed reshape and round trip", lambda: check_reshape(ASSIGNMENT_DIR)),
    )
    failures = []
    for label, check in checks:
        try:
            check()
        except Exception as error:
            failures.append(f"[FIX] {label}: {error}")
        else:
            print(f"[OK] {label}")
    if failures:
        print("\n".join(failures))
        print("Assignment 06 is not ready. Fix the messages and regenerate the committed artifacts.")
        return 1
    print("All public checks passed. Instructor review may run stronger checks separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
