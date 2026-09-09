# /// script
# requires-python = "==3.12.13"
# dependencies = ["numpy==2.0.2", "pandas==3.0.5"]
# ///

"""Independent central-grader reference for Assignment 05.

This module intentionally does not import the student-editable public checker.
Production grading-service wiring is external to this repository.
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


EXPECTED_SHA256 = "d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b"
EXPECTED_MANIFEST = {
    "fixture_id": "a05-people-cleaning-v1",
    "provenance": "course-authored synthetic teaching data; no real people",
    "row_meaning": "one submitted person record",
    "candidate_identifier": ["record_id"],
    "row_count": 12,
    "raw_columns": [
        "record_id",
        "full_name",
        "site",
        "status",
        "age_text",
        "visit_date",
    ],
    "sha256": EXPECTED_SHA256,
}
EXPECTED_ISSUES = [
    ("schema mismatch", 0),
    ("empty full-name tokens", 1),
    ("empty date tokens", 1),
    ("age sentinel tokens", 3),
    ("status sentinel tokens", 1),
    ("age parse failures", 1),
    ("numeric but noninteger age values", 1),
    ("age values outside 0 through 120", 1),
    ("date parse failures", 3),
    ("rows in exact duplicate sets", 2),
    ("rows with repeated candidate IDs", 2),
    ("site values needing format normalization", 4),
    ("status values needing format normalization", 3),
    ("unexpected site values", 0),
    ("unexpected non-sentinel status values", 0),
]
EXPECTED_DECISIONS = [
    ("full_name", "empty optional name", "retain as missing"),
    (
        "full_name, site, status",
        "surrounding whitespace and case variants",
        "strip surrounding whitespace and normalize bounded field case",
    ),
    ("status", "NA sentinel", "convert the documented sentinel to missing"),
    (
        "age_text",
        "unknown and -9 sentinels",
        "convert the documented sentinels to missing",
    ),
    (
        "age_text",
        "nonnumeric, fractional, or out-of-range values",
        "coerce invalid values to missing without rounding",
    ),
    (
        "visit_date",
        "empty, lexically invalid, or calendar-invalid values",
        "coerce invalid values to missing after an exact-format check",
    ),
    (
        "all raw columns",
        "exact duplicate submissions",
        "keep the first exact raw row only",
    ),
    (
        "all fields",
        "adjacent-row filling",
        "do not forward-fill or backward-fill",
    ),
]
OUTPUT_FILES = (
    "raw_preview.txt",
    "numpy_age_summary.csv",
    "pandas_selection.csv",
    "pipeline_summary.txt",
    "issue_audit.csv",
    "cleaned_people.csv",
    "decision_log.csv",
)
EXPECTED_RAW_PREVIEW = """$ head -n 4 data/people_raw.csv
record_id,full_name,site,status,age_text,visit_date
R001, Alice Smith , North ,Active,34,2026-01-15
R002,BOB JONES,north,active,unknown,2026-02-30
R002,BOB JONES,north,active,unknown,2026-02-30
$ tail -n 2 data/people_raw.csv
R010,Jamie Okafor,West,Complete,28,2026-07-15
R011,Kai Patel,south, pending ,0,2026-08-01
"""
EXPECTED_NUMPY_SUMMARY = pd.DataFrame(
    [("count", 6), ("min", 0), ("max", 52), ("sum", 198), ("mean", 33.0)],
    columns=["metric", "value"],
)
EXPECTED_PANDAS_SELECTION = pd.DataFrame(
    [("R001", " North ", "Active"), ("R003", "SOUTH", "pending"), ("R010", "West", "Complete")],
    columns=["record_id", "site", "status"],
)
EXPECTED_PIPELINE_SUMMARY = """raw_rows=12
raw_columns=6
exact_duplicate_rows=1
candidate_id_duplicate_rows=1
clean_rows=11
"""
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/people_raw.csv",
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
    _assert((root / ".python-version").read_text() == "3.12.13\n", "wrong Python record")
    _assert(
        (root / "requirements.txt").read_text() == "numpy==2.0.2\npandas==3.0.5\n",
        "wrong dependency records",
    )
    manifest_path = root / "data" / "fixture.json"
    data_path = root / "data" / "people_raw.csv"
    _assert(manifest_path.is_file(), "missing fixture manifest")
    _assert(data_path.is_file(), "missing data fixture")
    _assert(json.loads(manifest_path.read_text()) == EXPECTED_MANIFEST, "fixture manifest changed")
    data = data_path.read_bytes()
    _assert(len(data) == 570 and sha256(data).hexdigest() == EXPECTED_SHA256, "fixture bytes changed")
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "missing regular output directory")
    actual_outputs = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert({".gitkeep", *OUTPUT_FILES} <= actual_outputs, "required submission outputs are missing")
    return


def _expected_cleaned() -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "record_id": [f"R{number:03d}" for number in range(1, 12)],
            "full_name": [
                "Alice Smith", "Bob Jones", "Carla Ruiz", pd.NA, "Evan Li",
                "Fatima Noor", "Grace Chen", "Hugo Diaz", "Inez Park",
                "Jamie Okafor", "Kai Patel",
            ],
            "site": [
                "north", "north", "south", "south", "west", "north",
                "south", "west", "north", "west", "south",
            ],
            "status": [
                "active", "active", "pending", pd.NA, "complete", "active",
                "active", "pending", "complete", "complete", "pending",
            ],
            "age": [34, pd.NA, pd.NA, 45, 52, pd.NA, pd.NA, pd.NA, 39, 28, 0],
            "visit_date": [
                "2026-01-15", None, "2026-03-01", None, "2026-02-14",
                "2026-04-01", "2026-05-01", "2026-06-01", None,
                "2026-07-15", "2026-08-01",
            ],
            "needs_review": [False, True, True, True, False, True, True, True, True, False, False],
        }
    )
    for column in ("record_id", "full_name", "site", "status"):
        table[column] = table[column].astype("string")
    table["age"] = table["age"].astype("Int64")
    table["visit_date"] = pd.to_datetime(
        table["visit_date"], format="%Y-%m-%d", errors="coerce"
    ).astype("datetime64[us]")
    table["needs_review"] = table["needs_review"].astype("boolean")
    return table


def _artifact_path(root: Path, name: str) -> Path:
    output = root / "output"
    path = output / name
    _assert(output.is_dir() and not output.is_symlink() and path.is_file() and not path.is_symlink(), f"Missing regular output/{name}.")
    return path


def check_foundations(root: Path) -> None:
    _assert(_artifact_path(root, "raw_preview.txt").read_text(encoding="utf-8") == EXPECTED_RAW_PREVIEW, "raw_preview.txt does not match the required terminal evidence.")
    _assert(_artifact_path(root, "pipeline_summary.txt").read_text(encoding="utf-8") == EXPECTED_PIPELINE_SUMMARY, "pipeline_summary.txt does not match the required Python summary.")
    numpy_summary = pd.read_csv(_artifact_path(root, "numpy_age_summary.csv"), dtype={"metric": "string", "value": "float64"})
    expected_numpy = EXPECTED_NUMPY_SUMMARY.astype({"metric": "string", "value": "float64"})
    pd.testing.assert_frame_equal(numpy_summary, expected_numpy)
    selection = pd.read_csv(_artifact_path(root, "pandas_selection.csv"), dtype="string")
    pd.testing.assert_frame_equal(selection, EXPECTED_PANDAS_SELECTION.astype("string"))


def check_audit(root: Path) -> None:
    check_foundations(root)
    audit = pd.read_csv(_artifact_path(root, "issue_audit.csv"), dtype={"issue": "string", "count": "Int64"})
    expected = pd.DataFrame(EXPECTED_ISSUES, columns=["issue", "count"]).astype({"issue": "string", "count": "Int64"})
    _assert(audit.columns.tolist() == expected.columns.tolist(), "issue_audit.csv columns differ.")
    _assert(audit["issue"].is_unique and audit["issue"].notna().all(), "Audit issues must be unique and present.")
    pd.testing.assert_frame_equal(audit.sort_values("issue").reset_index(drop=True), expected.sort_values("issue").reset_index(drop=True))


def check_cleaned(root: Path) -> None:
    cleaned = pd.read_csv(_artifact_path(root, "cleaned_people.csv"), dtype={
        "record_id": "string", "full_name": "string", "site": "string",
        "status": "string", "age": "Int64", "visit_date": "string", "needs_review": "boolean",
    })
    _assert(cleaned.columns.tolist() == _expected_cleaned().columns.tolist(), "cleaned_people.csv columns differ.")
    _assert(cleaned["record_id"].notna().all() and cleaned["record_id"].is_unique, "Record IDs must be present and unique.")
    tokens = cleaned["visit_date"]
    _assert(tokens.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", na=True).all(), "Dates must be YYYY-MM-DD or missing.")
    dates = pd.to_datetime(tokens, format="%Y-%m-%d", errors="coerce")
    _assert(not (tokens.notna() & dates.isna()).any(), "Export valid calendar dates or missing values.")
    cleaned["visit_date"] = dates.astype("datetime64[us]")
    pd.testing.assert_frame_equal(cleaned.sort_values("record_id").reset_index(drop=True), _expected_cleaned().sort_values("record_id").reset_index(drop=True))


def check_decisions(root: Path) -> None:
    columns = ["field", "issue", "action", "reason", "source", "source_sha256", "rows_before", "rows_after"]
    decision = pd.read_csv(_artifact_path(root, "decision_log.csv"), dtype={
        **{name: "string" for name in columns[:6]}, "rows_before": "Int64", "rows_after": "Int64",
    })
    _assert(decision.columns.tolist() == columns, "decision_log.csv columns differ.")
    observed = list(decision[["field", "issue", "action"]].itertuples(index=False, name=None))
    _assert(len(observed) == len(EXPECTED_DECISIONS) and set(observed) == set(EXPECTED_DECISIONS), "Decision fields, issues, or actions differ.")
    _assert(decision["reason"].fillna("").str.strip().ne("").all(), "Every decision needs a nonblank reason.")
    _assert(decision["source"].fillna("").eq("data/people_raw.csv").all(), "Wrong source provenance.")
    _assert(decision["source_sha256"].fillna("").eq(EXPECTED_SHA256).all(), "Wrong source checksum.")
    _assert(decision["rows_before"].eq(12).fillna(False).all() and decision["rows_after"].eq(11).fillna(False).all(), "Wrong before/after row counts.")


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
    tests: list[dict] = []
    package_error: Exception | None = None
    try:
        _check_static_contract(root)
    except Exception as error:
        package_error = error
    tests.append(_result_test("Submission package", 0, package_error))
    for name, points, check in (
        ("Task 1 cumulative foundations and audit", 25, check_audit),
        ("Task 2 committed cleaning", 35, check_cleaned),
        ("Task 3 committed decision log", 25, check_decisions),
    ):
        error = None
        try:
            check(root)
        except Exception as failure:
            error = failure
        tests.append(_result_test(name, points, error))

    score = sum(test["score"] for test in tests)
    return {
        "schema": "datasci217/grading-result/v1",
        **context,
        "score": score,
        "max-score": 85,
        "tests": tests,
    }


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    try:
        result = grade_submission(target)
        Path("result.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2))
    except InfrastructureError as error:
        print(f"[INFRASTRUCTURE] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
