"""Run the discoverable public checks for Assignment 05.

Checks compare committed artifact values without inspecting or executing student code.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ASSIGNMENT_DIR = Path(__file__).resolve().parent
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
EXPECTED_NUMPY_SUMMARY = [("count", 6), ("min", 0), ("max", 52), ("sum", 198), ("mean", 33.0)]
EXPECTED_PANDAS_SELECTION = [("R001", " North ", "Active"), ("R003", "SOUTH", "pending"), ("R010", "West", "Complete")]
EXPECTED_PIPELINE_SUMMARY = """raw_rows=12
raw_columns=6
exact_duplicate_rows=1
candidate_id_duplicate_rows=1
clean_rows=11
"""
EXPECTED_PYTHON_FILE = "3.12.13\n"
EXPECTED_REQUIREMENTS = "numpy==2.0.2\npandas==3.0.5\n"
EXPECTED_GITIGNORE = (
    ".venv/\n"
    ".ipynb_checkpoints/\n"
    "__pycache__/\n"
    "*.py[cod]\n"
    ".pytest_cache/\n"
)
EXPECTED_DATA_SHA256 = (
    "d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b"
)
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
    "sha256": EXPECTED_DATA_SHA256,
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
    (
        "full_name",
        "empty optional name",
        "retain as missing",
    ),
    (
        "full_name, site, status",
        "surrounding whitespace and case variants",
        "strip surrounding whitespace and normalize bounded field case",
    ),
    (
        "status",
        "NA sentinel",
        "convert the documented sentinel to missing",
    ),
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
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/people_raw.csv",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_text(path: Path, label: str) -> str:
    _assert(path.is_file(), f"Missing {label}.")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError(f"{label} must be UTF-8 text.") from error


def _load_json(path: Path, label: str):
    try:
        return json.loads(_read_text(path, label))
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{label} is not valid JSON at line {error.lineno}: {error.msg}."
        ) from error


def _check_submission_inventory(root: Path) -> None:
    ignored_roots = {
        ".git", "output", "_grader_selftest", ".venv", "venv",
        "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "result.json",
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not any(part in ignored_roots for part in path.relative_to(root).parts)
    }
    _assert(STUDENT_PACKAGE_FILES <= actual, "Required submission files are missing.")


def check_environment_and_fixture(root: Path) -> None:
    _check_submission_inventory(root)
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Missing regular output directory.")
    _assert({".gitkeep", *OUTPUT_FILES} <= {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}, "Required output artifacts are missing.")
    _assert(
        sys.version_info[:3] == (3, 12, 13),
        "Run the checker with the recorded Python 3.12.13 interpreter.",
    )
    _assert(np.__version__ == "2.0.2", "Install the recorded NumPy 2.0.2.")
    _assert(pd.__version__ == "3.0.5", "Install the recorded pandas 3.0.5.")
    _assert(
        _read_text(root / ".python-version", ".python-version")
        == EXPECTED_PYTHON_FILE,
        "Restore .python-version to exactly 3.12.13 and one final newline.",
    )
    _assert(
        _read_text(root / "requirements.txt", "requirements.txt")
        == EXPECTED_REQUIREMENTS,
        "Restore requirements.txt to the exact NumPy and pandas records.",
    )
    _assert(
        _read_text(root / ".gitignore", ".gitignore") == EXPECTED_GITIGNORE,
        "Restore the supplied environment, notebook-cache, and output exclusions.",
    )

    manifest = _load_json(root / "data" / "fixture.json", "data/fixture.json")
    _assert(
        manifest == EXPECTED_MANIFEST,
        "Restore the exact supplied data/fixture.json manifest.",
    )
    data_path = root / "data" / "people_raw.csv"
    _assert(data_path.is_file(), "Missing data/people_raw.csv.")
    data_bytes = data_path.read_bytes()
    _assert(
        len(data_bytes) == 570,
        "Restore data/people_raw.csv to the supplied 570-byte fixture.",
    )
    _assert(
        sha256(data_bytes).hexdigest() == EXPECTED_DATA_SHA256,
        "Restore the immutable data/people_raw.csv bytes.",
    )


def _expected_cleaned() -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "record_id": [f"R{number:03d}" for number in range(1, 12)],
            "full_name": [
                "Alice Smith",
                "Bob Jones",
                "Carla Ruiz",
                pd.NA,
                "Evan Li",
                "Fatima Noor",
                "Grace Chen",
                "Hugo Diaz",
                "Inez Park",
                "Jamie Okafor",
                "Kai Patel",
            ],
            "site": [
                "north",
                "north",
                "south",
                "south",
                "west",
                "north",
                "south",
                "west",
                "north",
                "west",
                "south",
            ],
            "status": [
                "active",
                "active",
                "pending",
                pd.NA,
                "complete",
                "active",
                "active",
                "pending",
                "complete",
                "complete",
                "pending",
            ],
            "age": [34, pd.NA, pd.NA, 45, 52, pd.NA, pd.NA, pd.NA, 39, 28, 0],
            "visit_date": [
                "2026-01-15",
                None,
                "2026-03-01",
                None,
                "2026-02-14",
                "2026-04-01",
                "2026-05-01",
                "2026-06-01",
                None,
                "2026-07-15",
                "2026-08-01",
            ],
            "needs_review": [
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
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


def check_audit(root: Path) -> None:
    audit = pd.read_csv(_artifact_path(root, "issue_audit.csv"), dtype={"issue": "string", "count": "Int64"})
    expected = pd.DataFrame(EXPECTED_ISSUES, columns=["issue", "count"]).astype({"issue": "string", "count": "Int64"})
    _assert(audit.columns.tolist() == expected.columns.tolist(), "issue_audit.csv columns differ.")
    _assert(audit["issue"].is_unique and audit["issue"].notna().all(), "Audit issues must be unique and present.")
    pd.testing.assert_frame_equal(audit.sort_values("issue").reset_index(drop=True), expected.sort_values("issue").reset_index(drop=True))


def check_foundations(root: Path) -> None:
    assert _artifact_path(root, "raw_preview.txt").read_text(encoding="utf-8") == EXPECTED_RAW_PREVIEW, "raw_preview.txt does not match the required terminal evidence."
    assert _artifact_path(root, "pipeline_summary.txt").read_text(encoding="utf-8") == EXPECTED_PIPELINE_SUMMARY, "pipeline_summary.txt does not match the required Python summary."
    summary = pd.read_csv(_artifact_path(root, "numpy_age_summary.csv"), dtype={"metric": "string", "value": "float64"})
    assert list(summary.itertuples(index=False, name=None)) == EXPECTED_NUMPY_SUMMARY, "numpy_age_summary.csv values differ."
    selection = pd.read_csv(_artifact_path(root, "pandas_selection.csv"), dtype="string")
    assert list(selection.itertuples(index=False, name=None)) == EXPECTED_PANDAS_SELECTION, "pandas_selection.csv values differ."


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
    _assert(decision["source_sha256"].fillna("").eq(EXPECTED_DATA_SHA256).all(), "Wrong source checksum.")
    _assert(decision["rows_before"].eq(12).fillna(False).all() and decision["rows_after"].eq(11).fillna(False).all(), "Wrong before/after row counts.")


def run_public_checks(root: Path = ASSIGNMENT_DIR) -> list[tuple[str, bool, str]]:
    checks = [
        ("environment and immutable fixture", check_environment_and_fixture),
        ("cumulative foundation artifacts", check_foundations),
        ("committed issue audit", check_audit),
        ("committed cleaned records", check_cleaned),
        ("committed decision log", check_decisions),
    ]
    results: list[tuple[str, bool, str]] = []
    for label, check in checks:
        try:
            check(root)
        except Exception as error:  # keep all public feedback visible in one run
            results.append((label, False, str(error)))
        else:
            results.append((label, True, ""))
    return results


def main() -> int:
    results = run_public_checks()
    for label, passed, detail in results:
        if passed:
            print(f"[PASS] {label}")
        else:
            print(f"[FIX] {label}: {detail}")
    if all(passed for _, passed, _ in results):
        print("All public checks passed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
