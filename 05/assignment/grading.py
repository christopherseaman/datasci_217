"""Independent central-grader reference for Assignment 05.

This module intentionally does not import the student-editable public checker.
Production grading-service wiring is external to this repository.
"""

from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


EXPECTED_SHA256 = "d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b"
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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
    tests: list[dict] = []
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
        "score": score,
        "max-score": 85,
        "tests": tests,
    }

def _format_result(result: dict) -> str:
    return "\n".join(
        f"[{'PASS' if test['passed'] else 'FAIL'}] {test['test-name']}: "
        f"{test['score']}/{test['max-score']}"
        + (f" ({test['detail']})" if test.get("detail") else "")
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
