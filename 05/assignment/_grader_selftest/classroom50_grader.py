# /// script
# requires-python = "==3.12.13"
# dependencies = ["numpy==2.0.2", "pandas==3.0.3"]
# ///

"""Independent central-grader reference for Assignment 05.

This module intentionally does not import the student-editable public checker.
Production Classroom 50 wiring is external to this repository.
"""

from __future__ import annotations

import ast
import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


EXPECTED_SHA256 = "d13dc9676519c81729b33d53ffc2e8fec92e645c6978af7ebf325fcd7147753b"
EXPECTED_SETUP_SHA256 = "46ce6c927699603018f448982b1162c33e939edb1358688e977b9eb1e9c65ffa"
EXPECTED_FINAL_SHA256 = "d91e8b83bbcef6caf595893837586e3b1d1408b18bd15e1dedffd678fb69e802"
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
EXPECTED_CELL_IDS = [
    "a05-header",
    "a05-supplied-setup",
    "a05-task1-heading",
    "a05-task1-contract",
    "a05-task1-code",
    "a05-task2-heading",
    "a05-task2-decisions",
    "a05-task2-explanation",
    "a05-task2-clean",
    "a05-task3-heading",
    "a05-task3-validation",
    "a05-task3-save",
    "a05-final-heading",
    "a05-final-verification",
]
TASK_CODE_IDS = {
    "a05-task1-code",
    "a05-task2-decisions",
    "a05-task2-clean",
    "a05-task3-validation",
    "a05-task3-save",
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
OUTPUT_FILES = ("issue_audit.csv", "cleaned_people.csv", "decision_log.csv")
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/people_raw.csv",
}
DELIVERY_FILES = {".classroom50.yaml", ".github/workflows/autograde.yaml"}
REQUIRED_CONTEXT_ENV = {
    "classroom": "CLASSROOM",
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
            "missing required Classroom50 context: " + ", ".join(missing)
        )
    context["review"] = os.environ.get("REVIEW_URL", "").strip() or context["commit"]
    context["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return context


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _load_notebook(root: Path) -> tuple[dict, dict[str, dict]]:
    path = root / "assignment.ipynb"
    _assert(path.is_file(), "missing assignment.ipynb")
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError("assignment.ipynb is not valid UTF-8 notebook JSON") from error
    cells = notebook.get("cells")
    _assert(notebook.get("nbformat") == 4 and isinstance(cells, list), "invalid notebook format")
    ids = [cell.get("id") for cell in cells]
    _assert(ids == EXPECTED_CELL_IDS, "supplied cell IDs or order changed")
    _assert(len(ids) == len(set(ids)), "cell IDs are not unique")
    kernelspec = notebook.get("metadata", {}).get("kernelspec")
    _assert(
        kernelspec == {"display_name": "Python 3", "language": "python", "name": "python3"},
        "portable kernelspec changed",
    )
    return notebook, {cell["id"]: cell for cell in cells}


def _check_static_contract(root: Path) -> None:
    _assert(sys.version_info[:3] == (3, 12, 13), "grader must use Python 3.12.13")
    _assert(np.__version__ == "2.0.2", "grader must use NumPy 2.0.2")
    _assert(pd.__version__ == "3.0.3", "grader must use pandas 3.0.3")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(root).parts[0] != ".git"
        and path.relative_to(root).parts[0] != "output"
    }
    expected_files = STUDENT_PACKAGE_FILES | (actual_files & DELIVERY_FILES)
    _assert(actual_files == expected_files, "student package inventory differs")
    _assert(not any((root / relative).is_symlink() for relative in actual_files & DELIVERY_FILES), "delivery metadata must be regular files")
    _assert((root / ".python-version").read_text() == "3.12.13\n", "wrong Python record")
    _assert(
        (root / "requirements.txt").read_text() == "numpy==2.0.2\npandas==3.0.3\n",
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
    _assert(actual_outputs == {".gitkeep", *OUTPUT_FILES}, "submission output inventory differs")

    _, by_id = _load_notebook(root)
    _assert(
        sha256(_cell_source(by_id["a05-supplied-setup"]).encode()).hexdigest()
        == EXPECTED_SETUP_SHA256,
        "supplied setup cell changed",
    )
    _assert(
        sha256(_cell_source(by_id["a05-final-verification"]).encode()).hexdigest()
        == EXPECTED_FINAL_SHA256,
        "supplied final cell changed",
    )
    task_source = "\n".join(_cell_source(by_id[cell_id]) for cell_id in TASK_CODE_IDS)
    explanations = _cell_source(by_id["a05-task1-contract"]) + _cell_source(
        by_id["a05-task2-explanation"]
    )
    _assert("TODO" not in task_source + explanations, "unfinished TODO remains")
    tree = ast.parse(task_source)
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    _assert(
        {
            "audit_person_records",
            "clean_person_records",
            "validate_clean_records",
        }.issubset(functions),
        "required function missing",
    )
    lowered = task_source.lower()
    for fragment in ("/content", "drive.mount", "files.upload", "http://", "https://"):
        _assert(fragment not in lowered, f"forbidden scope fragment: {fragment}")
    for line in task_source.splitlines():
        _assert(not line.lstrip().startswith(("!", "%")), "notebook magic or shell line used")
    banned = {
        "agg",
        "aggregate",
        "bfill",
        "concat",
        "cut",
        "ffill",
        "get_dummies",
        "groupby",
        "join",
        "merge",
        "melt",
        "pivot",
        "pivot_table",
        "plot",
        "qcut",
        "round",
        "transform",
    }
    for node in ast.walk(tree):
        _assert(not isinstance(node, (ast.Import, ast.ImportFrom)), "student import used")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            _assert(node.func.attr not in banned, f"forbidden API used: {node.func.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            _assert(node.func.id not in {"eval", "exec", "round"}, f"forbidden call: {node.func.id}")
    for name in OUTPUT_FILES:
        _assert((root / "output" / name).is_file(), f"missing committed output/{name}")


RUNNER = r'''import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
extra_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
notebook = json.loads(path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
for position, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", "")
    source = "".join(source) if isinstance(source, list) else source
    exec(compile(source, f"{path}#cell-{position}", "exec"), namespace)
if extra_path is not None:
    source = extra_path.read_text(encoding="utf-8")
    exec(compile(source, str(extra_path), "exec"), namespace)
'''


def _execute(root: Path, cwd: Path, extra_source: str | None = None) -> None:
    command = [sys.executable, "-c", RUNNER, str(root / "assignment.ipynb")]
    extra_path = root / "__grader_assertions.py"
    if extra_source is not None:
        extra_path.write_text(extra_source, encoding="utf-8")
        command.append(str(extra_path))
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONWARNINGS"] = "error"
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if extra_path.exists():
        extra_path.unlink()
    if result.returncode:
        lines = result.stderr.strip().splitlines()
        raise AssertionError("fresh execution failed: " + " | ".join(lines[-8:]))


def _copy_submission(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {"_grader_selftest", ".venv", "__pycache__", ".ipynb_checkpoints"}.intersection(names)

    shutil.copytree(source, destination, ignore=ignore)


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


def _read_outputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = root / "output"
    audit = pd.read_csv(output / "issue_audit.csv", dtype={"issue": "string", "count": "Int64"})
    clean = pd.read_csv(
        output / "cleaned_people.csv",
        dtype={
            "record_id": "string", "full_name": "string", "site": "string",
            "status": "string", "age": "Int64", "visit_date": "string",
            "needs_review": "boolean",
        },
    )
    lexical = clean["visit_date"].str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", na=True)
    _assert(bool(lexical.all()), "noncontract exported date")
    clean["visit_date"] = pd.to_datetime(
        clean["visit_date"], format="%Y-%m-%d", errors="coerce"
    ).astype("datetime64[us]")
    decision = pd.read_csv(
        output / "decision_log.csv",
        dtype={
            "field": "string", "issue": "string", "action": "string", "reason": "string",
            "source": "string", "source_sha256": "string", "rows_before": "Int64",
            "rows_after": "Int64",
        },
    )
    return audit, clean, decision


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {name: (root / "output" / name).read_bytes() for name in OUTPUT_FILES}


ALTERNATE_TABLE_SOURCE = r'''
alternate = pd.DataFrame(
    [
        ['X100', ' Ann ', ' North ', 'Active', '12', '2026-01-01'],
        ['X100', 'ANN', 'north', 'active', '12', '2026-01-01'],
        ['X200', 'Bee', 'south', 'pending', 'twelve', '2026-01-02'],
        ['X300', 'Cee', 'west', 'complete', '7.5', '2026-1-03'],
        ['X400', 'Dee', 'west', 'complete', '999', '2026-02-30'],
        ['X400', 'Dee', 'west', 'complete', '999', '2026-02-30'],
        ['X300', 'Cee', 'south', 'pending', 'unknown', ''],
    ],
    columns=EXPECTED_RAW_COLUMNS,
    index=[101, 205, 309, 412, 518, 619, 723],
)
alternate_snapshot = alternate.copy(deep=True)
'''

TASK1_ALTERNATES = ALTERNATE_TABLE_SOURCE + r'''
alternate_audit = audit_person_records(alternate)
assert alternate.equals(alternate_snapshot), 'audit mutated a nondefault-index input'
alternate_counts = dict(zip(alternate_audit['issue'], alternate_audit['count'], strict=True))
assert alternate_counts == {
    'schema mismatch': 0,
    'empty full-name tokens': 0,
    'empty date tokens': 1,
    'age sentinel tokens': 1,
    'status sentinel tokens': 0,
    'age parse failures': 1,
    'numeric but noninteger age values': 1,
    'age values outside 0 through 120': 2,
    'date parse failures': 3,
    'rows in exact duplicate sets': 2,
    'rows with repeated candidate IDs': 6,
    'site values needing format normalization': 1,
    'status values needing format normalization': 1,
    'unexpected site values': 0,
    'unexpected non-sentinel status values': 0,
}
reordered = alternate.loc[:, list(reversed(EXPECTED_RAW_COLUMNS))]
assert int(audit_person_records(reordered).set_index('issue').loc['schema mismatch', 'count']) == 1
with_extra = alternate.assign(unexpected='value')
assert int(audit_person_records(with_extra).set_index('issue').loc['schema mismatch', 'count']) == 1
'''

TASK2_ALTERNATES = ALTERNATE_TABLE_SOURCE + r'''
alternate_clean = clean_person_records(alternate)
assert alternate.equals(alternate_snapshot), 'cleaning mutated its input'
assert alternate_clean['record_id'].tolist() == ['X100', 'X100', 'X200', 'X300', 'X400', 'X300']
assert len(alternate_clean) == len(alternate) - int(alternate.duplicated(keep='first').sum())
assert alternate_clean.loc[:1, 'full_name'].tolist() == ['Ann', 'Ann']
assert alternate_clean.loc[:1, 'site'].tolist() == ['north', 'north']
assert alternate_clean.loc[2:4, 'age'].isna().all()
assert alternate_clean.loc[3:4, 'visit_date'].isna().all()
repeat_clean = clean_person_records(alternate)
pd.testing.assert_frame_equal(alternate_clean, repeat_clean)
unique_alternate = alternate.drop(index=[205, 723])
unique_snapshot = unique_alternate.copy(deep=True)
unique_clean = clean_person_records(unique_alternate)
assert unique_alternate.equals(unique_snapshot)
assert bool(validate_clean_records(unique_alternate, unique_snapshot, unique_clean).all())
'''

TASK3_ALTERNATES = ALTERNATE_TABLE_SOURCE + r'''
alternate_clean = clean_person_records(alternate)
duplicate_validation = validate_clean_records(alternate, alternate_snapshot, alternate_clean)
assert not bool(duplicate_validation['candidate identifier unique'])
assert bool(duplicate_validation.drop('candidate identifier unique').all())
unique_alternate = alternate.drop(index=[205, 723])
unique_snapshot = unique_alternate.copy(deep=True)
unique_clean = clean_person_records(unique_alternate)
assert bool(validate_clean_records(unique_alternate, unique_snapshot, unique_clean).all())
wrong_age_dtype = unique_clean.copy(deep=True)
wrong_age_dtype['age'] = wrong_age_dtype['age'].astype('Float64')
assert not bool(validate_clean_records(unique_alternate, unique_snapshot, wrong_age_dtype)['age dtype Int64'])
wrong_review = unique_clean.copy(deep=True)
wrong_review['needs_review'] = ~wrong_review['needs_review']
assert not bool(validate_clean_records(unique_alternate, unique_snapshot, wrong_review)['review flag exact'])
'''


def _task1_checks(flat: Path, cwd: Path) -> None:
    audit, _, _ = _read_outputs(flat)
    expected = pd.DataFrame(EXPECTED_ISSUES, columns=["issue", "count"])
    expected["issue"] = expected["issue"].astype("string")
    expected["count"] = expected["count"].astype("Int64")
    pd.testing.assert_frame_equal(audit, expected)
    _execute(flat, cwd, TASK1_ALTERNATES)


def _task2_checks(flat: Path, cwd: Path) -> None:
    _, cleaned, decision = _read_outputs(flat)
    pd.testing.assert_frame_equal(cleaned, _expected_cleaned())
    _assert(cleaned["needs_review"].sum() == 7, "wrong review queue count")
    _assert(len(decision) == 8, "wrong decision count")
    observed = list(decision[["field", "issue", "action"]].itertuples(index=False, name=None))
    _assert(observed == EXPECTED_DECISIONS, "decision contract rows differ")
    _assert(bool(decision["reason"].str.strip().ne("").all()), "empty decision reason")
    _execute(flat, cwd, TASK2_ALTERNATES)


def _task3_checks(source: Path, flat: Path, cwd: Path, temporary: Path) -> None:
    _execute(flat, cwd, TASK3_ALTERNATES)
    _, _, decision = _read_outputs(flat)
    _assert(decision.columns.tolist() == [
        "field", "issue", "action", "reason", "source", "source_sha256",
        "rows_before", "rows_after",
    ], "wrong decision-log schema")
    _assert(decision["source"].eq("data/people_raw.csv").all(), "wrong source provenance")
    _assert(decision["source_sha256"].eq(EXPECTED_SHA256).all(), "wrong checksum provenance")
    _assert(decision["rows_before"].eq(12).all() and decision["rows_after"].eq(11).all(), "wrong row evidence")
    _assert(_artifact_bytes(flat) == _artifact_bytes(source), "committed outputs differ from fresh outputs")

    course_root = temporary / "relocated-course"
    nested = course_root / "05" / "assignment"
    nested.parent.mkdir(parents=True)
    _copy_submission(source, nested)
    for name in OUTPUT_FILES:
        path = nested / "output" / name
        if path.exists():
            path.unlink()
    _execute(nested, course_root)
    _assert(_artifact_bytes(nested) == _artifact_bytes(flat), "relocated deleted-output run differs")


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
    """Grade one local submission and return a Classroom 50 result object."""

    context = _context()
    root = Path(submission_root).resolve()
    tests: list[dict] = []
    try:
        _check_static_contract(root)
    except Exception as error:
        for name, points in (
            ("Task 1 automated", 25),
            ("Task 2 automated", 35),
            ("Task 3 automated", 25),
        ):
            tests.append(_result_test(name, points, error))
    else:
        with tempfile.TemporaryDirectory(prefix="a05-central-") as temporary_name:
            temporary = Path(temporary_name)
            flat = temporary / "renamed-submission"
            _copy_submission(root, flat)
            output = flat / "output"
            output.mkdir(exist_ok=True)
            for name in OUTPUT_FILES:
                (output / name).write_text("stale,artifact\n", encoding="utf-8")
            cwd = flat / "arbitrary" / "deep"
            cwd.mkdir(parents=True)
            try:
                _execute(flat, cwd)
            except Exception as error:
                for name, points in (
                    ("Task 1 automated", 25),
                    ("Task 2 automated", 35),
                    ("Task 3 automated", 25),
                ):
                    tests.append(_result_test(name, points, error))
            else:
                for name, points, check in (
                    ("Task 1 automated", 25, lambda: _task1_checks(flat, cwd)),
                    ("Task 2 automated", 35, lambda: _task2_checks(flat, cwd)),
                    (
                        "Task 3 automated",
                        25,
                        lambda: _task3_checks(root, flat, cwd, temporary),
                    ),
                ):
                    try:
                        check()
                    except Exception as error:
                        tests.append(_result_test(name, points, error))
                    else:
                        tests.append(_result_test(name, points, None))

    score = sum(test["score"] for test in tests)
    return {
        "schema": "classroom50/result/v1",
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
