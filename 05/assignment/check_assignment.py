"""Run the discoverable public checks for Assignment 05.

The checker deliberately executes notebook code from fresh state in disposable
directories. Saved cell output and editable notebook assertions are not used as
evidence of correctness.
"""

from __future__ import annotations

import ast
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


ASSIGNMENT_DIR = Path(__file__).resolve().parent
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
EXPECTED_SETUP_SHA256 = (
    "c54ad0ae9d10a0681aab686cda368a46e28695dc46c1af84b5f685cc9c4dd43d"
)
EXPECTED_FINAL_SHA256 = (
    "d91e8b83bbcef6caf595893837586e3b1d1408b18bd15e1dedffd678fb69e802"
)
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
STUDENT_CODE_CELL_IDS = {
    "a05-task1-code",
    "a05-task2-decisions",
    "a05-task2-clean",
    "a05-task3-validation",
    "a05-task3-save",
}
STUDENT_MARKDOWN_CELL_IDS = {
    "a05-task1-contract",
    "a05-task2-explanation",
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
OUTPUT_FILES = (
    "issue_audit.csv",
    "cleaned_people.csv",
    "decision_log.csv",
)
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/people_raw.csv",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}
DELIVERY_FILES = {".classroom50.yaml", ".github/workflows/autograde.yaml"}


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


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source if isinstance(source, str) else ""


def _check_submission_inventory(root: Path) -> None:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(root).parts[0] != ".git"
        and path.relative_to(root).parts[0] != "output"
    }
    expected = STUDENT_PACKAGE_FILES | (actual & DELIVERY_FILES)
    _assert(actual == expected, "Remove unexpected submission files; only optional delivery metadata is allowed.")
    for relative in actual & DELIVERY_FILES:
        _assert(not (root / relative).is_symlink(), f"{relative} must be a regular delivery-owned file.")


def check_environment_and_fixture(root: Path) -> None:
    _check_submission_inventory(root)
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


def _notebook_by_id(root: Path) -> tuple[dict, dict[str, dict]]:
    notebook = _load_json(root / "assignment.ipynb", "assignment.ipynb")
    _assert(notebook.get("nbformat") == 4, "assignment.ipynb must use format 4.")
    cells = notebook.get("cells")
    _assert(isinstance(cells, list), "assignment.ipynb must contain a cell list.")
    ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    _assert(
        len(ids) == len(cells) and all(isinstance(cell_id, str) for cell_id in ids),
        "Every notebook cell must retain a stable string ID.",
    )
    _assert(len(ids) == len(set(ids)), "Notebook cell IDs must remain unique.")
    _assert(
        ids == EXPECTED_CELL_IDS,
        "Restore the supplied 14-cell order and IDs; add work inside the task cells.",
    )
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    _assert(
        kernelspec
        == {"display_name": "Python 3", "language": "python", "name": "python3"},
        "Keep the portable supplied Python 3 kernelspec.",
    )
    return notebook, {cell["id"]: cell for cell in cells}


def check_notebook_contract(root: Path) -> None:
    _, by_id = _notebook_by_id(root)
    setup = _cell_source(by_id["a05-supplied-setup"])
    final = _cell_source(by_id["a05-final-verification"])
    _assert(
        sha256(setup.encode("utf-8")).hexdigest() == EXPECTED_SETUP_SHA256,
        "Restore the supplied setup cell without editing it.",
    )
    _assert(
        sha256(final.encode("utf-8")).hexdigest() == EXPECTED_FINAL_SHA256,
        "Restore the supplied final-verification cell without editing it.",
    )

    code_source = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_CODE_CELL_IDS)
    markdown_source = "\n".join(
        _cell_source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_CELL_IDS
    )
    _assert(
        "TODO" not in code_source and "TODO" not in markdown_source,
        "Complete every TODO in the five task code cells and two explanation cells.",
    )
    try:
        tree = ast.parse(code_source)
    except SyntaxError as error:
        raise AssertionError(
            f"Task code has a syntax error at line {error.lineno}: {error.msg}."
        ) from error
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    required = {
        "audit_person_records",
        "clean_person_records",
        "validate_clean_records",
    }
    missing = sorted(required - defined)
    _assert(not missing, f"Define the required function(s): {', '.join(missing)}.")

    contract = _cell_source(by_id["a05-task1-contract"]).lower()
    required_terms = {
        "one submitted person record",
        "record_id",
        "raw",
        "clean",
        "schema",
        "sentinel",
        "duplicate",
        "missing value",
        "validation invariant",
        "provenance",
    }
    missing_terms = sorted(term for term in required_terms if term not in contract)
    _assert(
        not missing_terms,
        "Task 1 contract must explicitly introduce: " + ", ".join(missing_terms) + ".",
    )
    explanation = _cell_source(by_id["a05-task2-explanation"]).lower()
    for required_phrase in ("forward", "backward", "missing", "contract", "perfect"):
        _assert(
            required_phrase in explanation,
            f"Task 2 explanation must address {required_phrase!r}.",
        )


def check_scope(root: Path) -> None:
    _, by_id = _notebook_by_id(root)
    sources = [_cell_source(by_id[cell_id]) for cell_id in STUDENT_CODE_CELL_IDS]
    combined = "\n".join(sources)
    lowered = combined.lower()
    forbidden_fragments = {
        "/content": "Colab-only /content paths",
        "drive.mount": "Drive mounts",
        "files.upload": "manual uploads",
        "http://": "network access",
        "https://": "network access",
        "!pip": "notebook shell installation",
        "%pip": "notebook package magics",
        "%run": "notebook execution magics",
    }
    for fragment, label in forbidden_fragments.items():
        _assert(fragment not in lowered, f"Remove out-of-scope {label} from task code.")
    for line in combined.splitlines():
        _assert(
            not line.lstrip().startswith(("!", "%")),
            "Remove notebook shell commands and magics from task code.",
        )
    tree = ast.parse(combined)
    banned_attributes = {
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
    banned_names = {"eval", "exec", "round"}
    for node in ast.walk(tree):
        _assert(
            not isinstance(node, (ast.Import, ast.ImportFrom)),
            "Use only the imports in the supplied setup cell.",
        )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                _assert(
                    node.func.attr not in banned_attributes,
                    f"Remove out-of-scope `{node.func.attr}` from task code.",
                )
            if isinstance(node.func, ast.Name):
                _assert(
                    node.func.id not in banned_names,
                    f"Remove out-of-scope `{node.func.id}` from task code.",
                )


RUNNER_SOURCE = r'''import json
import pathlib
import sys

notebook_path = pathlib.Path(sys.argv[1])
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
for position, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    exec(compile(source, f"{notebook_path}#cell-{position}", "exec"), namespace)
'''


def _execute_notebook(root: Path, working_directory: Path) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONWARNINGS"] = "error"
    result = subprocess.run(
        [sys.executable, "-c", RUNNER_SOURCE, str(root / "assignment.ipynb")],
        cwd=working_directory,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        tail = "\n".join(detail[-12:]) if detail else result.stdout.strip()
        raise AssertionError(
            "Fresh notebook execution failed. Restart, run all, and fix the first "
            f"error.\n{tail}"
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


def _read_artifacts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = root / "output"
    for name in OUTPUT_FILES:
        _assert(
            (output / name).is_file(),
            f"Missing output/{name}; restart the notebook and run all cells.",
        )
    audit = pd.read_csv(
        output / "issue_audit.csv",
        dtype={"issue": "string", "count": "Int64"},
    )
    cleaned = pd.read_csv(
        output / "cleaned_people.csv",
        dtype={
            "record_id": "string",
            "full_name": "string",
            "site": "string",
            "status": "string",
            "age": "Int64",
            "visit_date": "string",
            "needs_review": "boolean",
        },
    )
    lexical = cleaned["visit_date"].str.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}", na=True
    )
    _assert(bool(lexical.all()), "cleaned_people.csv contains a noncontract date token.")
    cleaned["visit_date"] = pd.to_datetime(
        cleaned["visit_date"], format="%Y-%m-%d", errors="coerce"
    ).astype("datetime64[us]")
    decision = pd.read_csv(
        output / "decision_log.csv",
        dtype={
            "field": "string",
            "issue": "string",
            "action": "string",
            "reason": "string",
            "source": "string",
            "source_sha256": "string",
            "rows_before": "Int64",
            "rows_after": "Int64",
        },
    )
    return audit, cleaned, decision


def check_artifacts(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Missing regular output/ directory.")
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(actual == {".gitkeep", *OUTPUT_FILES}, "Keep exactly .gitkeep and the three required output CSVs.")
    audit, cleaned, decision = _read_artifacts(root)
    expected_audit = pd.DataFrame(EXPECTED_ISSUES, columns=["issue", "count"])
    expected_audit["issue"] = expected_audit["issue"].astype("string")
    expected_audit["count"] = expected_audit["count"].astype("Int64")
    pd.testing.assert_frame_equal(audit, expected_audit)
    pd.testing.assert_frame_equal(cleaned, _expected_cleaned())

    expected_columns = [
        "field",
        "issue",
        "action",
        "reason",
        "source",
        "source_sha256",
        "rows_before",
        "rows_after",
    ]
    _assert(
        decision.columns.tolist() == expected_columns,
        "decision_log.csv must contain the eight required columns in order.",
    )
    _assert(len(decision) == 8, "decision_log.csv must contain eight decisions.")
    observed_decisions = list(
        decision[["field", "issue", "action"]].itertuples(index=False, name=None)
    )
    _assert(
        observed_decisions == EXPECTED_DECISIONS,
        "decision_log.csv field, issue, and action rows must match the supplied contract.",
    )
    _assert(
        bool(decision["reason"].str.strip().ne("").all()),
        "Every decision-log reason must be nonempty and grounded in the data purpose.",
    )
    _assert(
        decision["source"].eq("data/people_raw.csv").all(),
        "Repeat source=data/people_raw.csv on every decision-log row.",
    )
    _assert(
        decision["source_sha256"].eq(EXPECTED_DATA_SHA256).all(),
        "Repeat the verified source checksum on every decision-log row.",
    )
    _assert(
        decision["rows_before"].eq(12).all()
        and decision["rows_after"].eq(11).all(),
        "Record rows_before=12 and rows_after=11 on every decision-log row.",
    )


def _copy_submission(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {"_grader_selftest", ".venv", ".ipynb_checkpoints", "__pycache__"}
        return ignored.intersection(names)

    shutil.copytree(source, destination, ignore=ignore)


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    artifacts = {}
    for name in OUTPUT_FILES:
        path = root / "output" / name
        _assert(
            path.is_file(),
            f"Missing output/{name}; restart the notebook and run all cells.",
        )
        artifacts[name] = path.read_bytes()
    return artifacts


def check_fresh_relocated_execution(root: Path) -> None:
    committed = _artifact_bytes(root)
    with tempfile.TemporaryDirectory(prefix="a05-public-") as temporary:
        temporary_path = Path(temporary)

        flat = temporary_path / "renamed-assignment"
        _copy_submission(root, flat)
        flat_output = flat / "output"
        flat_output.mkdir(exist_ok=True)
        for name in OUTPUT_FILES:
            (flat_output / name).write_text("stale,artifact\n", encoding="utf-8")
        deep_working_directory = flat / "work" / "deep"
        deep_working_directory.mkdir(parents=True)
        _execute_notebook(flat, deep_working_directory)
        check_artifacts(flat)
        _assert(
            _artifact_bytes(flat) == committed,
            "Committed output files differ from a fresh run; restart, run all, and commit them.",
        )

        course_root = temporary_path / "relocated-course"
        nested = course_root / "05" / "assignment"
        nested.parent.mkdir(parents=True)
        _copy_submission(root, nested)
        for name in OUTPUT_FILES:
            path = nested / "output" / name
            if path.exists():
                path.unlink()
        _execute_notebook(nested, course_root)
        check_artifacts(nested)
        _assert(
            _artifact_bytes(nested) == committed,
            "A relocated course-tree run produced different artifacts.",
        )


def run_public_checks(root: Path = ASSIGNMENT_DIR) -> list[tuple[str, bool, str]]:
    checks = [
        ("environment and immutable fixture", check_environment_and_fixture),
        ("notebook contract and explanations", check_notebook_contract),
        ("Lecture 05 scope", check_scope),
        ("committed output artifacts", check_artifacts),
        ("fresh, stale-output, and relocated execution", check_fresh_relocated_execution),
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
