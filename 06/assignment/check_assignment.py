"""Dependency-free public structural checks for Assignment 06.

This checker does not execute notebook code and does not award a grade. The
independent instructor grader fresh-executes a disposable notebook copy.
"""

from __future__ import annotations

import ast
import csv
from hashlib import sha256
from importlib import metadata
import json
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
    "README.md": "2a439cc655b0447c6bd0b93be0fa998c8b4e08e82ca356f4b54fe168a0da4b1d",
    "PLATFORM_CHECK.md": "ebbca4c26b9701f231a6ef73a295c1619538db24cdd929828b8e3581bc8852d2",
    "data/fixture.json": "12b8d3375e4895b6cb443c156794dc9598f5598e64920d2f2818b50883a99f55",
}
EXPECTED_CELL_IDS = [
    "a06-header",
    "a06-setup",
    "a06-data-contract",
    "a06-load",
    "a06-task1-contract",
    "a06-contract-values",
    "a06-key-checks",
    "a06-duplicate-failure",
    "a06-task1-functions",
    "a06-task1-run",
    "a06-task1-save",
    "a06-task2-contract",
    "a06-stack-function",
    "a06-stack-run",
    "a06-schema-drift",
    "a06-align-function",
    "a06-align-run",
    "a06-task2-save",
    "a06-task3-contract",
    "a06-reshape-functions",
    "a06-reshape-run",
    "a06-duplicate-pivot",
    "a06-task3-save",
    "a06-reflection",
    "a06-final-verify",
]
EXPECTED_CELL_TYPES = {
    "a06-header": "markdown",
    "a06-setup": "code",
    "a06-data-contract": "markdown",
    "a06-load": "code",
    "a06-task1-contract": "markdown",
    "a06-contract-values": "code",
    "a06-key-checks": "code",
    "a06-duplicate-failure": "code",
    "a06-task1-functions": "code",
    "a06-task1-run": "code",
    "a06-task1-save": "code",
    "a06-task2-contract": "markdown",
    "a06-stack-function": "code",
    "a06-stack-run": "code",
    "a06-schema-drift": "code",
    "a06-align-function": "code",
    "a06-align-run": "code",
    "a06-task2-save": "code",
    "a06-task3-contract": "markdown",
    "a06-reshape-functions": "code",
    "a06-reshape-run": "code",
    "a06-duplicate-pivot": "code",
    "a06-task3-save": "code",
    "a06-reflection": "markdown",
    "a06-final-verify": "code",
}
PROTECTED_CELL_SHA256 = {
    "a06-header": "1c728cf8f59fd331446ab73f55086e782d4399685985a1969d5f9839ad41f3cd",
    "a06-setup": "68f227f40a9ed45e80664eaf939b2b270326e483389dd547abe76fa363bdec44",
    "a06-data-contract": "0265d9faa5bb57d9d5061a4ca68426982deb244a18b67506ec33cca4f1d91270",
    "a06-final-verify": "d3dd35f4762145d5f012c7798bc9072f2d8d2cc195b74e772c931dd18979b4c9",
}
STUDENT_MARKDOWN_IDS = {
    "a06-task1-contract",
    "a06-task2-contract",
    "a06-task3-contract",
    "a06-reflection",
}
STUDENT_CODE_IDS = set(EXPECTED_CELL_IDS) - STUDENT_MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
REQUIRED_FUNCTIONS = {
    "select_current_stations",
    "validated_station_merge",
    "stack_specimen_partitions",
    "align_specimen_features",
    "wide_to_long_scores",
    "long_to_wide_scores",
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
ARTIFACTS = {
    "specimen_merge_audit.csv": (
        7,
        ["specimen_id", "collector_id", "collection_number", "station_code", "material", "mass_g", "station_name", "region", "_merge"],
        "1bc33aeecbae2483e314399784bbcaf8b8847798fe3ca5b7662908053615e98c",
    ),
    "combined_specimens.csv": (
        7,
        ["specimen_id", "collector_id", "collection_number", "station_code", "material", "mass_g", "source_partition"],
        "78cbd883bea393fb84d699cdb9923a9d71d7c045d002eebe88cc84c9da61c666",
    ),
    "aligned_features.csv": (
        4,
        ["specimen_id", "mass_g", "review_score"],
        "19cb5d07f7ae51ce0347876802a44eadf48490076bb24dbdcae547d9388775e7",
    ),
    "sensor_scores_long.csv": (
        8,
        ["sensor_id", "station_code", "measurement_label", "value"],
        "989affb14d49ecd0e144e23a6b53ab4a093edd6211656390144869ecaa3126dd",
    ),
    "sensor_scores_round_trip.csv": (
        4,
        ["sensor_id", "station_code", "baseline_value", "followup_value"],
        "6eb9bfb9561fc7c55708bc0038b77b99e6d383f85843dff3e735c5962abe8701",
    ),
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
    *(f"data/{record['path']}" for record in FIXTURE_MANIFEST["files"]),
}
DELIVERY_FILES = {".classroom50.yaml", ".github/workflows/autograde.yaml"}
BANNED_ATTRIBUTES = {
    "agg",
    "aggregate",
    "bfill",
    "crosstab",
    "drop_duplicates",
    "dropna",
    "ewm",
    "expanding",
    "ffill",
    "fillna",
    "groupby",
    "interpolate",
    "join",
    "pivot_table",
    "plot",
    "replace",
    "resample",
    "rolling",
    "to_datetime",
    "transform",
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


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source if isinstance(source, str) else ""


def _check_submission_inventory() -> None:
    actual = {
        path.relative_to(ASSIGNMENT_DIR).as_posix()
        for path in ASSIGNMENT_DIR.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(ASSIGNMENT_DIR).parts[0] != ".git"
        and path.relative_to(ASSIGNMENT_DIR).parts[0] != "output"
    }
    expected = STUDENT_PACKAGE_FILES | (actual & DELIVERY_FILES)
    _assert(actual == expected, "Remove unexpected submission files; only optional delivery metadata is allowed.")
    for relative in actual & DELIVERY_FILES:
        _assert(not (ASSIGNMENT_DIR / relative).is_symlink(), f"{relative} must be a regular delivery-owned file.")


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
    _assert(actual_names == expected_names, "Restore the exact six-file fixture inventory.")
    for record in FIXTURE_MANIFEST["files"]:
        path = ASSIGNMENT_DIR / "data" / record["path"]
        data = path.read_bytes()
        _assert(data.endswith(b"\n") and b"\r" not in data, f"Restore LF/final-newline bytes in data/{path.name}.")
        _assert(sha256(data).hexdigest() == record["sha256"], f"Restore immutable data/{path.name}.")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        _assert(rows and rows[0] == record["columns"], f"Wrong columns in data/{path.name}.")
        _assert(len(rows) - 1 == record["row_count"], f"Wrong row count in data/{path.name}.")


def _load_notebook() -> tuple[dict, dict[str, dict]]:
    notebook = _read_json(ASSIGNMENT_DIR / "assignment.ipynb", "assignment.ipynb")
    cells = notebook.get("cells")
    _assert(notebook.get("nbformat") == 4 and isinstance(cells, list), "Keep notebook format 4 with a cell list.")
    _assert(len(cells) == 25 and all(isinstance(cell, dict) for cell in cells), "Restore the exact 25-cell notebook.")
    ids = [cell.get("id") for cell in cells]
    _assert(ids == EXPECTED_CELL_IDS and len(ids) == len(set(ids)), "Restore the supplied cell IDs and order.")
    for cell in cells:
        _assert(cell.get("cell_type") == EXPECTED_CELL_TYPES[cell["id"]], f"Restore the type of {cell['id']}.")
    kernelspec = notebook.get("metadata", {}).get("kernelspec")
    _assert(kernelspec == {"display_name": "Python 3", "language": "python", "name": "python3"}, "Restore the portable Python 3 kernelspec.")
    return notebook, {cell["id"]: cell for cell in cells}


def _keyword_literal(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _handler_assigns(handler: ast.ExceptHandler, name: str, value) -> bool:
    for node in ast.walk(handler):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if node.value.value != value:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return True
    return False


def check_notebook() -> None:
    _, by_id = _load_notebook()
    for cell_id, expected in PROTECTED_CELL_SHA256.items():
        observed = sha256(_cell_source(by_id[cell_id]).encode()).hexdigest()
        _assert(observed == expected, f"Restore protected notebook cell {cell_id}.")

    student_markdown = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_IDS)
    student_code = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_CODE_IDS)
    _assert("TODO" not in student_markdown + student_code, "Complete every TODO in the student cells.")
    _assert("pass" not in {line.strip() for line in student_code.splitlines()}, "Replace every scaffold pass statement.")
    try:
        tree = ast.parse(student_code)
    except SyntaxError as error:
        raise AssertionError(f"Student code has a syntax error: {error}") from error
    function_nodes = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    _assert(REQUIRED_FUNCTIONS.issubset(function_nodes), "Define all six required reusable functions.")
    for node in ast.walk(tree):
        _assert(not isinstance(node, (ast.Import, ast.ImportFrom)), "Do not add imports to student cells.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            _assert(node.func.attr not in BANNED_ATTRIBUTES, f"Out-of-scope API used: {node.func.attr}().")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            _assert(node.func.id not in {"eval", "exec", "__import__"}, f"Forbidden call used: {node.func.id}().")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            absolute = value.startswith(("/", "~")) or (
                len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"}
            )
            _assert(not absolute, f"Remove absolute path literal: {value!r}.")
    lowered = student_code.lower()
    for fragment in ("/content", "drive.mount", "files.upload", "http://", "https://", "urlopen", "requests."):
        _assert(fragment not in lowered, f"Remove nonportable or remote code: {fragment}.")
    for line in student_code.splitlines():
        _assert(not line.lstrip().startswith(("!", "%")), "Remove notebook magic or shell commands.")

    load_tree = ast.parse(_cell_source(by_id["a06-load"]))
    read_calls = [
        node for node in ast.walk(load_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_csv"
    ]
    _assert(len(read_calls) == 6, "Load each of the six protected fixtures exactly once with pd.read_csv.")
    _assert(not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "DataFrame"
        for node in ast.walk(load_tree)
    ), "Do not embed a replacement fixture in the load cell.")
    load_source = _cell_source(by_id["a06-load"])
    for filename in (record["path"] for record in FIXTURE_MANIFEST["files"]):
        _assert(filename in load_source, f"Load protected fixture {filename} from DATA_DIR.")

    failure_tree = ast.parse(_cell_source(by_id["a06-duplicate-failure"]))
    _assert(not any(isinstance(node, ast.Raise) for node in ast.walk(failure_tree)), "Do not manufacture the merge failure.")
    merge_calls = [
        node for node in ast.walk(failure_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "merge"
    ]
    _assert(any(
        _keyword_literal(call, "on") == "station_code"
        and _keyword_literal(call, "how") == "left"
        and _keyword_literal(call, "validate") == "many_to_one"
        for call in merge_calls
    ), "The duplicate-key cell must attempt the explicit validated left merge.")
    merge_handlers = [
        node for node in ast.walk(failure_tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Attribute)
        and node.type.attr == "MergeError"
    ]
    _assert(any(
        isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Attribute)
        and node.type.attr == "MergeError"
        for node in ast.walk(failure_tree)
    ), "Catch only pd.errors.MergeError in the duplicate-key cell.")
    _assert(any(
        _handler_assigns(handler, "duplicate_contract_failed", True)
        for handler in merge_handlers
    ), "Set duplicate_contract_failed only from the caught pandas merge failure.")

    pivot_tree = ast.parse(_cell_source(by_id["a06-duplicate-pivot"]))
    _assert(not any(isinstance(node, ast.Raise) for node in ast.walk(pivot_tree)), "Do not manufacture the pivot failure.")
    _assert(any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "long_to_wide_scores"
        for node in ast.walk(pivot_tree)
    ), "The duplicate-key cell must call long_to_wide_scores.")
    pivot_handlers = [
        node for node in ast.walk(pivot_tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "ValueError"
    ]
    _assert(any(
        isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) and node.type.id == "ValueError"
        for node in ast.walk(pivot_tree)
    ), "Catch the natural ValueError in the duplicate-pivot cell.")
    _assert(any(
        _handler_assigns(handler, "duplicate_pivot_failed", True)
        for handler in pivot_handlers
    ), "Set duplicate_pivot_failed only from the caught pivot failure.")
    _assert(not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "duplicated"
        for node in ast.walk(function_nodes["long_to_wide_scores"])
    ), "Let structural pivot reject duplicate long keys; do not pre-delete or pre-empt them.")


def check_artifacts() -> None:
    output = ASSIGNMENT_DIR / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Missing regular output directory.")
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(actual == {".gitkeep", *ARTIFACTS}, "Create exactly .gitkeep and the five required CSV artifacts in output/.")
    for name, (row_count, columns, expected_digest) in ARTIFACTS.items():
        path = output / name
        data = path.read_bytes()
        _assert(data.endswith(b"\n") and b"\r" not in data, f"Write output/{name} with LF and a final newline.")
        _assert(sha256(data).hexdigest() == expected_digest, f"output/{name} does not match the exact canonical result.")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        _assert(rows and rows[0] == columns, f"Wrong ordered columns in output/{name}.")
        _assert(len(rows) - 1 == row_count, f"Wrong row count in output/{name}.")


def main() -> int:
    checks = (
        ("environment and protected files", check_environment_and_protected_files),
        ("fixture integrity", check_fixtures),
        ("notebook contract", check_notebook),
        ("generated artifacts", check_artifacts),
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
        print("Assignment 06 is not ready. Fix the messages, restart and run all cells, then check again.")
        return 1
    print("All public checks passed. Instructor review may run stronger checks separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
