"""Dependency-free public structural checks for Assignment 08.

This checker reads the package, notebook source, and required CSVs. It does not
execute notebook code, trust stored output, award points, or judge explanations.
The independent instructor grader fresh-executes a disposable notebook copy.
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
    ".ipynb_checkpoints/\n"
    "__pycache__/\n"
    "*.py[cod]\n"
    ".pytest_cache/\n"
    ".venv/\n"
    "venv/\n"
)
PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    "README.md": "1ae599730f86fad8b1906d46aa3bc2ed7096fd8adb89af84394d9f5fbba2e0f3",
    "PLATFORM_CHECK.md": "b12b67322be29e1bfc641de85e82f7216f4ac52ef6a1ef738670d87114749281",
    "data/fixture.json": "b2fee1c48fb678b81318d2f085c42e2f9b480bd6c4eed6f07ef118b9bfd70860",
    "data/support_requests.csv": "a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6",
}
EXPECTED_CELL_IDS = [
    "a08-header", "a08-setup", "a08-terms-data", "a08-load",
    "a08-task1-contract", "a08-task1-values", "a08-count-function",
    "a08-task1-run", "a08-task1-save", "a08-task1-explain",
    "a08-task2-prompt", "a08-center-summary-function",
    "a08-context-function", "a08-two-key-function", "a08-task2-run",
    "a08-task2-save", "a08-task2-explain", "a08-task3-prompt",
    "a08-pivot-values", "a08-pivot-function", "a08-task3-run",
    "a08-task3-save", "a08-task3-explain", "a08-synthesis",
    "a08-final-verify",
]
MARKDOWN_IDS = {
    "a08-header", "a08-terms-data", "a08-task1-contract",
    "a08-task1-explain", "a08-task2-prompt", "a08-task2-explain",
    "a08-task3-prompt", "a08-task3-explain", "a08-synthesis",
}
EXPECTED_CELL_TYPES = {
    cell_id: "markdown" if cell_id in MARKDOWN_IDS else "code"
    for cell_id in EXPECTED_CELL_IDS
}
PROTECTED_CELL_SHA256 = {
    "a08-header": "3ff1edadbb47417ae04c521783a7d76f40f7e4bfbd2f2985effe384e8636b208",
    "a08-setup": "a27e20efd45d3556a420e9653ad5d2fb834bc45ec7d035c17cdb90bcc6bc6220",
    "a08-terms-data": "0888ac5c6da29f2fc882d06cc707956518908ad339ce2a3319625100cc9e2d0c",
    "a08-task2-prompt": "c121bd5943bff1925995836ece9508768e394ff2da0d9d906ee7d319ccb972e6",
    "a08-task3-prompt": "45e6d964a75c3a478c63d4623416fec1f814c576ffa344f7b1e416618604ead0",
    "a08-final-verify": "70e228d9b9a14fb1a6f111a42acb08f78103f125f099a449b53e148278713b76",
}
STUDENT_MARKDOWN_IDS = {
    "a08-task1-contract", "a08-task1-explain", "a08-task2-explain",
    "a08-task3-explain", "a08-synthesis",
}
STUDENT_CODE_IDS = (
    set(EXPECTED_CELL_IDS) - MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
)
REQUIRED_FUNCTIONS = {
    "build_count_summary",
    "build_center_summary",
    "add_center_context",
    "build_center_channel_summary",
    "build_resolution_pivot",
}
FIXTURE_MANIFEST = {
    "fixture_id": "a08-support-requests-v1",
    "provenance": "Course-authored synthetic support-request records; no real, identifying, or customer data.",
    "path": "support_requests.csv",
    "row_grain": "one row per synthetic support request",
    "row_count": 15,
    "columns": [
        "request_id", "center", "agent_id", "channel",
        "resolution_minutes", "satisfaction_score",
    ],
    "sha256": "a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6",
    "center_levels": ["Central", "Harbor", "Ridge", "Valley"],
    "channel_levels": ["Email", "Phone", "Chat"],
}
ARTIFACTS = {
    "center_count_summary.csv": (
        3,
        ["center", "request_count", "satisfaction_count", "unique_agent_count"],
        98,
        "0735d0647dbbe2199b1de03e1061bf6c3a7a9d15bb553d128bdc1ab295ef2f36",
    ),
    "center_summary.csv": (
        3,
        [
            "center", "request_count", "satisfaction_count",
            "unique_agent_count", "total_resolution_minutes",
            "mean_resolution_minutes",
        ],
        174,
        "6c528bd229cd0ce2db2f4c90f09fd2a9ba670fb3aa659951bc113d70a33afad4",
    ),
    "requests_with_context.csv": (
        15,
        [
            "request_id", "center", "agent_id", "channel",
            "resolution_minutes", "satisfaction_score",
            "center_mean_resolution_minutes", "difference_from_center_mean",
        ],
        680,
        "391d56794e1537244c8d0b97f39e25e822b1e54d45b049fca98760ba646b1a7a",
    ),
    "center_channel_summary.csv": (
        8,
        ["center", "channel", "request_count", "mean_resolution_minutes"],
        210,
        "41b74a8dac05eff1695e6b972b360bd2b1730e77f5e2060e42801533a07180da",
    ),
    "mean_resolution_pivot.csv": (
        3,
        ["center", "Email", "Phone", "Chat"],
        86,
        "1274782fc4e773bfd572736c0af106842d92751d672b6ad341207574e636dedf",
    ),
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/support_requests.csv",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}
DELIVERY_FILES = {".classroom50.yaml", ".github/workflows/autograde.yaml"}
BANNED_ATTRIBUTES = {
    "apply", "filter", "crosstab", "merge", "join", "pivot", "melt",
    "stack", "unstack", "fillna", "ffill", "bfill", "interpolate",
    "drop_duplicates", "replace", "plot", "hist", "boxplot", "to_datetime",
    "to_period", "resample", "rolling", "expanding", "ewm", "shift",
    "asfreq", "corr", "cov",
}
BANNED_IMPORT_ROOTS = {
    "matplotlib", "seaborn", "altair", "bokeh", "plotly", "scipy",
    "statsmodels", "sklearn", "xgboost", "tensorflow", "torch", "requests",
    "urllib", "dask", "multiprocessing", "joblib", "numba",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_bytes(relative: str) -> bytes:
    path = ASSIGNMENT_DIR / relative
    _assert(path.is_file(), f"Missing protected file: {relative}.")
    return path.read_bytes()


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


def _literal_keyword(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _calls(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == attribute
    ]


def check_environment_and_protected_files() -> None:
    _check_submission_inventory()
    _assert(
        sys.version_info[:3] == (3, 12, 13),
        "Run this checker with the Assignment 08 CPython 3.12.13 interpreter.",
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
    _assert("output/" not in gitignore and "*.csv" not in gitignore, "Required CSVs must remain visible to Git.")
    for relative, expected in PROTECTED_FILE_SHA256.items():
        observed = sha256(_read_bytes(relative)).hexdigest()
        _assert(observed == expected, f"Restore the protected {relative} file.")
    for legacy in (
        "assignment.md", "data_generator.ipynb",
        "data_generator.md", "DATA_SCHEMA.md", "TIPS.md",
    ):
        _assert(not (ASSIGNMENT_DIR / legacy).exists(), f"Remove legacy assignment surface: {legacy}.")


def check_fixture() -> None:
    manifest_bytes = _read_bytes("data/fixture.json")
    _assert(manifest_bytes.endswith(b"\n") and b"\r" not in manifest_bytes, "Restore exact fixture manifest line endings.")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    _assert(manifest == FIXTURE_MANIFEST, "Restore the exact fixture manifest semantics.")
    relative = Path(manifest["path"])
    _assert(not relative.is_absolute() and relative.parts == ("support_requests.csv",), "Fixture path must be safe and relative.")
    data_dir = ASSIGNMENT_DIR / "data"
    actual = {path.name for path in data_dir.iterdir() if path.is_file()}
    _assert(actual == {"fixture.json", "support_requests.csv"}, "Keep exactly the supplied fixture and manifest in data/.")
    fixture = _read_bytes("data/support_requests.csv")
    _assert(len(fixture) == 469, "Restore the exact 469-byte support fixture.")
    _assert(fixture.endswith(b"\n") and b"\r" not in fixture, "Restore exact support fixture line endings.")
    _assert(sha256(fixture).hexdigest() == manifest["sha256"], "Restore data/support_requests.csv.")
    rows = list(csv.reader(fixture.decode("utf-8").splitlines()))
    _assert(rows[0] == manifest["columns"], "Fixture columns changed.")
    _assert(len(rows) - 1 == manifest["row_count"], "Fixture row count changed.")


def _load_notebook() -> tuple[dict, dict[str, dict]]:
    path = ASSIGNMENT_DIR / "assignment.ipynb"
    _assert(path.is_file(), "Missing assignment.ipynb.")
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"assignment.ipynb must be valid UTF-8 notebook JSON: {error}") from error
    cells = notebook.get("cells")
    _assert(notebook.get("nbformat") == 4 and notebook.get("nbformat_minor") == 5, "Keep notebook format 4, minor 5.")
    _assert(isinstance(cells, list) and len(cells) == 25, "Restore the exact 25-cell notebook.")
    ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    _assert(ids == EXPECTED_CELL_IDS and len(ids) == len(set(ids)), "Restore the exact unique cell IDs and order.")
    for cell in cells:
        _assert(cell.get("cell_type") == EXPECTED_CELL_TYPES[cell["id"]], f"Restore cell type: {cell['id']}.")
    _assert(
        notebook.get("metadata", {}).get("kernelspec")
        == {"display_name": "Python 3", "language": "python", "name": "python3"},
        "Restore the portable Python 3 kernelspec.",
    )
    return notebook, {cell["id"]: cell for cell in cells}


def check_notebook_source() -> None:
    _notebook, by_id = _load_notebook()
    for cell_id, expected in PROTECTED_CELL_SHA256.items():
        observed = sha256(_cell_source(by_id[cell_id]).encode()).hexdigest()
        _assert(observed == expected, f"Restore the protected notebook cell {cell_id}.")

    student_markdown = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_IDS)
    student_code = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_CODE_IDS)
    _assert("TODO" not in student_markdown + student_code, "Complete every TODO in the notebook.")
    _assert("NotImplementedError" not in student_code, "Replace every starter function scaffold.")
    try:
        tree = ast.parse(student_code)
    except SyntaxError as error:
        raise AssertionError(f"Student code has a syntax error: {error}") from error
    _assert(not any(isinstance(node, ast.Pass) for node in ast.walk(tree)), "Remove unfinished pass statements.")
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    _assert(REQUIRED_FUNCTIONS.issubset(functions), "Define all five required public functions.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = {alias.name.split(".")[0] for alias in node.names}
            _assert(not roots.intersection(BANNED_IMPORT_ROOTS), "Remove out-of-scope imports.")
            raise AssertionError("Student cells must use the supplied imports; remove added imports.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            _assert(node.func.attr not in BANNED_ATTRIBUTES, f"Remove out-of-scope API call: {node.func.attr}.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            _assert(node.func.id not in {"eval", "exec", "__import__"}, f"Remove forbidden call: {node.func.id}.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            absolute = value.startswith(("/", "~")) or (
                len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"}
            )
            _assert(not absolute, f"Remove absolute path literal: {value!r}.")
    lowered = student_code.lower()
    for fragment in (
        "/content", "drive.mount", "files.upload", "http://", "https://",
        "urlopen", "urlretrieve", "requests.get", "requests.post",
        "requests.request", "requests.session", "random.", "datetime.now",
        "timestamp.now", "date.today", "read_html(",
    ):
        _assert(fragment not in lowered, f"Remove nonportable or out-of-scope code: {fragment}.")
    for line in student_code.splitlines():
        _assert(not line.lstrip().startswith(("!", "%")), "Remove notebook magics and shell commands.")

    groupby_calls = _calls(tree, "groupby")
    _assert(len(groupby_calls) >= 5, "Use the required explicit grouped operations.")
    for call in groupby_calls:
        for policy in ("observed", "sort", "dropna"):
            _assert(_literal_keyword(call, policy) is True, f"Every groupby must set {policy}=True explicitly.")

    for name in REQUIRED_FUNCTIONS:
        function_source = ast.unparse(functions[name])
        for canonical in ("Central", "Harbor", "Ridge", "Valley", "Q001"):
            _assert(canonical not in function_source, f"{name} must not hard-code canonical labels or IDs.")
        _assert(not _calls(functions[name], "read_csv") and not _calls(functions[name], "to_csv"), f"{name} must not perform file I/O.")

    count_node = functions["build_count_summary"]
    _assert(len(_calls(count_node, "groupby")) == 1, "Count function must create one reusable center GroupBy.")
    for operation in ("size", "count", "nunique", "concat", "reset_index"):
        _assert(_calls(count_node, operation), f"Count function must use {operation}.")
    _assert("satisfaction_score" in ast.unparse(count_node) and "agent_id" in ast.unparse(count_node), "Count the required selected columns.")

    for function_name in ("build_center_summary", "build_center_channel_summary"):
        calls = _calls(functions[function_name], "groupby")
        _assert(len(calls) == 1 and _literal_keyword(calls[0], "as_index") is False, f"{function_name} must use as_index=False.")
        _assert(len(_calls(functions[function_name], "agg")) == 1, f"{function_name} must use one named aggregation.")
    center_source = ast.unparse(functions["build_center_summary"])
    for name in (
        "request_count", "satisfaction_count", "unique_agent_count",
        "total_resolution_minutes", "mean_resolution_minutes",
    ):
        _assert(name in center_source, f"Center named aggregation is missing {name}.")
    context_node = functions["add_center_context"]
    transform_calls = _calls(context_node, "transform")
    _assert(len(transform_calls) == 1 and transform_calls[0].args and isinstance(transform_calls[0].args[0], ast.Constant) and transform_calls[0].args[0].value == "mean", "Context function must use exactly transform('mean').")
    _assert("copy" in {_call.func.attr for _call in ast.walk(context_node) if isinstance(_call, ast.Call) and isinstance(_call.func, ast.Attribute)}, "Context function must copy its input.")

    pivot_calls = _calls(tree, "pivot_table")
    _assert(len(pivot_calls) == 1, "Student code must contain exactly one pivot_table call.")
    pivot_call = pivot_calls[0]
    _assert(isinstance(pivot_call.func.value, ast.Name) and pivot_call.func.value.id == "pd", "Call pd.pivot_table exactly once.")
    expected_pivot = {
        "index": "center", "columns": "channel", "values": "resolution_minutes",
        "aggfunc": "mean", "observed": True, "sort": True, "dropna": True,
    }
    for keyword, expected in expected_pivot.items():
        _assert(_literal_keyword(pivot_call, keyword) == expected, f"Pivot must set {keyword}={expected!r} explicitly.")
    _assert(len(_calls(functions["build_resolution_pivot"], "pivot_table")) == 1, "The sole pivot_table call belongs in build_resolution_pivot.")
    task3_run_source = _cell_source(by_id["a08-task3-run"])
    for evidence in (
        "pivot_reference", "build_center_channel_summary", "itertuples",
        "mean_resolution_minutes", "resolution_pivot.loc",
    ):
        _assert(evidence in task3_run_source, f"Task 3 cell-for-cell comparison is missing {evidence}.")


def check_artifacts() -> None:
    output_dir = ASSIGNMENT_DIR / "output"
    _assert(output_dir.is_dir() and not output_dir.is_symlink(), "Missing regular output/ directory.")
    actual = {path.name for path in output_dir.iterdir() if path.is_file() or path.is_symlink()}
    _assert(actual == set(ARTIFACTS) | {".gitkeep"}, "Keep exactly the five required CSVs plus output/.gitkeep.")
    for name, (rows, columns, byte_count, digest) in ARTIFACTS.items():
        data = (output_dir / name).read_bytes()
        _assert(len(data) == byte_count, f"Rerun the notebook to rebuild output/{name}.")
        _assert(data.endswith(b"\n") and b"\r" not in data, f"Use exact LF/final-newline output for {name}.")
        _assert(sha256(data).hexdigest() == digest, f"Rerun the notebook to rebuild output/{name}.")
        parsed = list(csv.reader(data.decode("utf-8").splitlines()))
        _assert(parsed and parsed[0] == columns, f"Wrong columns in output/{name}.")
        _assert(len(parsed) - 1 == rows, f"Wrong row count in output/{name}.")


def main() -> int:
    checks = (
        ("environment and protected files", check_environment_and_protected_files),
        ("prepared fixture", check_fixture),
        ("notebook structure and source", check_notebook_source),
        ("five generated artifacts", check_artifacts),
    )
    failures: list[str] = []
    for label, function in checks:
        try:
            function()
        except Exception as error:
            failures.append(f"[FIX] {label}: {error}")
    if failures:
        print("\n".join(failures))
        return 1
    print("All public checks passed. The notebook and five artifacts are ready for fresh central grading.")
    print("The public checker does not award points or assess the quality of Markdown reasoning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
