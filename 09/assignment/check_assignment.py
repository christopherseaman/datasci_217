# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "numpy==2.0.2",
#   "pandas==3.0.5",
# ]
# ///

"""Dependency-free public structural checks for Assignment 09.

This checker reads package files, notebook source, and required CSV artifacts.
It never executes notebook code, trusts stored output, awards points, or judges
student explanations. The independent central grader fresh-executes a copy.
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
    ".ipynb_checkpoints/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n.venv/\nvenv/\n"
)
PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    "README.md": "bf1516bddac77a5b8e1003a237b4e954cdccac377f31f237c8ffe39840fb7558",
    "PLATFORM_CHECK.md": "2ada43d46e2d26412f5f13a12a7315ddd24f77e333303a0b071e46662a381c1c",
    "data/fixture.json": "27558bc4da7738775879501a6f11a0a9d874f3948823e54bb5e82ab91a02d703",
    "data/zone_co2_readings.csv": "c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4",
}
CELL_IDS = [
    "a09-header", "a09-setup", "a09-terms-data", "a09-load",
    "a09-task1-contract", "a09-task1-values", "a09-prepare-function",
    "a09-task1-run", "a09-task1-save", "a09-task1-explain",
    "a09-task2-prompt", "a09-hourly-function", "a09-summary-function",
    "a09-task2-run", "a09-task2-save", "a09-task2-explain",
    "a09-task3-prompt", "a09-features-function", "a09-task3-features-run",
    "a09-availability-values", "a09-blocks-function", "a09-task3-run",
    "a09-task3-save", "a09-task3-explain", "a09-synthesis",
    "a09-final-verify",
]
MARKDOWN_IDS = {
    "a09-header", "a09-terms-data", "a09-task1-contract",
    "a09-task1-explain", "a09-task2-prompt", "a09-task2-explain",
    "a09-task3-prompt", "a09-task3-explain", "a09-synthesis",
}
PROTECTED_CELL_SHA256 = {
    "a09-header": "fd9bfd600830518361e0779324743bc86f613c975b7ce0ac46027a10b0242542",
    "a09-setup": "46d251a68a1b2b740e64bdb90e892500cade7bb6235b69e0c2876670cd4f837d",
    "a09-terms-data": "f8ad0129811ecd03ed7c0dc60b860b68a1be545003e4961cdaea6b43489dfa59",
    "a09-task2-prompt": "d1d9d124b7fd9904ee9a47256745def60937945d1bfe40194239314199bedc1b",
    "a09-task3-prompt": "90f1797b0b65472c3567451467e85e144a2add9110a3d11a1714d3d31fe75966",
    "a09-final-verify": "015c87bbc3742a61961efbf806ed513f67363191766ad8bcb7d1bf7200fb996a",
}
STUDENT_MARKDOWN_IDS = MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
STUDENT_CODE_IDS = set(CELL_IDS) - MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
FUNCTION_SIGNATURES = {
    "prepare_temporal_panel": ["reading_table", "source_timezone"],
    "build_hourly_grid": ["prepared_table"],
    "build_two_hour_summary": ["prepared_table"],
    "build_past_features": ["prepared_table"],
    "build_chronological_blocks": ["prepared_table", "holdout_start"],
}
MANIFEST = {
    "fixture_id": "a09-temporal-panel-v1",
    "provenance": "Course-authored synthetic indoor-air sensor readings; no real, identifying, or occupant data.",
    "path": "zone_co2_readings.csv",
    "row_grain": "one recorded CO2 reading for one zone and local timestamp",
    "row_count": 12,
    "columns": ["zone", "recorded_at", "co2_ppm"],
    "sha256": "c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4",
    "source_timezone": "America/New_York",
}
ARTIFACTS = {
    "prepared_panel.csv": (12, ["zone", "recorded_at", "co2_ppm", "source_row"], 523, "e29aa2ac53cffe29c2f412170100e0725ce4b6a2e0cfdd09e5f1cb92fd5fcd64"),
    "hourly_grid.csv": (16, ["zone", "recorded_at", "co2_ppm", "source_row", "grid_created_row", "source_value_missing"], 912, "d4de5178f7ca56960061efb1d263ff4022a9608bc44a6f679c397cb814c150c0"),
    "two_hour_summary.csv": (8, ["zone", "recorded_at", "mean_co2_ppm", "reading_count"], 367, "0805cb42799880b85afdee35b0af36c53d606311e1a83a0a995509c647b9d999"),
    "temporal_features.csv": (12, ["zone", "recorded_at", "co2_ppm", "co2_lag_1", "co2_difference", "mean_previous_2_observations", "mean_previous_2h"], 779, "a83d7b858bdcf0203f211cd4dbfc907f0530d132a989ee6ead0fc46e6401d0bb"),
    "availability_decisions.csv": (4, ["candidate", "latest_required_timestamp", "available_by_prediction_time", "decision"], 310, "07128f0c67a5765d115c8feb7f3a5ee547450b985b48647c3dcc2324d27a4607"),
    "chronological_blocks.csv": (12, ["zone", "recorded_at", "co2_ppm", "source_row", "block"], 649, "ddde39feea7e3b864088919675fe279dee82021d514980bfd9271cfacf9ec0d2"),
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/zone_co2_readings.csv",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}
DELIVERY_FILES = {".classroom50.yaml", ".github/workflows/autograde.yaml"}
BANNED_ATTRIBUTES = {
    "fillna", "ffill", "bfill", "backfill", "interpolate", "replace",
    "apply", "filter", "expanding", "ewm", "to_period", "infer_freq",
    "plot", "hist", "boxplot", "corr", "cov", "predict", "fit", "score",
}
BANNED_IMPORT_ROOTS = {
    "matplotlib", "seaborn", "altair", "bokeh", "plotly", "scipy",
    "statsmodels", "sklearn", "xgboost", "tensorflow", "torch", "requests",
    "urllib", "dask", "multiprocessing", "joblib", "numba",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read(relative: str) -> bytes:
    path = ASSIGNMENT_DIR / relative
    _assert(path.is_file(), f"Missing protected file: {relative}.")
    return path.read_bytes()


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


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


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == name
    ]


def _keyword(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _first_literal(call: ast.Call):
    if call.args and isinstance(call.args[0], ast.Constant):
        return call.args[0].value
    return None


def _literal_argument(call: ast.Call, position: int = 0):
    if len(call.args) <= position:
        return None
    try:
        return ast.literal_eval(call.args[position])
    except (ValueError, TypeError):
        return None


def check_environment_and_files() -> None:
    _check_submission_inventory()
    _assert(sys.version_info[:3] == (3, 12, 13), "Use the Assignment 09 CPython 3.12.13 interpreter.")
    for package, expected in (("numpy", "2.0.2"), ("pandas", "3.0.5")):
        try:
            observed = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise AssertionError(f"Install {package}=={expected}.") from error
        _assert(observed == expected, f"Expected {package}=={expected}; found {observed}.")
    _assert(_read(".python-version").decode() == EXPECTED_PYTHON, "Restore .python-version.")
    _assert(_read("requirements.txt").decode() == EXPECTED_REQUIREMENTS, "Restore requirements.txt.")
    gitignore = _read(".gitignore").decode()
    _assert(gitignore == EXPECTED_GITIGNORE, "Restore .gitignore.")
    _assert("output/" not in gitignore and "*.csv" not in gitignore, "Required CSVs must remain visible to Git.")
    for relative, digest in PROTECTED_FILE_SHA256.items():
        _assert(sha256(_read(relative)).hexdigest() == digest, f"Restore protected {relative}.")
    for legacy in ("assignment.md", "data_generator.ipynb", "data_generator.md", "q1_datetime.ipynb", "q2_resampling.ipynb", "q3_rolling.ipynb"):
        _assert(not (ASSIGNMENT_DIR / legacy).exists(), f"Remove legacy surface: {legacy}.")


def check_fixture() -> None:
    manifest_bytes = _read("data/fixture.json")
    _assert(len(manifest_bytes) == 473 and manifest_bytes.endswith(b"\n") and b"\r" not in manifest_bytes, "Restore exact fixture manifest bytes.")
    _assert(json.loads(manifest_bytes) == MANIFEST, "Restore exact fixture manifest semantics.")
    relative = Path(MANIFEST["path"])
    _assert(not relative.is_absolute() and relative.parts == ("zone_co2_readings.csv",), "Fixture path must remain safe and relative.")
    actual = {path.name for path in (ASSIGNMENT_DIR / "data").iterdir() if path.is_file()}
    _assert(actual == {"fixture.json", "zone_co2_readings.csv"}, "Keep exactly the supplied two fixture files.")
    fixture = _read("data/zone_co2_readings.csv")
    _assert(len(fixture) == 380 and fixture.endswith(b"\n") and b"\r" not in fixture, "Restore exact fixture bytes and line endings.")
    _assert(sha256(fixture).hexdigest() == MANIFEST["sha256"], "Restore zone_co2_readings.csv.")
    rows = list(csv.reader(fixture.decode().splitlines()))
    _assert(rows[0] == MANIFEST["columns"] and len(rows) - 1 == 12, "Fixture schema or row count changed.")


def _load_notebook() -> dict[str, dict]:
    try:
        notebook = json.loads(_read("assignment.ipynb"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"assignment.ipynb must be valid UTF-8 notebook JSON: {error}") from error
    _assert(notebook.get("nbformat") == 4 and notebook.get("nbformat_minor") == 5, "Keep notebook format 4.5.")
    cells = notebook.get("cells")
    _assert(isinstance(cells, list) and len(cells) == 26, "Restore the exact 26-cell notebook.")
    ids = [cell.get("id") for cell in cells]
    _assert(ids == CELL_IDS and len(ids) == len(set(ids)), "Restore exact unique cell IDs/order.")
    for cell in cells:
        expected = "markdown" if cell["id"] in MARKDOWN_IDS else "code"
        _assert(cell.get("cell_type") == expected, f"Restore cell type: {cell['id']}.")
    _assert(notebook.get("metadata", {}).get("kernelspec") == {"display_name": "Python 3", "language": "python", "name": "python3"}, "Restore portable Python 3 kernelspec.")
    return {cell["id"]: cell for cell in cells}


def check_notebook_source() -> None:
    by_id = _load_notebook()
    for cell_id, digest in PROTECTED_CELL_SHA256.items():
        _assert(sha256(_source(by_id[cell_id]).encode()).hexdigest() == digest, f"Restore protected cell {cell_id}.")
    student_markdown = "\n".join(_source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_IDS)
    student_code = "\n".join(_source(by_id[cell_id]) for cell_id in STUDENT_CODE_IDS)
    _assert("TODO" not in student_markdown + student_code, "Complete every TODO in the notebook.")
    _assert("NotImplementedError" not in student_code, "Replace every starter function scaffold.")
    try:
        tree = ast.parse(student_code)
    except SyntaxError as error:
        raise AssertionError(f"Student code has a syntax error: {error}") from error
    _assert(not any(isinstance(node, ast.Pass) for node in ast.walk(tree)), "Remove unfinished pass statements.")
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    _assert(set(FUNCTION_SIGNATURES).issubset(functions), "Define all five required functions.")
    for name, arguments in FUNCTION_SIGNATURES.items():
        observed = [argument.arg for argument in functions[name].args.args]
        _assert(observed == arguments and not functions[name].args.vararg and not functions[name].args.kwarg, f"Restore exact signature for {name}.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = {alias.name.split(".")[0] for alias in node.names}
            _assert(not roots.intersection(BANNED_IMPORT_ROOTS), "Remove out-of-scope imports.")
            raise AssertionError("Student cells must use supplied imports; remove added imports.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            _assert(node.func.attr not in BANNED_ATTRIBUTES, f"Remove out-of-scope API: {node.func.attr}.")
            if node.func.attr == "shift" and node.args and isinstance(node.args[0], ast.UnaryOp) and isinstance(node.args[0].op, ast.USub):
                raise AssertionError("Negative shift/lead computation is outside scope.")
            if node.func.attr == "rolling":
                _assert(_keyword(node, "center") in (None, False), "Centered windows are outside scope.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            _assert(node.func.id not in {"eval", "exec", "__import__"}, f"Remove forbidden call: {node.func.id}.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            absolute = value.startswith(("/", "~")) or (len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"})
            _assert(not absolute, f"Remove absolute path literal: {value!r}.")
    lowered = student_code.lower()
    for fragment in ("/content", "drive.mount", "files.upload", "http://", "https://", "urlopen", "urlretrieve", "requests.get", "random.", "datetime.now", "timestamp.now", "date.today", "read_html("):
        _assert(fragment not in lowered, f"Remove nonportable/out-of-scope code: {fragment}.")
    for line in student_code.splitlines():
        _assert(not line.lstrip().startswith(("!", "%")), "Remove shell commands and notebook magics.")

    prepare = functions["prepare_temporal_panel"]
    _assert(len(_calls(prepare, "to_datetime")) == 1, "Prepare with one exact-format to_datetime call.")
    parse = _calls(prepare, "to_datetime")[0]
    _assert(_keyword(parse, "format") == "%Y-%m-%d %H:%M", "Parse with the documented exact format.")
    _assert(len(_calls(prepare, "tz_localize")) == 1 and len(_calls(prepare, "tz_convert")) == 1, "Localize once and convert once.")
    _assert(len(_calls(prepare, "sort_values")) == 1 and _keyword(_calls(prepare, "sort_values")[0], "kind") == "stable" and _literal_argument(_calls(prepare, "sort_values")[0]) == ["zone", "recorded_at"], "Use one stable zone/time sort.")
    _assert(_calls(prepare, "copy"), "Prepare must deep-copy its input.")

    hourly = functions["build_hourly_grid"]
    _assert(len(_calls(hourly, "groupby")) == 1 and len(_calls(hourly, "resample")) == 1 and len(_calls(hourly, "asfreq")) == 1, "Hourly grid must use one entity-grouped resample('h').asfreq().")
    _assert(_calls(hourly, "floor") and "ValueError" in ast.unparse(hourly), "Reject off-grid labels before asfreq.")
    hourly_source = ast.unparse(hourly).replace("'", '"')
    for evidence in ('hourly["source_row"].isna()', 'hourly["source_row"].eq(1)', 'hourly["co2_ppm"].isna()'):
        _assert(evidence in hourly_source, f"Hourly provenance rule is missing {evidence}.")
    summary = functions["build_two_hour_summary"]
    _assert(len(_calls(summary, "groupby")) == 1 and len(_calls(summary, "resample")) == 1 and len(_calls(summary, "agg")) == 1, "Two-hour summary must use grouped named aggregation.")
    for node in (hourly, summary):
        group = _calls(node, "groupby")[0]
        for policy in ("observed", "sort", "dropna"):
            _assert(_keyword(group, policy) is True, f"Temporal groupby must set {policy}=True.")
    _assert(_first_literal(_calls(hourly, "resample")[0]) == "h", "Use lowercase hourly alias 'h'.")
    summary_source = ast.unparse(summary)
    summary_resample = _calls(summary, "resample")[0]
    _assert(_first_literal(summary_resample) == "2h" and _keyword(summary_resample, "closed") == "left" and _keyword(summary_resample, "label") == "left", "Use explicit left-closed, left-labeled two-hour bins.")
    named_aggregation = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in _calls(summary, "agg")[0].keywords
        if keyword.arg is not None
    }
    _assert(named_aggregation == {"mean_co2_ppm": ("co2_ppm", "mean"), "reading_count": ("source_row", "sum")}, "Use the exact state mean and additive source-row count.")

    features = functions["build_past_features"]
    _assert(len(_calls(features, "shift")) == 2 and len(_calls(features, "diff")) == 1 and len(_calls(features, "rolling")) == 2, "Use exact grouped lag/difference and two window operations.")
    feature_groups = _calls(features, "groupby")
    _assert(len(feature_groups) == 3, "Keep lag/difference and both window meanings entity-scoped.")
    for group in feature_groups:
        _assert(_first_literal(group) == "zone", "Every past-feature group must use zone.")
        for policy in ("observed", "sort", "dropna"):
            _assert(_keyword(group, policy) is True, f"Every past-feature groupby must set {policy}=True.")
    _assert(len(_calls(features, "merge")) == 1 and _keyword(_calls(features, "merge")[0], "validate") == "one_to_one", "Return elapsed windows with one validated merge.")
    feature_source = ast.unparse(features)
    elapsed_rolls = [call for call in _calls(features, "rolling") if _first_literal(call) == "2h"]
    _assert(len(elapsed_rolls) == 1 and _keyword(elapsed_rolls[0], "closed") == "left" and _keyword(elapsed_rolls[0], "min_periods") == 1, "Elapsed window must exclude the current row.")
    for name, function in functions.items():
        source = ast.unparse(function)
        for canonical in ("atrium", "studio", "2026-01-20", "America/New_York"):
            _assert(canonical not in source, f"{name} must derive values from its arguments.")
        _assert(not _calls(function, "read_csv") and not _calls(function, "to_csv"), f"{name} must not perform file I/O.")


def check_artifacts() -> None:
    output = ASSIGNMENT_DIR / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Missing regular output/ directory.")
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(actual == set(ARTIFACTS) | {".gitkeep"}, "Keep exactly six required CSVs plus output/.gitkeep.")
    for name, (rows, columns, byte_count, digest) in ARTIFACTS.items():
        data = (output / name).read_bytes()
        _assert(len(data) == byte_count and data.endswith(b"\n") and b"\r" not in data, f"Rerun the notebook to rebuild output/{name}.")
        _assert(sha256(data).hexdigest() == digest, f"Rerun the notebook to rebuild output/{name}.")
        parsed = list(csv.reader(data.decode().splitlines()))
        _assert(parsed[0] == columns and len(parsed) - 1 == rows, f"Wrong schema or row count in output/{name}.")


def main() -> int:
    checks = (
        ("environment and protected files", check_environment_and_files),
        ("prepared fixture", check_fixture),
        ("notebook source and five functions", check_notebook_source),
        ("six generated artifacts", check_artifacts),
    )
    failures = []
    for label, function in checks:
        try:
            function()
        except Exception as error:
            failures.append(f"[FIX] {label}: {error}")
    if failures:
        print("\n".join(failures))
        return 1
    print("All public checks passed. The notebook and six artifacts are ready for fresh central grading.")
    print("The public checker does not award points or assess Markdown reasoning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
