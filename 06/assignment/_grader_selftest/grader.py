# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==7.1.0",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.5",
# ]
# ///

"""Independent central-grader reference for Assignment 06.

This instructor-only module never imports the student-editable public checker.
It fresh-executes stripped disposable notebook copies, appends grader-owned
behavioral checks, and emits the official grading result contract.
"""

from __future__ import annotations

import ast
import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import numpy as np
import pandas as pd


EXPECTED_CELL_IDS = [
    "a06-header", "a06-setup", "a06-data-contract", "a06-load",
    "a06-task1-contract", "a06-contract-values", "a06-key-checks",
    "a06-duplicate-failure", "a06-task1-functions", "a06-task1-run",
    "a06-task1-save", "a06-task2-contract", "a06-stack-function",
    "a06-stack-run", "a06-schema-drift", "a06-align-function",
    "a06-align-run", "a06-task2-save", "a06-task3-contract",
    "a06-reshape-functions", "a06-reshape-run", "a06-duplicate-pivot",
    "a06-task3-save", "a06-reflection", "a06-final-verify",
]
EXPECTED_CELL_TYPES = {
    cell_id: "markdown" if cell_id in {
        "a06-header", "a06-data-contract", "a06-task1-contract",
        "a06-task2-contract", "a06-task3-contract", "a06-reflection",
    } else "code"
    for cell_id in EXPECTED_CELL_IDS
}
PROTECTED_CELL_SHA256 = {
    "a06-header": "c55dd34558c7a8730e18a3b5df29a6c999285142c69ae5a29ec06be1dead7766",
    "a06-setup": "68f227f40a9ed45e80664eaf939b2b270326e483389dd547abe76fa363bdec44",
    "a06-data-contract": "0265d9faa5bb57d9d5061a4ca68426982deb244a18b67506ec33cca4f1d91270",
    "a06-final-verify": "d3dd35f4762145d5f012c7798bc9072f2d8d2cc195b74e772c931dd18979b4c9",
}
PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21",
    ".gitignore": "2d857aeb38b492c9cac001ba2bef86d2287357f7f5b3f1203d929ac1e79fa138",
    "README.md": "b832a10866a5cd8b90517f014d5a9e7abec040c24834515296b18aa4ede7a721",
    "PLATFORM_CHECK.md": "be24dc511a18966dbe361835c4ff62d3f24f841f629fa7e0d8791c71759c54d5",
    "check_assignment.py": "fcae068af4c7eccfa9b2022c08cff4e19ecbdd9c7076f50c76e0e19fba1540f1",
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
ARTIFACT_SHA256 = {
    "specimen_merge_audit.csv": "1bc33aeecbae2483e314399784bbcaf8b8847798fe3ca5b7662908053615e98c",
    "combined_specimens.csv": "78cbd883bea393fb84d699cdb9923a9d71d7c045d002eebe88cc84c9da61c666",
    "aligned_features.csv": "19cb5d07f7ae51ce0347876802a44eadf48490076bb24dbdcae547d9388775e7",
    "sensor_scores_long.csv": "989affb14d49ecd0e144e23a6b53ab4a093edd6211656390144869ecaa3126dd",
    "sensor_scores_round_trip.csv": "6eb9bfb9561fc7c55708bc0038b77b99e6d383f85843dff3e735c5962abe8701",
}
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
STUDENT_MARKDOWN_IDS = {
    "a06-task1-contract", "a06-task2-contract", "a06-task3-contract",
    "a06-reflection",
}
STUDENT_CODE_IDS = set(EXPECTED_CELL_IDS) - STUDENT_MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
REQUIRED_FUNCTIONS = {
    "select_current_stations", "validated_station_merge",
    "stack_specimen_partitions", "align_specimen_features",
    "wide_to_long_scores", "long_to_wide_scores",
}
BANNED_ATTRIBUTES = {
    "agg", "aggregate", "bfill", "crosstab", "drop_duplicates", "dropna",
    "ewm", "expanding", "ffill", "fillna", "groupby", "interpolate",
    "join", "pivot_table", "plot", "replace", "resample", "rolling",
    "to_datetime", "transform",
}


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
    ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    _assert(ids == EXPECTED_CELL_IDS and len(ids) == len(set(ids)), "cell IDs or order changed")
    for cell in cells:
        _assert(cell.get("cell_type") == EXPECTED_CELL_TYPES[cell["id"]], f"cell type changed: {cell['id']}")
    _assert(
        notebook.get("metadata", {}).get("kernelspec")
        == {"display_name": "Python 3", "language": "python", "name": "python3"},
        "portable kernelspec changed",
    )
    return notebook, {cell["id"]: cell for cell in cells}


def _keyword_literal(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _attribute_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
        and item.func.attr == name
    ]


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


def _check_source_contract(by_id: dict[str, dict]) -> None:
    for cell_id, expected in PROTECTED_CELL_SHA256.items():
        observed = sha256(_cell_source(by_id[cell_id]).encode()).hexdigest()
        _assert(observed == expected, f"protected cell changed: {cell_id}")
    student_markdown = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_IDS)
    student_code = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_CODE_IDS)
    _assert("TODO" not in student_markdown + student_code, "unfinished TODO remains")
    tree = ast.parse(student_code)
    _assert(not any(isinstance(node, ast.Pass) for node in ast.walk(tree)), "scaffold pass remains")
    function_nodes = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    _assert(REQUIRED_FUNCTIONS.issubset(function_nodes), "required reusable function missing")
    for node in ast.walk(tree):
        _assert(not isinstance(node, (ast.Import, ast.ImportFrom)), "student import used")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            _assert(node.func.attr not in BANNED_ATTRIBUTES, f"out-of-scope API used: {node.func.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            _assert(node.func.id not in {"eval", "exec", "__import__"}, f"forbidden call: {node.func.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            absolute = value.startswith(("/", "~")) or (
                len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"}
            )
            _assert(not absolute, f"absolute path literal used: {value!r}")
    lowered = student_code.lower()
    for fragment in ("/content", "drive.mount", "files.upload", "http://", "https://", "urlopen", "requests."):
        _assert(fragment not in lowered, f"nonportable or remote code: {fragment}")
    for line in student_code.splitlines():
        _assert(not line.lstrip().startswith(("!", "%")), "notebook magic or shell line used")

    load_tree = ast.parse(_cell_source(by_id["a06-load"]))
    read_calls = _attribute_calls(load_tree, "read_csv")
    _assert(len(read_calls) == 6, "six protected fixtures must be read exactly once")
    _assert(not _attribute_calls(load_tree, "DataFrame"), "embedded replacement fixture used")
    load_source = _cell_source(by_id["a06-load"])
    for filename in FIXTURE_NAMES:
        _assert(filename in load_source, f"protected fixture not loaded: {filename}")

    merge_function = function_nodes["validated_station_merge"]
    merge_calls = _attribute_calls(merge_function, "merge")
    _assert(any(
        _keyword_literal(call, "on") == "station_code"
        and _keyword_literal(call, "how") == "left"
        and _keyword_literal(call, "validate") == "many_to_one"
        and _keyword_literal(call, "indicator") is True
        for call in merge_calls
    ), "validated merge must expose explicit key/how/cardinality/indicator")
    selector_source = ast.unparse(function_nodes["select_current_stations"])
    _assert("record_status" in selector_source and "current" in selector_source, "current-record source rule missing")

    stack_calls = _attribute_calls(function_nodes["stack_specimen_partitions"], "concat")
    _assert(any(_keyword_literal(call, "ignore_index") is True for call in stack_calls), "row concat must use ignore_index=True")
    align_calls = _attribute_calls(function_nodes["align_specimen_features"], "concat")
    _assert(any(_keyword_literal(call, "axis") == 1 and _keyword_literal(call, "join") == "outer" for call in align_calls), "horizontal concat must use outer label alignment")
    melt_calls = _attribute_calls(function_nodes["wide_to_long_scores"], "melt")
    _assert(any(
        _keyword_literal(call, "var_name") == "measurement_label"
        and _keyword_literal(call, "value_name") == "value"
        for call in melt_calls
    ), "wide-to-long function must use the required melt roles")
    pivot_calls = _attribute_calls(function_nodes["long_to_wide_scores"], "pivot")
    _assert(any(
        _keyword_literal(call, "columns") == "measurement_label"
        and _keyword_literal(call, "values") == "value"
        for call in pivot_calls
    ), "long-to-wide function must use structural pivot")
    _assert(
        not _attribute_calls(function_nodes["long_to_wide_scores"], "duplicated"),
        "structural pivot must naturally reject duplicate long keys",
    )

    failure_tree = ast.parse(_cell_source(by_id["a06-duplicate-failure"]))
    _assert(not any(isinstance(node, ast.Raise) for node in ast.walk(failure_tree)), "merge failure was manufactured")
    merge_handlers = [
        node for node in ast.walk(failure_tree)
        if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Attribute)
        and node.type.attr == "MergeError"
    ]
    _assert(any(
        isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Attribute)
        and node.type.attr == "MergeError"
        for node in ast.walk(failure_tree)
    ), "merge failure does not catch pd.errors.MergeError")
    _assert(any(
        _handler_assigns(handler, "duplicate_contract_failed", True)
        for handler in merge_handlers
    ), "merge failure flag was not set by the caught pandas failure")
    failure_calls = _attribute_calls(failure_tree, "merge")
    _assert(any(
        _keyword_literal(call, "on") == "station_code"
        and _keyword_literal(call, "how") == "left"
        and _keyword_literal(call, "validate") == "many_to_one"
        for call in failure_calls
    ), "unfiltered duplicate-key failure is not a validated left merge")

    pivot_failure_tree = ast.parse(_cell_source(by_id["a06-duplicate-pivot"]))
    _assert(not any(isinstance(node, ast.Raise) for node in ast.walk(pivot_failure_tree)), "pivot failure was manufactured")
    pivot_handlers = [
        node for node in ast.walk(pivot_failure_tree)
        if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name)
        and node.type.id == "ValueError"
    ]
    _assert(any(
        isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name)
        and node.type.id == "ValueError"
        for node in ast.walk(pivot_failure_tree)
    ), "duplicate pivot does not catch ValueError")
    _assert(any(
        _handler_assigns(handler, "duplicate_pivot_failed", True)
        for handler in pivot_handlers
    ), "pivot failure flag was not set by the caught pivot failure")
    _assert(any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "long_to_wide_scores"
        for node in ast.walk(pivot_failure_tree)
    ), "duplicate-pivot evidence does not call the student function")


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
    _assert(actual_files == STUDENT_PACKAGE_FILES, "student package inventory differs")
    for relative, expected in PROTECTED_FILE_SHA256.items():
        path = root / relative
        _assert(path.is_file(), f"missing protected file: {relative}")
        _assert(sha256(path.read_bytes()).hexdigest() == expected, f"protected file changed: {relative}")
    _assert((root / ".python-version").read_text() == "3.12.13\n", "wrong Python record")
    _assert((root / "requirements.txt").read_text() == "numpy==2.0.2\npandas==3.0.5\n", "wrong dependency records")
    gitignore = (root / ".gitignore").read_text()
    _assert("output/" not in gitignore and "*.csv" not in gitignore, "required CSV artifacts are ignored")
    actual_fixtures = {path.name for path in (root / "data").glob("*.csv") if path.is_file()}
    _assert(actual_fixtures == FIXTURE_NAMES, "fixture inventory changed")
    _, by_id = _load_notebook(root)
    _check_source_contract(by_id)
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "missing regular output directory")
    actual_outputs = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(actual_outputs == {".gitkeep", *ARTIFACT_SHA256}, "submission must contain exactly .gitkeep and five artifacts")
    for name, expected in ARTIFACT_SHA256.items():
        _assert(sha256((output / name).read_bytes()).hexdigest() == expected, f"committed artifact differs: {name}")


def _copy_submission(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            "_grader_selftest", ".venv", "__pycache__", ".ipynb_checkpoints",
            ".pytest_cache", "result.json",
        }.intersection(names)

    shutil.copytree(source, destination, ignore=ignore)


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {name: (root / "output" / name).read_bytes() for name in ARTIFACT_SHA256}


GRADER_CHECKS_SOURCE = r'''
central_results = {}

def _central_record(name, function):
    try:
        function()
    except Exception as error:
        central_results[name] = {'passed': False, 'detail': f'{type(error).__name__}: {error}'}
    else:
        central_results[name] = {'passed': True, 'detail': 'alternate behavioral checks passed'}

def _central_task1():
    alternate_history = pd.DataFrame(
        [
            ['K', 'Keep Current', 'east', 'current'],
            ['K', 'Keep Retired', 'east', 'retired'],
            ['L', 'Lake Current', 'north', 'current'],
            ['M', 'Mesa Current', 'west', 'current'],
        ],
        columns=['station_code', 'station_name', 'region', 'record_status'],
    ).astype('string')
    alternate_specimens = pd.DataFrame(
        [
            ['ALT9', 'Z9', 2, 'K', 'rock', 3.5],
            ['ALT2', 'Z3', 7, 'Q', 'water', 4.5],
            ['ALT7', 'Z1', 4, 'W', 'air', 1.5],
        ],
        columns=['specimen_id', 'collector_id', 'collection_number', 'station_code', 'material', 'mass_g'],
    )
    for column in ['specimen_id', 'collector_id', 'station_code', 'material']:
        alternate_specimens[column] = alternate_specimens[column].astype('string')
    alternate_specimens['collection_number'] = alternate_specimens['collection_number'].astype('int64')
    alternate_specimens['mass_g'] = alternate_specimens['mass_g'].astype('float64')
    history_snapshot = alternate_history.copy(deep=True)
    specimen_snapshot = alternate_specimens.copy(deep=True)
    selected = select_current_stations(alternate_history)
    assert alternate_history.equals(history_snapshot), 'selector mutated its input'
    assert selected['station_code'].tolist() == ['K', 'L', 'M']
    assert selected.loc[selected['station_code'].eq('K'), 'station_name'].item() == 'Keep Current'
    merged = validated_station_merge(alternate_specimens, selected)
    assert alternate_specimens.equals(specimen_snapshot), 'merge mutated specimens'
    assert selected['station_code'].tolist() == ['K', 'L', 'M'], 'merge mutated lookup'
    assert merged['specimen_id'].tolist() == ['ALT9', 'ALT2', 'ALT7']
    assert str(merged['_merge'].dtype) == 'category'
    counts = merged['_merge'].astype('string').value_counts().reindex(['both', 'left_only', 'right_only'], fill_value=0).astype('int64').to_dict()
    assert counts == {'both': 1, 'left_only': 2, 'right_only': 0}
    assert merged.loc[merged['_merge'].astype('string').eq('left_only'), 'station_code'].tolist() == ['Q', 'W']
    try:
        validated_station_merge(alternate_specimens, alternate_history[['station_code', 'station_name', 'region']])
    except pd.errors.MergeError:
        pass
    else:
        raise AssertionError('validated merge accepted a duplicated right key')

def _central_task2():
    north = pd.DataFrame(
        [
            ['rock', 'N9', 'C9', 3, 'A', 5.5],
            ['water', 'N2', 'C2', 1, 'B', 2.5],
        ],
        columns=['material', 'specimen_id', 'collector_id', 'collection_number', 'station_code', 'mass_g'],
    )
    south = pd.DataFrame(
        [['S8', 'D8', 5, 'C', 'air', 'manual review']],
        columns=['specimen_id', 'collector_id', 'collection_number', 'station_code', 'material', 'review_note'],
    )
    north_snapshot = north.copy(deep=True)
    south_snapshot = south.copy(deep=True)
    stacked = stack_specimen_partitions({'north_source': north, 'south_source': south})
    assert north.equals(north_snapshot) and south.equals(south_snapshot), 'stack mutated an input'
    assert stacked.columns.tolist() == ['material', 'specimen_id', 'collector_id', 'collection_number', 'station_code', 'mass_g', 'source_partition', 'review_note']
    assert stacked['specimen_id'].tolist() == ['N9', 'N2', 'S8']
    assert stacked['source_partition'].tolist() == ['north_source', 'north_source', 'south_source']
    assert str(stacked['source_partition'].dtype) == 'string'
    assert isinstance(stacked.index, pd.RangeIndex) and stacked.index.tolist() == [0, 1, 2]
    assert int(stacked['mass_g'].isna().sum()) == 1 and int(stacked['review_note'].isna().sum()) == 2
    collision = north.assign(source_partition='existing')
    try:
        stack_specimen_partitions({'bad': collision})
    except ValueError:
        pass
    else:
        raise AssertionError('reserved provenance collision was accepted')

    masses = pd.DataFrame({'specimen_id': ['M9', 'M7', 'M8'], 'mass_g': [4.0, 7.0, 8.0]})
    reviews = pd.DataFrame({'specimen_id': ['M8', 'M2'], 'review_score': [18.0, 12.0]})
    masses['specimen_id'] = masses['specimen_id'].astype('string')
    reviews['specimen_id'] = reviews['specimen_id'].astype('string')
    masses['mass_g'] = masses['mass_g'].astype('float64')
    reviews['review_score'] = reviews['review_score'].astype('float64')
    mass_snapshot = masses.copy(deep=True)
    review_snapshot = reviews.copy(deep=True)
    aligned = align_specimen_features(masses, reviews)
    assert masses.equals(mass_snapshot) and reviews.equals(review_snapshot), 'alignment mutated an input'
    assert aligned.index.name == 'specimen_id'
    assert aligned.index.tolist() == ['M9', 'M7', 'M8', 'M2']
    assert aligned.columns.tolist() == ['mass_g', 'review_score']
    assert pd.isna(aligned.loc['M9', 'review_score']) and pd.isna(aligned.loc['M2', 'mass_g'])
    assert aligned.loc['M8'].tolist() == [8.0, 18.0]

def _central_task3():
    alternate_wide = pd.DataFrame(
        [
            ['ZX3', 'C', 2.25, 9.75],
            ['ZX1', 'A', 4.25, 8.75],
            ['ZX2', 'B', 6.25, 7.75],
        ],
        columns=['sensor_id', 'station_code', 'baseline_value', 'followup_value'],
    )
    alternate_wide['sensor_id'] = alternate_wide['sensor_id'].astype('string')
    alternate_wide['station_code'] = alternate_wide['station_code'].astype('string')
    alternate_wide['baseline_value'] = alternate_wide['baseline_value'].astype('float64')
    alternate_wide['followup_value'] = alternate_wide['followup_value'].astype('float64')
    snapshot = alternate_wide.copy(deep=True)
    long = wide_to_long_scores(alternate_wide)
    assert alternate_wide.equals(snapshot), 'melt mutated its input'
    assert long.columns.tolist() == ['sensor_id', 'station_code', 'measurement_label', 'value']
    assert long['sensor_id'].tolist() == ['ZX3', 'ZX1', 'ZX2', 'ZX3', 'ZX1', 'ZX2']
    assert long['measurement_label'].tolist() == ['baseline_value'] * 3 + ['followup_value'] * 3
    long_snapshot = long.copy(deep=True)
    restored = long_to_wide_scores(long, alternate_wide.columns.tolist())
    assert long.equals(long_snapshot), 'pivot mutated its input'
    assert restored.columns.name is None
    pd.testing.assert_frame_equal(restored, alternate_wide)
    duplicate = pd.concat([long, long.iloc[[0]].copy()], ignore_index=True)
    assert int(duplicate.duplicated(['sensor_id', 'station_code', 'measurement_label'], keep=False).sum()) == 2
    try:
        long_to_wide_scores(duplicate, alternate_wide.columns.tolist())
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate long key did not reach pivot ValueError')

_central_record('task1', _central_task1)
_central_record('task2', _central_task2)
_central_record('task3', _central_task3)
(ASSIGNMENT_ROOT / '__central_checks.json').write_text(json.dumps(central_results), encoding='utf-8')
'''


def _execute_notebook(root: Path, cwd: Path, extra_source: str | None = None) -> None:
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    if extra_source is not None:
        notebook.cells.append(nbformat.v4.new_code_cell(extra_source, id="a06-central-checks"))
    previous = os.environ.get("PYTHONDONTWRITEBYTECODE")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        client = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(cwd)}},
            allow_errors=False,
        )
        client.execute()
    except CellExecutionError as error:
        lines = str(error).strip().splitlines()
        raise AssertionError("fresh notebook execution failed: " + " | ".join(lines[-10:])) from error
    finally:
        if previous is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = previous


def _read_alternate_results(root: Path) -> dict:
    path = root / "__central_checks.json"
    _assert(path.is_file(), "grader-owned alternate result was not written")
    result = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    _assert(set(result) == {"task1", "task2", "task3"}, "grader-owned alternate result malformed")
    return result


def _check_canonical_task1(root: Path) -> None:
    merged = pd.read_csv(
        root / "output" / "specimen_merge_audit.csv",
        dtype={
            "specimen_id": "string", "collector_id": "string",
            "collection_number": "int64", "station_code": "string",
            "material": "string", "mass_g": "float64", "station_name": "string",
            "region": "string", "_merge": "string",
        },
    )
    _assert(merged.columns.tolist() == [
        "specimen_id", "collector_id", "collection_number", "station_code",
        "material", "mass_g", "station_name", "region", "_merge",
    ], "merge artifact schema differs")
    _assert(merged["specimen_id"].tolist() == [f"SP{number}" for number in range(101, 108)], "specimen rows were not preserved")
    _assert(merged["_merge"].value_counts().to_dict() == {"both": 6, "left_only": 1}, "merge indicator counts differ")
    orphan = merged.loc[merged["_merge"].eq("left_only")]
    _assert(orphan[["specimen_id", "station_code"]].values.tolist() == [["SP106", "X"]], "wrong unmatched key")
    _assert(bool(orphan[["station_name", "region"]].isna().all().all()), "orphan metadata is not missing")
    _assert(merged.loc[merged["station_code"].eq("R"), "station_name"].eq("River Station").all(), "retired station selected")


def _check_canonical_task2(root: Path) -> None:
    combined = pd.read_csv(
        root / "output" / "combined_specimens.csv",
        dtype={
            "specimen_id": "string", "collector_id": "string",
            "collection_number": "int64", "station_code": "string",
            "material": "string", "mass_g": "float64", "source_partition": "string",
        },
    )
    _assert(combined["specimen_id"].tolist() == [f"SP{number}" for number in range(101, 108)], "stacked order differs")
    _assert(combined["source_partition"].tolist() == ["batch_a"] * 4 + ["batch_b"] * 3, "provenance differs")
    aligned = pd.read_csv(
        root / "output" / "aligned_features.csv",
        dtype={"specimen_id": "string", "mass_g": "float64", "review_score": "float64"},
        index_col="specimen_id",
    )
    _assert(aligned.index.name == "specimen_id" and aligned.index.tolist() == ["SP101", "SP102", "SP103", "SP108"], "aligned index differs")
    _assert(aligned.columns.tolist() == ["mass_g", "review_score"], "aligned columns differ")
    _assert(pd.isna(aligned.loc["SP101", "review_score"]) and pd.isna(aligned.loc["SP108", "mass_g"]), "alignment missingness differs")
    _assert(aligned.loc["SP102"].tolist() == [8.0, 7.0] and aligned.loc["SP103"].tolist() == [10.5, 9.0], "overlap values differ")


def _check_canonical_task3(root: Path) -> None:
    long = pd.read_csv(
        root / "output" / "sensor_scores_long.csv",
        dtype={"sensor_id": "string", "station_code": "string", "measurement_label": "string", "value": "float64"},
    )
    _assert(long.shape == (8, 4), "long shape differs")
    _assert(long["measurement_label"].tolist() == ["baseline_value"] * 4 + ["followup_value"] * 4, "melt order differs")
    _assert(not long.duplicated(["sensor_id", "station_code", "measurement_label"]).any(), "long key is not unique")
    wide = pd.read_csv(
        root / "output" / "sensor_scores_round_trip.csv",
        dtype={"sensor_id": "string", "station_code": "string", "baseline_value": "float64", "followup_value": "float64"},
    )
    source = pd.read_csv(
        root / "data" / "sensor_scores_wide.csv",
        dtype={"sensor_id": "string", "station_code": "string", "baseline_value": "float64", "followup_value": "float64"},
    )
    pd.testing.assert_frame_equal(wide, source)


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
        ("Task 1 automated", 40),
        ("Task 2 automated", 27),
        ("Task 3 automated", 23),
    )
    errors: dict[str, Exception | None] = {name: None for name, _ in test_specs}
    try:
        _check_static_contract(root)
    except Exception as error:
        for name in errors:
            errors[name] = error
    else:
        with tempfile.TemporaryDirectory(prefix="a06-central-") as temporary_name:
            temporary = Path(temporary_name)
            flat = temporary / "submission with spaces"
            _copy_submission(root, flat)
            output = flat / "output"
            output.mkdir(exist_ok=True)
            for name in ARTIFACT_SHA256:
                path = output / name
                if path.exists():
                    path.unlink()
                path.write_text("stale,artifact\n", encoding="utf-8")
            cwd = flat / "arbitrary" / "deep working directory"
            cwd.mkdir(parents=True)
            try:
                _execute_notebook(flat, cwd, GRADER_CHECKS_SOURCE)
                alternate = _read_alternate_results(flat)
            except Exception as error:
                for name in errors:
                    errors[name] = error
            else:
                for task_name, result_name, canonical_check in (
                    ("task1", "Task 1 automated", _check_canonical_task1),
                    ("task2", "Task 2 automated", _check_canonical_task2),
                    ("task3", "Task 3 automated", _check_canonical_task3),
                ):
                    try:
                        _assert(alternate[task_name]["passed"], alternate[task_name]["detail"])
                        canonical_check(flat)
                    except Exception as error:
                        errors[result_name] = error
                try:
                    fresh_bytes = _artifact_bytes(flat)
                    _assert(fresh_bytes == _artifact_bytes(root), "committed artifacts differ from fresh execution")
                    _execute_notebook(flat, cwd)
                    _assert(_artifact_bytes(flat) == fresh_bytes, "repeat execution is not deterministic")

                    course_root = temporary / "relocated course with spaces"
                    nested = course_root / "06" / "assignment"
                    nested.parent.mkdir(parents=True)
                    _copy_submission(root, nested)
                    for name in ARTIFACT_SHA256:
                        path = nested / "output" / name
                        if path.exists():
                            path.unlink()
                    _execute_notebook(nested, course_root)
                    _assert(_artifact_bytes(nested) == fresh_bytes, "course-root relocation differs")
                except Exception as error:
                    errors["Task 3 automated"] = error

    tests = [
        _result_test(name, points, errors[name]) for name, points in test_specs
    ]
    score = sum(test["score"] for test in tests)
    return {
        "schema": "datasci217/grading-result/v1",
        **context,
        "score": score,
        "max-score": 90,
        "tests": tests,
    }


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
