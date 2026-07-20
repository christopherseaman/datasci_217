# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==6.29.5",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.3",
# ]
# ///

"""Independent Classroom50 central-grader reference for Assignment 08.

This instructor-only module never imports the student-editable public checker.
It validates protected surfaces, fresh-executes stripped disposable notebook
copies, calls all five functions on disclosed alternate data, and emits the
official Classroom50 result contract. Human review remains outside the automated
90 points and uses the context-supplied ``review`` URL.
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
    "a08-header": "c49240238f3ebd296cf0c211170c56764f657972b9976a2a6c821d745b59b700",
    "a08-setup": "7d64bb798e93b090281c6f427dd10d113caffe31debf836eeefcb18b8b162778",
    "a08-terms-data": "0888ac5c6da29f2fc882d06cc707956518908ad339ce2a3319625100cc9e2d0c",
    "a08-task2-prompt": "c121bd5943bff1925995836ece9508768e394ff2da0d9d906ee7d319ccb972e6",
    "a08-task3-prompt": "45e6d964a75c3a478c63d4623416fec1f814c576ffa344f7b1e416618604ead0",
    "a08-final-verify": "70e228d9b9a14fb1a6f111a42acb08f78103f125f099a449b53e148278713b76",
}
PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "29cb7d486f7fab60576ddf66c0b0164fe830b43138feb1f4c7c6b7a8ecc6a4fb",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    "README.md": "c7382a76e6cce665176d8a3d65dfb2c103d65a70132b1b41ba68c7cc79079f32",
    "PLATFORM_CHECK.md": "d60455f2ea443990929cea97260c509399454e8bb839acc7043e60bbc3120b41",
    "check_assignment.py": "64256b16b0bae2a29192b2397ca52c80bac5c5e21a0edf32fefb4d53038d6144",
    "data/fixture.json": "b2fee1c48fb678b81318d2f085c42e2f9b480bd6c4eed6f07ef118b9bfd70860",
    "data/support_requests.csv": "a9136161332c5da9f8f1251d869bbd014ed762751675fb757f81a79cff5352d6",
}
ARTIFACT_SHA256 = {
    "center_count_summary.csv": "0735d0647dbbe2199b1de03e1061bf6c3a7a9d15bb553d128bdc1ab295ef2f36",
    "center_summary.csv": "6c528bd229cd0ce2db2f4c90f09fd2a9ba670fb3aa659951bc113d70a33afad4",
    "requests_with_context.csv": "391d56794e1537244c8d0b97f39e25e822b1e54d45b049fca98760ba646b1a7a",
    "center_channel_summary.csv": "41b74a8dac05eff1695e6b972b360bd2b1730e77f5e2060e42801533a07180da",
    "mean_resolution_pivot.csv": "1274782fc4e773bfd572736c0af106842d92751d672b6ad341207574e636dedf",
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/support_requests.csv",
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
REQUIRED_FUNCTIONS = {
    "build_count_summary", "build_center_summary", "add_center_context",
    "build_center_channel_summary", "build_resolution_pivot",
}
STUDENT_MARKDOWN_IDS = {
    "a08-task1-contract", "a08-task1-explain", "a08-task2-explain",
    "a08-task3-explain", "a08-synthesis",
}
STUDENT_CODE_IDS = set(EXPECTED_CELL_IDS) - MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
BANNED_ATTRIBUTES = {
    "apply", "filter", "crosstab", "merge", "join", "pivot", "melt",
    "stack", "unstack", "fillna", "ffill", "bfill", "interpolate",
    "drop_duplicates", "replace", "plot", "hist", "boxplot", "to_datetime",
    "to_period", "resample", "rolling", "expanding", "ewm", "shift",
    "asfreq", "corr", "cov",
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
    _assert(notebook.get("nbformat") == 4 and notebook.get("nbformat_minor") == 5, "invalid notebook format")
    _assert(isinstance(cells, list) and len(cells) == 25, "notebook must have exactly 25 cells")
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


def _function_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }


def _check_runtime_and_protected(root: Path) -> tuple[dict[str, dict], ast.AST]:
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
    for relative, expected in PROTECTED_FILE_SHA256.items():
        path = root / relative
        _assert(path.is_file(), f"missing protected file: {relative}")
        _assert(sha256(path.read_bytes()).hexdigest() == expected, f"protected file changed: {relative}")
    _assert((root / ".python-version").read_text() == "3.12.13\n", "wrong Python record")
    _assert((root / "requirements.txt").read_text() == "numpy==2.0.2\npandas==3.0.3\n", "wrong dependency records")
    gitignore = (root / ".gitignore").read_text()
    _assert("output/" not in gitignore and "*.csv" not in gitignore, "required CSV artifacts are ignored")
    _assert(
        {path.name for path in (root / "data").iterdir() if path.is_file()}
        == {"fixture.json", "support_requests.csv"},
        "fixture inventory changed",
    )
    for legacy in (
        "assignment.md", "data_generator.ipynb",
        "data_generator.md", "DATA_SCHEMA.md", "TIPS.md",
    ):
        _assert(not (root / legacy).exists(), f"legacy surface remains: {legacy}")
    _notebook, by_id = _load_notebook(root)
    for cell_id, expected in PROTECTED_CELL_SHA256.items():
        _assert(
            sha256(_cell_source(by_id[cell_id]).encode()).hexdigest() == expected,
            f"protected cell changed: {cell_id}",
        )
    student_markdown = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_IDS)
    student_code = "\n".join(_cell_source(by_id[cell_id]) for cell_id in STUDENT_CODE_IDS)
    _assert("TODO" not in student_markdown + student_code, "unfinished TODO remains")
    _assert("NotImplementedError" not in student_code, "starter scaffold remains")
    try:
        tree = ast.parse(student_code)
    except SyntaxError as error:
        raise AssertionError("student source has a syntax error") from error
    _assert(not any(isinstance(node, ast.Pass) for node in ast.walk(tree)), "scaffold pass remains")
    functions = _function_nodes(tree)
    _assert(REQUIRED_FUNCTIONS.issubset(functions), "required public function missing")
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
    for fragment in (
        "/content", "drive.mount", "files.upload", "http://", "https://",
        "urlopen", "urlretrieve", "requests.get", "requests.post",
        "requests.request", "requests.session", "random.", "datetime.now",
        "timestamp.now", "date.today", "read_html(",
    ):
        _assert(fragment not in lowered, f"nonportable or out-of-scope code: {fragment}")
    for line in student_code.splitlines():
        _assert(not line.lstrip().startswith(("!", "%")), "notebook magic or shell line used")
    return by_id, tree


def _check_task1_source(by_id: dict[str, dict], tree: ast.AST) -> None:
    functions = _function_nodes(tree)
    node = functions["build_count_summary"]
    groupbys = _calls(node, "groupby")
    _assert(len(groupbys) == 1, "count function must create one reusable GroupBy")
    for policy in ("observed", "sort", "dropna"):
        _assert(_literal_keyword(groupbys[0], policy) is True, f"count groupby must set {policy}=True")
    for operation in ("size", "count", "nunique", "concat", "reset_index"):
        _assert(_calls(node, operation), f"count function must use {operation}")
    source = ast.unparse(node)
    _assert("satisfaction_score" in source and "agent_id" in source, "wrong selected count columns")
    _assert(_literal_keyword(_calls(node, "nunique")[0], "dropna") is True, "nunique must set dropna=True")
    for canonical in ("Central", "Harbor", "Ridge", "Valley", "Q001"):
        _assert(canonical not in source, "count function hard-codes canonical data")
    _assert(not _calls(node, "read_csv") and not _calls(node, "to_csv"), "count function performs file I/O")
    values = _cell_source(by_id["a08-task1-values"])
    for required in (
        "one row per synthetic support request", "one observed support center",
        "Central", "Harbor", "Ridge", "one row per observed support center",
        "request_count", "satisfaction_count", "unique_agent_count",
    ):
        _assert(required in values, f"Task 1 machine-readable contract missing {required}")


def _check_task2_source(_by_id: dict[str, dict], tree: ast.AST) -> None:
    functions = _function_nodes(tree)
    for function_name in (
        "build_center_summary", "add_center_context",
        "build_center_channel_summary",
    ):
        node = functions[function_name]
        groupbys = _calls(node, "groupby")
        _assert(len(groupbys) == 1, f"{function_name} must contain one groupby")
        for policy in ("observed", "sort", "dropna"):
            _assert(_literal_keyword(groupbys[0], policy) is True, f"{function_name} must set {policy}=True")
        for canonical in ("Central", "Harbor", "Ridge", "Valley", "Q001"):
            _assert(canonical not in ast.unparse(node), f"{function_name} hard-codes canonical data")
        _assert(not _calls(node, "read_csv") and not _calls(node, "to_csv"), f"{function_name} performs file I/O")
    for function_name in ("build_center_summary", "build_center_channel_summary"):
        node = functions[function_name]
        groupby = _calls(node, "groupby")[0]
        _assert(_literal_keyword(groupby, "as_index") is False, f"{function_name} must use as_index=False")
        _assert(len(_calls(node, "agg")) == 1, f"{function_name} must use one named aggregation")
    center_source = ast.unparse(functions["build_center_summary"])
    for name in (
        "request_count", "satisfaction_count", "unique_agent_count",
        "total_resolution_minutes", "mean_resolution_minutes",
    ):
        _assert(name in center_source, f"center named aggregation missing {name}")
    context = functions["add_center_context"]
    transforms = _calls(context, "transform")
    _assert(
        len(transforms) == 1 and transforms[0].args
        and isinstance(transforms[0].args[0], ast.Constant)
        and transforms[0].args[0].value == "mean",
        "context must use exactly transform('mean')",
    )
    _assert(_calls(context, "copy"), "context function must copy its input")


def _check_task3_source(by_id: dict[str, dict], tree: ast.AST) -> None:
    functions = _function_nodes(tree)
    pivot_calls = _calls(tree, "pivot_table")
    _assert(len(pivot_calls) == 1, "student code must contain exactly one pivot_table")
    call = pivot_calls[0]
    _assert(isinstance(call.func.value, ast.Name) and call.func.value.id == "pd", "must call pd.pivot_table")
    expected = {
        "index": "center", "columns": "channel", "values": "resolution_minutes",
        "aggfunc": "mean", "observed": True, "sort": True, "dropna": True,
    }
    for keyword, value in expected.items():
        _assert(_literal_keyword(call, keyword) == value, f"pivot must set {keyword}={value!r}")
    _assert(len(_calls(functions["build_resolution_pivot"], "pivot_table")) == 1, "sole pivot belongs in pivot function")
    node = functions["build_resolution_pivot"]
    for canonical in ("Central", "Harbor", "Ridge", "Valley", "Q001"):
        _assert(canonical not in ast.unparse(node), "pivot function hard-codes canonical data")
    _assert(not _calls(node, "read_csv") and not _calls(node, "to_csv"), "pivot function performs file I/O")
    run_source = _cell_source(by_id["a08-task3-run"])
    for evidence in (
        "pivot_reference", "build_center_channel_summary", "itertuples",
        "mean_resolution_minutes", "resolution_pivot.loc",
    ):
        _assert(evidence in run_source, f"Task 3 cell-for-cell comparison missing {evidence}")


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
        central_results[name] = {'passed': True, 'detail': 'canonical and alternate behavioral checks passed'}

def _alternate_requests():
    table = pd.DataFrame(
        [
            ['Z06', 'Coast', 'B3', 'Desk', 35, pd.NA],
            ['Z01', 'Metro', 'B1', 'Web', 10, 5],
            ['Z09', 'Hill', 'B6', 'Voice', 70, 4],
            ['Z04', 'Metro', 'B2', 'Web', 14, 3],
            ['Z08', 'Hill', 'B5', 'Web', 50, 3],
            ['Z02', 'Metro', 'B1', 'Voice', 30, pd.NA],
            ['Z10', 'Hill', 'B5', 'Desk', 60, 5],
            ['Z05', 'Coast', 'B3', 'Web', 25, 4],
            ['Z03', 'Metro', 'B2', 'Desk', 20, 4],
            ['Z07', 'Coast', 'B4', 'Desk', 45, 5],
        ],
        index=[42, 5, 91, 12, 63, 8, 77, 24, 3, 55],
        columns=['request_id', 'center', 'agent_id', 'channel', 'resolution_minutes', 'satisfaction_score'],
    )
    table['request_id'] = table['request_id'].astype('string')
    table['center'] = pd.Categorical(table['center'], ['Metro', 'Coast', 'Hill', 'Plains'], ordered=True)
    table['agent_id'] = table['agent_id'].astype('string')
    table['channel'] = pd.Categorical(table['channel'], ['Web', 'Voice', 'Desk'], ordered=True)
    table['resolution_minutes'] = table['resolution_minutes'].astype('int64')
    table['satisfaction_score'] = table['satisfaction_score'].astype('Int64')
    return table

def _central_task1():
    assert input_row_grain == 'one row per synthetic support request'
    assert grouping_key == ['center']
    assert grouping_unit == 'one observed support center'
    assert predicted_group_identities == ['Central', 'Harbor', 'Ridge']
    assert predicted_group_count == 3 and observed_category_policy is True
    assert output_row_grain == 'one row per observed support center'
    assert count_plan == {
        'request_count': {
            'question': 'How many support-request rows were recorded?',
            'operation': 'size',
        },
        'satisfaction_count': {
            'question': 'How many requests have a recorded satisfaction score?',
            'operation': 'count',
        },
        'unique_agent_count': {
            'question': 'How many distinct agents appear?',
            'operation': 'nunique',
        },
    }
    alternate = _alternate_requests()
    snapshot = alternate.copy(deep=True)
    result = build_count_summary(alternate)
    pd.testing.assert_frame_equal(alternate, snapshot)
    assert result.columns.tolist() == ['center', 'request_count', 'satisfaction_count', 'unique_agent_count']
    assert result['center'].tolist() == ['Metro', 'Coast', 'Hill']
    assert result['request_count'].tolist() == [4, 3, 3]
    assert result['satisfaction_count'].tolist() == [3, 2, 3]
    assert result['unique_agent_count'].tolist() == [2, 2, 2]
    assert str(result['request_count'].dtype) == 'int64'
    assert str(result['satisfaction_count'].dtype) == 'Int64'
    assert str(result['unique_agent_count'].dtype) == 'int64'
    assert int(result['request_count'].sum()) == len(alternate)
    assert 'Plains' not in result['center'].tolist()

def _central_task2():
    alternate = _alternate_requests()
    snapshot = alternate.copy(deep=True)
    summary = build_center_summary(alternate)
    context = add_center_context(alternate)
    two_key = build_center_channel_summary(alternate)
    pd.testing.assert_frame_equal(alternate, snapshot)
    assert summary.columns.tolist() == [
        'center', 'request_count', 'satisfaction_count', 'unique_agent_count',
        'total_resolution_minutes', 'mean_resolution_minutes',
    ]
    assert summary['center'].tolist() == ['Metro', 'Coast', 'Hill']
    assert summary['request_count'].tolist() == [4, 3, 3]
    assert summary['satisfaction_count'].tolist() == [3, 2, 3]
    assert summary['unique_agent_count'].tolist() == [2, 2, 2]
    assert summary['total_resolution_minutes'].tolist() == [74, 105, 180]
    assert summary['mean_resolution_minutes'].tolist() == [18.5, 35.0, 60.0]
    assert context.index.tolist() == [42, 5, 91, 12, 63, 8, 77, 24, 3, 55]
    assert context['center_mean_resolution_minutes'].tolist() == [35.0, 18.5, 60.0, 18.5, 60.0, 18.5, 60.0, 35.0, 18.5, 35.0]
    assert context['difference_from_center_mean'].tolist() == [0.0, -8.5, 10.0, -4.5, -10.0, 11.5, 0.0, -10.0, 1.5, 10.0]
    assert two_key.columns.tolist() == ['center', 'channel', 'request_count', 'mean_resolution_minutes']
    assert two_key[['center', 'channel']].values.tolist() == [
        ['Metro', 'Web'], ['Metro', 'Voice'], ['Metro', 'Desk'],
        ['Coast', 'Web'], ['Coast', 'Desk'],
        ['Hill', 'Web'], ['Hill', 'Voice'], ['Hill', 'Desk'],
    ]
    assert two_key['request_count'].tolist() == [2, 1, 1, 1, 2, 1, 1, 1]
    assert two_key['mean_resolution_minutes'].tolist() == [12.0, 30.0, 20.0, 25.0, 40.0, 50.0, 70.0, 60.0]
    assert int(two_key['request_count'].sum()) == len(alternate)
    assert ['Coast', 'Voice'] not in two_key[['center', 'channel']].values.tolist()
    assert not isinstance(two_key.index, pd.MultiIndex)

def _central_task3():
    assert pivot_spec == {
        'index': 'center', 'columns': 'channel',
        'values': 'resolution_minutes', 'aggfunc': 'mean',
        'observed': True, 'sort': True, 'dropna': True,
    }
    assert pivot_display_row_grain == 'one observed support center'
    assert pivot_cell_grain == 'one observed center-channel group'
    assert absent_combination == ['Harbor', 'Phone']
    assert absent_combination_meaning == 'no input row for this center-channel combination'
    alternate = _alternate_requests()
    snapshot = alternate.copy(deep=True)
    pivot = build_resolution_pivot(alternate)
    reference = build_center_channel_summary(alternate)
    pd.testing.assert_frame_equal(alternate, snapshot)
    assert pivot.index.tolist() == ['Metro', 'Coast', 'Hill']
    assert pivot.columns.tolist() == ['Web', 'Voice', 'Desk']
    assert pivot.shape == (3, 3) and int(pivot.notna().sum().sum()) == 8
    assert pd.isna(pivot.loc['Coast', 'Voice']) and 'Plains' not in pivot.index
    assert not (pivot == 0).any().any()
    for row in reference.itertuples(index=False):
        assert pivot.loc[row.center, row.channel] == row.mean_resolution_minutes

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
        notebook.cells.append(nbformat.v4.new_code_cell(extra_source, id="a08-central-checks"))
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


def _execute_setup_only(root: Path, cwd: Path) -> None:
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    setup = next(cell for cell in notebook.cells if cell.get("id") == "a08-setup")
    stripped = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(setup.source, id="a08-setup-only")],
        metadata=notebook.metadata,
    )
    NotebookClient(
        stripped,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
        allow_errors=False,
    ).execute()


def _read_alternate_results(root: Path) -> dict:
    path = root / "__central_checks.json"
    _assert(path.is_file(), "grader-owned alternate result was not written")
    result = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    _assert(set(result) == {"task1", "task2", "task3"}, "grader-owned alternate result malformed")
    return result


def _read_count(root: Path) -> pd.DataFrame:
    return pd.read_csv(
        root / "output" / "center_count_summary.csv",
        dtype={
            "center": "string", "request_count": "int64",
            "satisfaction_count": "Int64", "unique_agent_count": "int64",
        },
    )


def _check_canonical_task1(root: Path) -> None:
    result = _read_count(root)
    _assert(result.columns.tolist() == ["center", "request_count", "satisfaction_count", "unique_agent_count"], "Task 1 schema differs")
    _assert(result["center"].tolist() == ["Central", "Harbor", "Ridge"], "Task 1 group order differs")
    _assert(result["request_count"].tolist() == [5, 5, 5], "size results differ")
    _assert(result["satisfaction_count"].tolist() == [4, 3, 5], "count results differ")
    _assert(result["unique_agent_count"].tolist() == [3, 2, 3], "nunique results differ")
    _assert(int(result["request_count"].sum()) == 15, "Task 1 rows do not conserve source requests")


def _check_canonical_task2(root: Path) -> None:
    summary = pd.read_csv(
        root / "output" / "center_summary.csv",
        dtype={
            "center": "string", "request_count": "int64",
            "satisfaction_count": "Int64", "unique_agent_count": "int64",
            "total_resolution_minutes": "int64", "mean_resolution_minutes": "float64",
        },
    )
    _assert(summary["total_resolution_minutes"].tolist() == [180, 200, 210], "center totals differ")
    _assert(summary["mean_resolution_minutes"].tolist() == [36.0, 40.0, 42.0], "center means differ")
    context = pd.read_csv(
        root / "output" / "requests_with_context.csv",
        dtype={
            "request_id": "string", "center": "string", "agent_id": "string",
            "channel": "string", "resolution_minutes": "int64",
            "satisfaction_score": "Int64", "center_mean_resolution_minutes": "float64",
            "difference_from_center_mean": "float64",
        },
    )
    _assert(len(context) == 15 and context["request_id"].tolist() == [f"Q{number:03d}" for number in range(1, 16)], "context row grain/order differs")
    _assert(context["center_mean_resolution_minutes"].tolist() == [36.0] * 5 + [40.0] * 5 + [42.0] * 5, "transform means differ")
    two = pd.read_csv(
        root / "output" / "center_channel_summary.csv",
        dtype={
            "center": "string", "channel": "string",
            "request_count": "int64", "mean_resolution_minutes": "float64",
        },
    )
    _assert(len(two) == 8 and int(two["request_count"].sum()) == 15, "two-key row conservation differs")
    _assert(["Harbor", "Phone"] not in two[["center", "channel"]].values.tolist(), "absent two-key combination materialized")


def _check_canonical_task3(root: Path) -> None:
    pivot = pd.read_csv(
        root / "output" / "mean_resolution_pivot.csv",
        dtype={"center": "string", "Email": "float64", "Phone": "float64", "Chat": "float64"},
    ).set_index("center")
    _assert(pivot.index.tolist() == ["Central", "Harbor", "Ridge"], "pivot row order differs")
    _assert(pivot.columns.tolist() == ["Email", "Phone", "Chat"], "pivot column order differs")
    _assert(int(pivot.notna().sum().sum()) == 8 and pd.isna(pivot.loc["Harbor", "Phone"]), "pivot occupancy differs")
    two = pd.read_csv(
        root / "output" / "center_channel_summary.csv",
        dtype={"center": "string", "channel": "string", "request_count": "int64", "mean_resolution_minutes": "float64"},
    )
    for row in two.itertuples(index=False):
        _assert(pivot.loc[row.center, row.channel] == row.mean_resolution_minutes, "GroupBy-to-pivot cell differs")


def _check_committed_artifacts(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "missing regular output directory")
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(actual == set(ARTIFACT_SHA256) | {".gitkeep"}, "submission must contain exactly five artifacts plus .gitkeep")
    for name, expected in ARTIFACT_SHA256.items():
        data = (output / name).read_bytes()
        _assert(data.endswith(b"\n") and b"\r" not in data, f"wrong line endings: {name}")
        _assert(sha256(data).hexdigest() == expected, f"committed artifact differs: {name}")


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
    """Grade one local submission and return a Classroom50 result object."""

    context = _context()
    root = Path(submission_root).resolve()
    specs = (
        ("Template, environment, fixture, notebook, and protected integrity", 10),
        ("Task 1 grain prediction and count semantics", 20),
        ("Task 2 aggregation, transform, and two-key result", 35),
        ("Task 3 pivot equivalence and absent combination", 20),
        ("Portability, visible outputs, repeatability, and resubmission", 5),
    )
    errors: dict[str, Exception | None] = {name: None for name, _ in specs}
    template_name, task1_name, task2_name, task3_name, portability_name = [name for name, _ in specs]
    try:
        by_id, tree = _check_runtime_and_protected(root)
    except Exception as error:
        for name in errors:
            errors[name] = error
    else:
        for name, check in (
            (task1_name, _check_task1_source),
            (task2_name, _check_task2_source),
            (task3_name, _check_task3_source),
        ):
            try:
                check(by_id, tree)
            except Exception as error:
                errors[name] = error
        try:
            _check_committed_artifacts(root)
        except Exception as error:
            errors[portability_name] = error

        with tempfile.TemporaryDirectory(prefix="a08-central-") as temporary_name:
            temporary = Path(temporary_name)
            flat = temporary / "standalone submission with spaces"
            _copy_submission(root, flat)
            output = flat / "output"
            output.mkdir(exist_ok=True)
            for name in ARTIFACT_SHA256:
                path = output / name
                if path.exists():
                    path.unlink()
                path.write_text("stale,artifact\n", encoding="utf-8")
            nested_cwd = flat / "nested" / "working directory"
            nested_cwd.mkdir(parents=True)
            try:
                _execute_notebook(flat, nested_cwd, GRADER_CHECKS_SOURCE)
                alternate = _read_alternate_results(flat)
            except Exception as error:
                for name in (task1_name, task2_name, task3_name, portability_name):
                    if errors[name] is None:
                        errors[name] = error
            else:
                for key, name, canonical_check in (
                    ("task1", task1_name, _check_canonical_task1),
                    ("task2", task2_name, _check_canonical_task2),
                    ("task3", task3_name, _check_canonical_task3),
                ):
                    if errors[name] is None:
                        try:
                            _assert(alternate[key]["passed"], alternate[key]["detail"])
                            canonical_check(flat)
                        except Exception as error:
                            errors[name] = error
                if errors[portability_name] is None:
                    try:
                        fresh_bytes = _artifact_bytes(flat)
                        _assert(fresh_bytes == _artifact_bytes(root), "committed artifacts differ from fresh execution")
                        _execute_notebook(flat, nested_cwd)
                        _assert(_artifact_bytes(flat) == fresh_bytes, "repeat execution is not deterministic")

                        course_root = temporary / "relocated course with spaces"
                        nested = course_root / "08" / "assignment"
                        nested.parent.mkdir(parents=True)
                        _copy_submission(root, nested)
                        for artifact in ARTIFACT_SHA256:
                            path = nested / "output" / artifact
                            if path.exists():
                                path.unlink()
                        _execute_notebook(nested, course_root)
                        _assert(_artifact_bytes(nested) == fresh_bytes, "course-root relocation differs")

                        standalone = temporary / "relocated standalone"
                        _copy_submission(root, standalone)
                        for artifact in ARTIFACT_SHA256:
                            path = standalone / "output" / artifact
                            if path.exists():
                                path.unlink()
                        _execute_notebook(standalone, standalone)
                        _assert(_artifact_bytes(standalone) == fresh_bytes, "standalone relocation differs")
                    except Exception as error:
                        errors[portability_name] = error

    tests = [_result_test(name, maximum, errors[name]) for name, maximum in specs]
    score = sum(test["score"] for test in tests)
    return {
        "schema": "classroom50/result/v1",
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
    except Exception as error:
        print(f"Grader infrastructure failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
