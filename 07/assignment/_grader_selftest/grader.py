# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "numpy==2.0.2",
#   "pandas==3.0.5",
#   "matplotlib==3.11.1",
#   "seaborn==0.13.2",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "ipykernel==6.29.5",
#   "Pillow==12.3.0",
# ]
# ///

"""Independent central-grader candidate for Assignment 07."""

from __future__ import annotations

import ast
import datetime
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import matplotlib
from PIL import Image
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import numpy as np
import pandas as pd
import seaborn as sns


EXPECTED_CELL_IDS = [
    "a07-header", "a07-setup", "a07-terms-data", "a07-load",
    "a07-task1-contract", "a07-task1-evidence", "a07-explore-function",
    "a07-explore-run", "a07-task1-reflection", "a07-task2-context",
    "a07-supplied-flawed", "a07-task2-critique", "a07-critique-evidence",
    "a07-redesign-function", "a07-redesign-run", "a07-task3-contract",
    "a07-final-contract-values", "a07-supporting-data",
    "a07-explanatory-function", "a07-explanatory-run",
    "a07-evidence-export", "a07-visual-review", "a07-final-verify",
]
MARKDOWN_IDS = {
    "a07-header", "a07-terms-data", "a07-task1-contract",
    "a07-task1-reflection", "a07-task2-context", "a07-task2-critique",
    "a07-task3-contract", "a07-visual-review",
}
PROTECTED_CELL_SHA256 = {
    "a07-header": "98a0484f819c84e29bd6dc972e5e26b473d5dc840a2344615ef5c58f53691943",
    "a07-setup": "8e898c45d900d895f092d3fb235460d66e460391bdb83e32ae27cf38da0ec3ff",
    "a07-terms-data": "f762645fcbbe9d28dbd3d77ae4d124baa9030fdc41a1295be3a5b4d9634dcad4",
    "a07-task2-context": "7ddee594b77784ea7b2684f82f1fb1215bbdf79224ffd153a2269bfec3278fa2",
    "a07-supplied-flawed": "96bfd9dd6114ad84f305dc8567e757ac8ce33cfed55c546447f763bb73bf867b",
    "a07-final-verify": "990573f70aa3ef7a8cd27cc5db117ffe2c50b55ad1f8cbb7615b46f2c7f2c22d",
}
STUDENT_MARKDOWN_IDS = MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
STUDENT_CODE_IDS = set(EXPECTED_CELL_IDS) - MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "5072907d928869027f0ab9884599bce2fda548faad48bf8c766bd58998655763",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    "README.md": "d9bb0b8bb9699a2fee850e72a05e4e69cf5f26af1f74ed64a61b399ab3fe1481",
    "PLATFORM_CHECK.md": "12e9b9ab5ad249c23f50b56c8d393c24af65796e66ad3345df3c18e19f85b7f6",
    "check_assignment.py": "5143858feea168e921ef8f40a8c095f597d0a651da6f083e154d69a97d5789a2",
    "data/fixture.json": "1c3397cb2d98ae239f6a7cd254bb3aa9980d94cd23af4546c834a9262de0a28c",
    "data/format_completion.csv": "20ad900633154f5f3a2c09cfbc2f890f8423da0897d6345841745332110be66a",
    "data/pathway_checkpoints.csv": "ec9a336b7fb97418a6f058704f2509c8cee6b13d744efb7a6e3e99224ef8c258",
    "data/session_observations.csv": "fc4d69ab836288a2fe9c505c65c08413e137e51ab1914cd0e350f6e6636da096",
}
OUTPUT_NAMES = {
    "critique_redesign.png", "pathway_explanatory.png",
    "explanatory_supporting_data.csv", "visualization_evidence.json",
    "explanatory_text_alternative.txt",
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/format_completion.csv",
    "data/pathway_checkpoints.csv", "data/session_observations.csv",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}
SUPPORTING_HASH = "ec9a336b7fb97418a6f058704f2509c8cee6b13d744efb7a6e3e99224ef8c258"
REQUIRED_FUNCTIONS = {
    "build_exploratory_chart", "build_critique_redesign",
    "build_explanatory_chart",
}
BANNED_CALLS = {
    "agg", "aggregate", "groupby", "pivot_table", "transform",
    "corr", "corrwith", "cov", "describe", "mean", "median", "sum",
    "std", "var", "sem", "quantile", "nunique", "value_counts",
    "merge", "join", "concat", "melt", "pivot", "crosstab",
    "bfill", "drop", "drop_duplicates", "dropna", "ffill", "fillna",
    "interpolate", "replace", "to_datetime", "astype", "resample",
    "rolling", "expanding", "ewm", "shift", "pie", "heatmap", "kdeplot",
    "regplot", "lmplot", "violinplot", "pairplot", "jointplot", "load_dataset",
}
REQUIRED_CONTEXT_ENV = {
    "assignment": "ASSIGNMENT",
    "submission": "SUBMISSION_TAG",
    "commit": "COMMIT_URL",
    "release": "RELEASE_URL",
}
TEST_SPECS = (
    ("Fixtures and reproducibility", 10),
    ("Task 1 bounded exploration", 15),
    ("Task 2 critique and redesign", 25),
    ("Task 3 explanatory evidence", 25),
    ("Artifact integrity", 5),
)


class InfrastructureError(RuntimeError):
    """Raised only for grader platform/context/storage failures."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _context() -> dict[str, str]:
    context: dict[str, str] = {}
    missing: list[str] = []
    for key, environment_name in REQUIRED_CONTEXT_ENV.items():
        value = os.environ.get(environment_name, "").strip()
        if not value:
            missing.append(environment_name)
        context[key] = value
    if missing:
        raise InfrastructureError("missing required grading context: " + ", ".join(missing))
    context["review"] = os.environ.get("REVIEW_URL", "").strip() or context["commit"]
    context["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return context


def _load_notebook(root: Path) -> tuple[dict, dict[str, dict]]:
    try:
        notebook = json.loads((root / "assignment.ipynb").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError("assignment.ipynb is not valid UTF-8 notebook JSON") from error
    cells = notebook.get("cells")
    _assert(notebook.get("nbformat") == 4 and notebook.get("nbformat_minor") == 5, "notebook format changed")
    _assert(isinstance(cells, list) and len(cells) == 23, "cell count changed")
    ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    _assert(ids == EXPECTED_CELL_IDS and len(ids) == len(set(ids)), "cell IDs/order changed")
    for cell in cells:
        expected = "markdown" if cell["id"] in MARKDOWN_IDS else "code"
        _assert(cell.get("cell_type") == expected, f"cell type changed: {cell['id']}")
    _assert(
        notebook.get("metadata", {}).get("kernelspec")
        == {"display_name": "Python 3", "language": "python", "name": "python3"},
        "portable kernelspec changed",
    )
    return notebook, {cell["id"]: cell for cell in cells}


def _check_source(by_id: dict[str, dict]) -> None:
    for cell_id, expected in PROTECTED_CELL_SHA256.items():
        _assert(sha256(_source(by_id[cell_id]).encode()).hexdigest() == expected, f"protected cell changed: {cell_id}")
    student_markdown = "\n".join(_source(by_id[cell_id]) for cell_id in STUDENT_MARKDOWN_IDS)
    student_code = "\n".join(_source(by_id[cell_id]) for cell_id in STUDENT_CODE_IDS)
    _assert("TODO" not in student_markdown + student_code, "unfinished TODO remains")
    _assert("NotImplementedError" not in student_code, "starter scaffold remains")
    try:
        tree = ast.parse(student_code)
    except SyntaxError as error:
        raise AssertionError(f"student source syntax error: {error}") from error
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    _assert(REQUIRED_FUNCTIONS.issubset(functions), "required chart function missing")
    for node in ast.walk(tree):
        _assert(not isinstance(node, (ast.Import, ast.ImportFrom)), "student import used")
        if isinstance(node, ast.Call):
            _assert(_call_name(node) not in BANNED_CALLS, f"out-of-scope API used: {_call_name(node)}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            absolute = value not in {"/", "//", "\\", "\\\\"} and (
                value.startswith(("/", "~")) or (
                len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"}
                )
            )
            _assert(not absolute, f"absolute path literal used: {value!r}")
    lowered = student_code.lower()
    for fragment in (
        "/content", "drive.mount", "files.upload", "http://", "https://",
        "urlopen", "requests.", "np.random", "datetime.now", "plotly",
        "altair", "bokeh", "holoviews", "streamlit", "animation",
    ):
        _assert(fragment not in lowered, f"nonportable/out-of-scope code used: {fragment}")
    for line in student_code.splitlines():
        _assert(not line.lstrip().startswith(("!", "%")), "shell or magic line used")
    load_tree = ast.parse(_source(by_id["a07-load"]))
    reads = [node for node in ast.walk(load_tree) if isinstance(node, ast.Call) and _call_name(node) == "read_csv"]
    _assert(len(reads) == 3, "three fixtures must be read exactly once")
    _assert(not any(isinstance(node, ast.Call) and _call_name(node) == "DataFrame" for node in ast.walk(load_tree)), "replacement fixture embedded")
    for filename in ("format_completion.csv", "pathway_checkpoints.csv", "session_observations.csv"):
        _assert(filename in _source(by_id["a07-load"]), f"fixture not loaded: {filename}")
    explore_tree = ast.parse(_source(by_id["a07-explore-function"]))
    _assert(sum(1 for node in ast.walk(explore_tree) if isinstance(node, ast.Call) and _call_name(node) == "scatterplot") == 1, "exploration needs exactly one scatterplot call")
    _assert(not any(isinstance(node, ast.Call) and _call_name(node) == "savefig" for node in ast.walk(explore_tree)), "exploration saved an unrequested image")


def _output_inventory(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "missing output directory")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _assert(actual == {".gitkeep", *OUTPUT_NAMES}, "output inventory changed")
    _assert(not any((output / name).is_symlink() for name in OUTPUT_NAMES), "output symlink used")


def _check_image(path: Path) -> tuple[int, int]:
    _assert(path.is_file() and not path.is_symlink(), f"missing {path.name}")
    _assert(10_000 <= path.stat().st_size <= 2_000_000, f"implausible bytes: {path.name}")
    try:
        with Image.open(path) as image:
            _assert(image.format == "PNG", f"not PNG: {path.name}")
            size = image.size
            image.verify()
        with Image.open(path) as image:
            converted = image.convert("RGB")
            extrema = converted.getextrema()
    except Exception as error:
        raise AssertionError(f"PNG decode failed: {path.name}: {error}") from error
    _assert(800 <= size[0] <= 2000 and 450 <= size[1] <= 1400, f"implausible dimensions: {path.name}")
    _assert(any(low != high for low, high in extrema), f"uniform-pixel proxy failed: {path.name}")
    return size


def _check_evidence(root: Path) -> dict:
    output = root / "output"
    supporting = (output / "explanatory_supporting_data.csv").read_bytes()
    _assert(sha256(supporting).hexdigest() == SUPPORTING_HASH, "supporting CSV differs")
    raw = (output / "visualization_evidence.json").read_bytes()
    _assert(raw.endswith(b"\n") and b"\r" not in raw, "evidence JSON line endings differ")
    evidence = json.loads(raw.decode("utf-8"))
    _assert(list(evidence) == [
        "schema", "question", "audience", "intended_claim", "displayed_unit",
        "grain", "variable_roles", "exploration", "critique", "text_alternative",
    ], "evidence topology differs")
    _assert(evidence["schema"] == "datasci217/a07-visualization-evidence/v1", "evidence schema differs")
    _assert(evidence["displayed_unit"] == "prepared completion percent", "displayed unit differs")
    _assert(evidence["grain"] == "one row per pathway and checkpoint", "grain differs")
    _assert(evidence["variable_roles"] == {"pathway": "categorical", "checkpoint_number": "ordered", "completion_percent": "quantitative"}, "roles differ")
    exploration = evidence["exploration"]
    _assert(list(exploration) == ["question", "grain", "variable_roles", "observation", "limitation"], "exploration topology differs")
    _assert(exploration["grain"] == "one row per synthetic learning session", "exploration grain differs")
    categories = ["unsupported claim", "truncated baseline", "missing unit", "color-only encoding", "distracting decoration"]
    _assert([entry.get("category") for entry in evidence["critique"]] == categories, "critique categories differ")
    _assert(all(list(entry) == ["category", "problem", "repair"] for entry in evidence["critique"]), "critique entry topology differs")
    for value in [evidence["question"], evidence["audience"], evidence["intended_claim"], evidence["text_alternative"], exploration["question"], exploration["observation"], exploration["limitation"]]:
        _assert(isinstance(value, str) and value.strip(), "authored evidence text is empty")
    _assert(raw == (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode(), "evidence serialization differs")
    text = (output / "explanatory_text_alternative.txt").read_bytes()
    _assert(text == (evidence["text_alternative"] + "\n").encode(), "text alternative differs from JSON")
    return evidence


def _check_static(root: Path) -> tuple[dict, dict[str, dict]]:
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
        _assert(path.is_file() and not path.is_symlink(), f"missing protected file: {relative}")
        _assert(sha256(path.read_bytes()).hexdigest() == expected, f"protected file changed: {relative}")
    data_files = {path.relative_to(root / "data").as_posix() for path in (root / "data").rglob("*") if path.is_file()}
    _assert(data_files == {"fixture.json", "format_completion.csv", "pathway_checkpoints.csv", "session_observations.csv"}, "fixture inventory changed")
    notebook, by_id = _load_notebook(root)
    _check_source(by_id)
    _output_inventory(root)
    for name in ("critique_redesign.png", "pathway_explanatory.png"):
        _check_image(root / "output" / name)
    _check_evidence(root)
    return notebook, by_id


def _check_grader_runtime() -> None:
    expected = {
        "numpy": "2.0.2", "pandas": "3.0.5", "matplotlib": "3.11.1",
        "seaborn": "0.13.2", "nbclient": "0.10.2", "nbformat": "5.10.4",
        "ipykernel": "6.29.5", "Pillow": "12.3.0",
    }
    observed = {package: metadata.version(package) for package in expected}
    if sys.version_info[:3] != (3, 12, 13) or observed != expected:
        raise InfrastructureError(
            f"grader runtime mismatch: Python {sys.version_info[:3]}, packages {observed}"
        )


def _copy_submission(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {"_grader_selftest", ".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", "result.json", ".git"}.intersection(names)
    shutil.copytree(source, destination, ignore=ignore)


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {name: (root / "output" / name).read_bytes() for name in OUTPUT_NAMES}


GRADER_CHECKS_SOURCE = r'''
central_results = {}

def _central_record(name, function):
    try:
        function()
    except Exception as error:
        central_results[name] = {'passed': False, 'detail': f'{type(error).__name__}: {error}'}
    else:
        central_results[name] = {'passed': True, 'detail': 'fresh object and alternate-data checks passed'}

def _marker_geometries(axes):
    geometries = set()
    for collection in axes.collections:
        for path in collection.get_paths():
            vertices = tuple(tuple(round(float(value), 6) for value in row) for row in path.vertices)
            codes = tuple(path.codes.tolist()) if path.codes is not None else ()
            if vertices:
                geometries.add((vertices, codes))
    return geometries

def _task1():
    assert exploratory_figure.axes == [exploratory_axes]
    offsets = exploratory_axes.collections[0].get_offsets()
    assert len(offsets) == 12
    assert exploratory_axes.get_xlabel() == 'Activities completed (count)'
    assert exploratory_axes.get_ylabel() == 'Reflection score (points)'
    assert exploratory_axes.get_title() == 'Exploratory view of activities completed and reflection score'
    assert exploratory_axes.get_legend().get_title().get_text() == 'Pathway'
    assert len(_marker_geometries(exploratory_axes)) == 2
    assert not (ASSIGNMENT_ROOT / 'output' / 'exploratory.png').exists()

    alternate = pd.DataFrame({
        'session_id': pd.Series(['A1', 'A2', 'B1', 'B2'], dtype='string'),
        'pathway': pd.Series(['Self paced', 'Self paced', 'Coached', 'Coached'], dtype='string'),
        'activities_completed': pd.Series([4, 2, 5, 1], dtype='int64'),
        'reflection_score': pd.Series([71, 60, 75, 54], dtype='int64'),
    })
    snapshot = alternate.copy(deep=True)
    figure, axes = build_exploratory_chart(alternate, ['Coached', 'Self paced'])
    assert alternate.equals(snapshot), 'exploration mutated input'
    assert figure.axes == [axes] and len(axes.collections[0].get_offsets()) == 4
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ['Coached', 'Self paced']
    assert len(_marker_geometries(axes)) == 2
    plt.close(figure)
    for bad_order in (['Coached'], ['Coached', 'Coached']):
        try:
            build_exploratory_chart(alternate, bad_order)
        except ValueError:
            pass
        else:
            raise AssertionError('exploration accepted invalid pathway order')

def _bar_heights(axes):
    return [round(float(patch.get_height()), 6) for patch in axes.patches]

def _task2():
    assert flawed_axes.get_title() == 'Live delivery caused stronger completion'
    assert tuple(round(value) for value in flawed_axes.get_ylim()) == (76, 83)
    assert flawed_axes.get_ylabel() == '' and len(flawed_axes.patches) == 4
    assert flawed_figure.get_facecolor() == matplotlib.colors.to_rgba('#FFF4CC')
    assert all(line.get_visible() and line.get_linewidth() == 2.0 for line in [*flawed_axes.get_xgridlines(), *flawed_axes.get_ygridlines()])
    assert redesign_figure.axes == [redesign_axes]
    assert _bar_heights(redesign_axes) == [81.0, 77.0, 82.0, 80.0]
    assert redesign_axes.get_ylim()[0] == 0
    assert redesign_axes.get_xlabel() == 'Stage' and redesign_axes.get_ylabel() == 'Prepared completion (%)'
    assert redesign_axes.get_title() == 'Prepared completion by delivery format and stage'
    assert {patch.get_hatch() for patch in redesign_axes.patches} == {'//', '\\\\'}
    assert {text.get_text() for text in redesign_axes.texts} == {'81%', '77%', '82%', '80%'}
    legend = redesign_axes.get_legend()
    assert legend.get_title().get_text() == 'Delivery format' and not legend.get_frame_on() and legend._loc == 2
    assert not redesign_axes.spines['top'].get_visible() and not redesign_axes.spines['right'].get_visible()

    alternate = pd.DataFrame({
        'format': pd.Series(['Remote', 'Remote', 'Studio', 'Studio'], dtype='string'),
        'stage': pd.Series(['After', 'Before', 'After', 'Before'], dtype='string'),
        'completion_percent': pd.Series([61, 55, 72, 64], dtype='int64'),
    })
    snapshot = alternate.copy(deep=True)
    figure, axes = build_critique_redesign(alternate, ['Studio', 'Remote'], ['Before', 'After'])
    assert alternate.equals(snapshot), 'redesign mutated input'
    assert _bar_heights(axes) == [64.0, 72.0, 55.0, 61.0]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ['Studio', 'Remote']
    assert {text.get_text() for text in axes.texts} == {'64%', '72%', '55%', '61%'}
    plt.close(figure)
    incomplete = alternate.iloc[:-1].copy()
    try:
        build_critique_redesign(incomplete, ['Studio', 'Remote'], ['Before', 'After'])
    except ValueError:
        pass
    else:
        raise AssertionError('redesign accepted incomplete 2-by-2 input')

def _check_path(axes, labels, expected_y, title, text, xy):
    assert [line.get_label() for line in axes.lines] == labels
    assert [line.get_ydata().tolist() for line in axes.lines] == expected_y
    assert axes.get_title() == title
    annotation = axes.texts[-1]
    assert annotation.get_text() == text and tuple(annotation.xy) == xy

def _task3():
    assert explanatory_figure.axes == [explanatory_axes]
    _check_path(
        explanatory_axes,
        ['Independent', 'Facilitated'],
        [[58, 63, 67, 70], [57, 65, 72, 79]],
        'Facilitated finishes higher in the prepared 4-checkpoint summary',
        'Checkpoint 4 observed gap: 9 percentage points',
        (4, 79),
    )
    assert final_gap_annotation in explanatory_axes.texts
    assert explanatory_axes.get_xlabel() == 'Checkpoint' and explanatory_axes.get_ylabel() == 'Prepared completion (%)'
    assert [line.get_marker() for line in explanatory_axes.lines] == ['o', 's']
    assert [line.get_linestyle() for line in explanatory_axes.lines] == ['-', '--']
    assert explanatory_axes.get_legend().get_title().get_text() == 'Pathway'

    reverse = pd.DataFrame({
        'pathway': pd.Series(['Route A'] * 3 + ['Route B'] * 3, dtype='string'),
        'checkpoint_number': pd.Series([2, 4, 7, 2, 4, 7], dtype='int64'),
        'completion_percent': pd.Series([40, 57, 88, 51, 64, 73], dtype='int64'),
    })
    snapshot = reverse.copy(deep=True)
    figure, axes, annotation = build_explanatory_chart(reverse, ['Route B', 'Route A'])
    assert reverse.equals(snapshot), 'explanatory function mutated input'
    _check_path(axes, ['Route B', 'Route A'], [[51, 64, 73], [40, 57, 88]], 'Route A finishes higher in the prepared 3-checkpoint summary', 'Checkpoint 7 observed gap: 15 percentage points', (7, 88))
    assert annotation is axes.texts[-1]
    plt.close(figure)

    tie = reverse.copy(deep=True)
    tie.loc[tie['pathway'].eq('Route A') & tie['checkpoint_number'].eq(7), 'completion_percent'] = 73
    figure, axes, annotation = build_explanatory_chart(tie, ['Route A', 'Route B'])
    _check_path(axes, ['Route A', 'Route B'], [[40, 57, 73], [51, 64, 73]], 'Both pathways finish equally in the prepared 3-checkpoint summary', 'Checkpoint 7 observed gap: 0 percentage points', (7, 73))
    assert annotation is axes.texts[-1]
    assert matplotlib.colors.to_hex(annotation.get_color()).upper() == ORANGE
    assert matplotlib.colors.to_hex(annotation.arrow_patch.get_edgecolor()).upper() == ORANGE
    plt.close(figure)
    for bad in (reverse.iloc[:-1].copy(), pd.concat([reverse, reverse.iloc[[0]]], ignore_index=True)):
        try:
            build_explanatory_chart(bad, ['Route A', 'Route B'])
        except ValueError:
            pass
        else:
            raise AssertionError('explanatory function accepted invalid grain/checkpoints')

_central_record('task1', _task1)
_central_record('task2', _task2)
_central_record('task3', _task3)
(ASSIGNMENT_ROOT / '__central_checks.json').write_text(json.dumps(central_results), encoding='utf-8')
'''


def _execute_notebook(root: Path, cwd: Path, extra_source: str | None = None) -> nbformat.NotebookNode:
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    if extra_source is not None:
        notebook.cells.append(nbformat.v4.new_code_cell(extra_source, id="a07-central-checks"))
    old_backend = os.environ.get("MPLBACKEND")
    old_bytes = os.environ.get("PYTHONDONTWRITEBYTECODE")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(cwd)}},
            allow_errors=False,
        )
        return client.execute()
    except CellExecutionError as error:
        lines = str(error).strip().splitlines()
        raise AssertionError("fresh notebook execution failed: " + " | ".join(lines[-8:])) from error
    except Exception as error:
        raise InfrastructureError(f"kernel infrastructure failed: {type(error).__name__}: {error}") from error
    finally:
        if old_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = old_backend
        if old_bytes is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = old_bytes


def _central_results(root: Path) -> dict:
    path = root / "__central_checks.json"
    _assert(path.is_file(), "grader alternate checks did not produce a result")
    result = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    _assert(set(result) == {"task1", "task2", "task3"}, "alternate result malformed")
    return result


def _embedded_teaching_images(notebook: nbformat.NotebookNode) -> None:
    by_id = {cell.id: cell for cell in notebook.cells}
    for cell_id in ("a07-explore-run", "a07-redesign-run", "a07-explanatory-run"):
        images = [
            output.get("data", {}).get("image/png")
            for output in by_id[cell_id].get("outputs", [])
            if output.get("output_type") == "display_data"
        ]
        _assert(any(images), f"fresh teaching figure was not visibly displayed in {cell_id}")


def _record(name: str, maximum: int, error: Exception | None) -> dict:
    passed = error is None
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {'checks passed' if passed else error}")
    return {
        "test-name": name,
        "passed": passed,
        "score": maximum if passed else 0,
        "max-score": maximum,
    }


def grade_submission(submission_root: str | Path) -> dict:
    context = _context()
    _check_grader_runtime()
    root = Path(submission_root).resolve()
    errors: dict[str, Exception | None] = {name: None for name, _ in TEST_SPECS}
    try:
        _check_static(root)
    except Exception as error:
        for name in errors:
            errors[name] = error
    else:
        committed = _artifact_bytes(root)
        with tempfile.TemporaryDirectory(prefix="a07-central-") as temporary_name:
            temporary = Path(temporary_name)
            flat = temporary / "relocated flattened submission with spaces"
            _copy_submission(root, flat)
            sentinel = flat / "grader-owned-sentinel.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            for name in OUTPUT_NAMES:
                path = flat / "output" / name
                if path.exists():
                    path.unlink()
                path.write_bytes(b"stale grader artifact\n")
            cwd = flat / "arbitrary" / "deep nested cwd"
            cwd.mkdir(parents=True)
            try:
                executed = _execute_notebook(flat, cwd, GRADER_CHECKS_SOURCE)
                alternate = _central_results(flat)
                _assert(sentinel.read_text() == "preserve me\n", "setup removed unrelated grader sentinel")
                _embedded_teaching_images(executed)
                _output_inventory(flat)
                fresh = _artifact_bytes(flat)
                _assert(fresh == committed, "committed artifacts differ from fresh execution")
            except InfrastructureError:
                raise
            except Exception as error:
                for name in errors:
                    errors[name] = error
            else:
                try:
                    _assert(alternate["task1"]["passed"], alternate["task1"]["detail"])
                except Exception as error:
                    errors["Task 1 bounded exploration"] = error
                try:
                    _assert(alternate["task2"]["passed"], alternate["task2"]["detail"])
                except Exception as error:
                    errors["Task 2 critique and redesign"] = error
                try:
                    _assert(alternate["task3"]["passed"], alternate["task3"]["detail"])
                except Exception as error:
                    errors["Task 3 explanatory evidence"] = error
                try:
                    _check_evidence(flat)
                    first_sizes = {name: _check_image(flat / "output" / name) for name in ("critique_redesign.png", "pathway_explanatory.png")}
                except Exception as error:
                    errors["Artifact integrity"] = error
                try:
                    for name in OUTPUT_NAMES:
                        (flat / "output" / name).write_bytes(b"second stale artifact\n")
                    _execute_notebook(flat, cwd)
                    _assert(sentinel.read_text() == "preserve me\n", "repeat removed unrelated sentinel")
                    _assert(_artifact_bytes(flat) == fresh, "second fresh kernel is not deterministic")
                except InfrastructureError:
                    raise
                except Exception as error:
                    errors["Fixtures and reproducibility"] = error

                course_root = temporary / "relocated course root"
                nested = course_root / "07" / "assignment"
                nested.parent.mkdir(parents=True)
                _copy_submission(root, nested)
                for name in OUTPUT_NAMES:
                    (nested / "output" / name).unlink()
                try:
                    _execute_notebook(nested, course_root)
                    _assert(_artifact_bytes(nested) == fresh, "course-root layout differs from flattened layout")
                except InfrastructureError:
                    raise
                except Exception as error:
                    errors["Fixtures and reproducibility"] = error

    tests = [_record(name, maximum, errors[name]) for name, maximum in TEST_SPECS]
    return {
        "schema": "datasci217/grading-result/v1",
        **context,
        "score": sum(test["score"] for test in tests),
        "max-score": 80,
        "tests": tests,
    }


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    try:
        result = grade_submission(target)
        Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    except InfrastructureError as error:
        print(f"[INFRASTRUCTURE] {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"[INFRASTRUCTURE] unexpected grader failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
