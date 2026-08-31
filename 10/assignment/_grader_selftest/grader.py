# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==6.29.5",
#   "matplotlib==3.11.1",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.5",
#   "Pillow==12.3.0",
#   "scikit-learn==1.9.0",
#   "statsmodels==0.14.6",
# ]
# ///

"""Instructor-controlled direct-environment grader for Assignment 10."""

from __future__ import annotations

import ast
import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import matplotlib
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import numpy as np
import pandas as pd
from PIL import Image
import sklearn
import statsmodels


PEP_REQUIRES_PYTHON = "==3.12.13"
PEP_DEPENDENCIES = [
    "matplotlib==3.11.1",
    "numpy==2.0.2",
    "pandas==3.0.5",
    "scikit-learn==1.9.0",
    "statsmodels==0.14.6",
]
PEP_BLOCK = [
    "# /// script",
    f'# requires-python = "{PEP_REQUIRES_PYTHON}"',
    "# dependencies = [",
    *[f'#   "{dependency}",' for dependency in PEP_DEPENDENCIES],
    "# ]",
    "# ///",
]


CELL_IDS = [
    "a10-header", "a10-setup", "a10-terms-inference", "a10-load",
    "a10-task1-prompt", "a10-ols-function", "a10-task1-run",
    "a10-residual-figure", "a10-task1-save", "a10-task1-explain",
    "a10-terms-prediction", "a10-task2-contract", "a10-task2-values",
    "a10-availability-function", "a10-split-function", "a10-task2-run",
    "a10-task2-save", "a10-task2-explain", "a10-terms-evaluation",
    "a10-regression-metrics-function", "a10-candidates-function",
    "a10-validation-run", "a10-validation-save", "a10-freeze",
    "a10-final-test-run", "a10-final-test-save", "a10-binary-function",
    "a10-binary-run-save", "a10-task3-explain", "a10-final-verify",
]
PROTECTED_CELL_IDS = {
    "a10-header", "a10-setup", "a10-terms-inference", "a10-task1-prompt",
    "a10-terms-prediction", "a10-terms-evaluation", "a10-freeze",
    "a10-final-verify",
}
PROTECTED_FILE_SHA256 = {
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "PLATFORM_CHECK.md": "26fa8f87d119d95cc556197c9c9304ffa4d9a19c74fbf6d6bee6d00a73c93cf1",
    "README.md": "8f2a523f5fc000601c3421950be36974735b045ac819148353a21256856a893c",
    "check_assignment.py": "f354cdffae6c538aba0bd06613d9d5704186d952afb924d38d9e3ec6b8b74f66",
    "requirements.txt": "4c6d9eaa5d730c7dfb71124d1576070dfabefe9162124c74162d4bb172c77984",
    "data/fixture.json": "aa50eeffc2b07c5d98cb56a0e3d18115909958f777899d5d403cf6323dd1de41",
    "data/mixing_runs.csv": "00b8a1ce84110f4a7fa85620742283c82a4b9d600dbe0ebea0d4721956938957",
    "data/batch_strength.csv": "f14faf7da64347dfc255aa84b14e79eef7f2d0de94b394c747323319d937baa3",
    "data/feature_availability.csv": "a47b8df048607045640b9a6785b038fe1c70036f58d5b61ed20ec98860b556da",
    "data/supplied_binary_predictions.csv": "7a8809010fa94345cd04787c826ef86ee5fd13cbf0bd95953e2220c3294a239a",
    "output/.gitkeep": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
PROTECTED_CELL_SHA256 = {
    "a10-header": "d0a854bb0eebce383b25c11f2d17151432d27928110bd461b2e038de3ef5343a",
    "a10-setup": "a5caad874818b2edf045770960e00c610ea6d29a2e50d3dd89c3d6328caf20e8",
    "a10-terms-inference": "4c167e07f0ae870026a9746b40138fe5c3d6a4e74dddd417580e0e39e096696e",
    "a10-task1-prompt": "855729ff10c4244d077be4152576d05e17c99d104bff7e896495bdde402c193b",
    "a10-terms-prediction": "86136b5c6caedbd549a7148563f05280a83a589a32dcfae7ce7e3ba8cafc0a95",
    "a10-terms-evaluation": "84adc778e76702fb952469be3601793888ac302a66fb0360898cd8f52b5752b7",
    "a10-freeze": "2f1c03daef8bdec283d35e485918e3a4ab55bf55eb1cbd10cb509f74301ddfc1",
    "a10-final-verify": "e401fcb4ea4be0881f2b381f662d7c4a45d2b9b494d986fe29baa551ef4d88a8",
}
MARKDOWN_IDS = {
    "a10-header", "a10-terms-inference", "a10-task1-prompt",
    "a10-task1-explain", "a10-terms-prediction", "a10-task2-contract",
    "a10-task2-explain", "a10-terms-evaluation", "a10-task3-explain",
}
STUDENT_CODE_IDS = {
    "a10-load", "a10-ols-function", "a10-task1-run", "a10-residual-figure",
    "a10-task1-save", "a10-task2-values", "a10-availability-function",
    "a10-split-function", "a10-task2-run", "a10-task2-save",
    "a10-regression-metrics-function", "a10-candidates-function",
    "a10-validation-run", "a10-validation-save", "a10-final-test-run",
    "a10-final-test-save", "a10-binary-function", "a10-binary-run-save",
}
SIGNATURES = {
    "fit_bounded_ols": ["inference_table", "predictor_columns", "outcome_column"],
    "audit_feature_availability": ["candidate_table"],
    "build_chronological_splits": ["prediction_table", "validation_start", "test_start"],
    "regression_metrics": ["actual", "predicted"],
    "fit_prediction_candidates": ["train_table", "feature_columns", "target_column"],
    "choose_validation_winner": ["metrics_table", "metric_column"],
    "compute_binary_metrics": ["prediction_table", "actual_column", "prediction_columns"],
}
FIXTURES = {
    "data/fixture.json": "aa50eeffc2b07c5d98cb56a0e3d18115909958f777899d5d403cf6323dd1de41",
    "data/mixing_runs.csv": "00b8a1ce84110f4a7fa85620742283c82a4b9d600dbe0ebea0d4721956938957",
    "data/batch_strength.csv": "f14faf7da64347dfc255aa84b14e79eef7f2d0de94b394c747323319d937baa3",
    "data/feature_availability.csv": "a47b8df048607045640b9a6785b038fe1c70036f58d5b61ed20ec98860b556da",
    "data/supplied_binary_predictions.csv": "7a8809010fa94345cd04787c826ef86ee5fd13cbf0bd95953e2220c3294a239a",
}
BASE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt", *FIXTURES,
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}
CSV_HASHES = {
    "inference_summary.csv": "36965b53df5133e3e05f86502d230ec9241b58e9ffd93163eba588385c9f3f48",
    "inference_case_intervals.csv": "345e0d3aefc422606fa9a9ee1b35a06bd7a9f9007873fc7b05162cb9ef3e0951",
    "availability_decisions.csv": "36042dc19dd45f75603f2fb2d5783b0a7750dad274a54bd39e8d21d5f5c2ac81",
    "split_manifest.csv": "2b0f3f57e323fa7bfe7a0703c671755ed7b009854236e62dd0c3459b1aa67b21",
    "validation_metrics.csv": "65b105be797b109c2031ccde552972320c1d08cb59174cde628a23c1879832dc",
    "final_test_metrics.csv": "ca1bd6d4320ed84cd2ca5befe97c3c0f238746452b648e64103522517b9a77ce",
    "final_predictions.csv": "60b7457821655c387b07694e18cad262a873c50bc69093a9638bd8ea99239a1d",
    "binary_metrics.csv": "25d7b50cdb8160f8e275812010a9a90b295d700b03591b3ce7bfd712483616fa",
}
POINTS = [10, 20, 25, 30, 5]
TEST_NAMES = [
    "Template, environment, fixture, notebook, protected integrity",
    "Task 1 bounded inference and intervals",
    "Task 2 contract, availability, leakage, chronological split",
    "Task 3 train-only comparison, freeze, final test, binary metrics",
    "Portability, visible output, repeatability, resubmission",
]


class InfrastructureError(RuntimeError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _normalized_source(cell: dict) -> str:
    return _source(cell).replace("\r\n", "\n").replace("\r", "\n")


def _validate_integrity_profile() -> None:
    expected_file_keys = {
        ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
        "check_assignment.py", "requirements.txt", "data/fixture.json",
        "data/mixing_runs.csv", "data/batch_strength.csv",
        "data/feature_availability.csv", "data/supplied_binary_predictions.csv",
        "output/.gitkeep",
    }
    _assert(set(PROTECTED_FILE_SHA256) == expected_file_keys, "protected-file map keys differ")
    _assert(set(PROTECTED_CELL_SHA256) == PROTECTED_CELL_IDS, "protected-cell map keys differ")
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    _assert(all(digest_pattern.fullmatch(value) for value in PROTECTED_FILE_SHA256.values()), "protected-file map has a non-SHA256 digest")
    _assert(all(digest_pattern.fullmatch(value) for value in PROTECTED_CELL_SHA256.values()), "protected-cell map has a non-SHA256 digest")


def _validate_checker_static(root: Path) -> None:
    checker_path = root / "check_assignment.py"
    checker_bytes = checker_path.read_bytes()
    checker_text = checker_bytes.decode("utf-8")
    lines = checker_text.splitlines()
    start = 1 if lines and lines[0] == "#!/usr/bin/env python3" else 0
    _assert(lines[start:start + len(PEP_BLOCK)] == PEP_BLOCK, "checker PEP 723 block differs")
    requirement_lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    _assert(requirement_lines == PEP_DEPENDENCIES, "checker PEP 723 dependencies differ from requirements.txt")
    checker_tree = ast.parse(checker_text)
    imported_roots = set()
    for node in ast.walk(checker_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    forbidden = {"matplotlib", "numpy", "pandas", "sklearn", "statsmodels"}
    _assert(not imported_roots.intersection(forbidden), "checker implementation imports a candidate library")


def _validate_candidate_integrity(root: Path, notebook) -> None:
    _validate_integrity_profile()
    for relative, expected in PROTECTED_FILE_SHA256.items():
        path = root / relative
        _assert(path.is_file() and not path.is_symlink(), f"immutable learner file missing: {relative}")
        _assert(sha256(path.read_bytes()).hexdigest() == expected, f"protected learner file changed: {relative}")
    by_id = {cell.id: cell for cell in notebook.cells}
    for cell_id, expected in PROTECTED_CELL_SHA256.items():
        _assert(cell_id in by_id, f"protected cell missing: {cell_id}")
        actual = sha256(_normalized_source(by_id[cell_id]).encode("utf-8")).hexdigest()
        _assert(actual == expected, f"protected cell changed: {cell_id}")


def _resolve_root(start: Path) -> Path:
    start = start.resolve()
    for base in (start, *start.parents):
        for candidate in (base, base / "10" / "assignment"):
            if (candidate / "assignment.ipynb").is_file() and (candidate / "data/fixture.json").is_file():
                return candidate.resolve()
    raise InfrastructureError("cannot locate the complete Assignment 10 learner package")


def _inventory(root: Path) -> None:
    git_entry = root / ".git"
    if git_entry.exists() or git_entry.is_symlink():
        _assert(git_entry.is_dir() and not git_entry.is_symlink(), "top-level .git must be a genuine directory")
    output_entry = root / "output"
    _assert(output_entry.is_dir() and not output_entry.is_symlink(), "output must be a genuine directory")
    _assert(not (root / "_grader_selftest").exists() and not (root / "_grader_selftest").is_symlink(), "instructor bundle entered learner package")
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in {".git", "output"}:
            continue
        if path.is_file() or path.is_symlink():
            actual.add(relative.as_posix())
    expected = BASE_FILES
    _assert(actual == expected, f"learner package inventory differs: {sorted(actual ^ expected)}")
    _assert(not any((root / relative).is_symlink() for relative in actual), "learner package contains a symlink")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] == ".git":
            continue
        _assert(not path.is_symlink(), f"symlink rejected: {relative.as_posix()}")


def _validate_template(root: Path) -> tuple[dict, dict[str, dict], ast.Module]:
    _assert(sys.version_info[:3] == (3, 12, 13), "grader Python differs from 3.12.13")
    versions = (matplotlib.__version__, np.__version__, pd.__version__, sklearn.__version__, statsmodels.__version__)
    _assert(versions == ("3.11.1", "2.0.2", "3.0.5", "1.9.0", "0.14.6"), f"direct candidate versions differ: {versions}")
    _inventory(root)
    _validate_checker_static(root)
    for relative, digest in FIXTURES.items():
        path = root / relative
        _assert(path.is_file() and not path.is_symlink(), f"fixture missing: {relative}")
        raw = path.read_bytes()
        _assert(sha256(raw).hexdigest() == digest and raw.endswith(b"\n") and b"\r" not in raw, f"fixture changed: {relative}")
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    _validate_candidate_integrity(root, notebook)
    cells = notebook.cells
    ids = [cell.id for cell in cells]
    _assert(ids == CELL_IDS and len(ids) == len(set(ids)), "notebook topology differs")
    _assert(notebook.nbformat == 4 and notebook.nbformat_minor == 5, "notebook is not format 4.5")
    _assert(notebook.metadata.kernelspec == {"display_name": "Python 3", "language": "python", "name": "python3"}, "kernelspec differs")
    by_id = {cell.id: cell for cell in cells}
    for cell in cells:
        _assert(cell.cell_type == ("markdown" if cell.id in MARKDOWN_IDS else "code"), f"cell type differs: {cell.id}")
    student_code = "\n".join(_source(by_id[cell_id]) for cell_id in CELL_IDS if cell_id in STUDENT_CODE_IDS)
    student_markdown = "\n".join(_source(by_id[cell_id]) for cell_id in ("a10-task1-explain", "a10-task2-contract", "a10-task2-explain", "a10-task3-explain"))
    _assert("TODO" not in student_code + student_markdown and "NotImplementedError" not in student_code, "submission contains starter scaffolds")
    tree = ast.parse(student_code)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for name, arguments in SIGNATURES.items():
        _assert(name in functions and [arg.arg for arg in functions[name].args.args] == arguments, f"wrong signature: {name}")
        fn_source = ast.unparse(functions[name])
        _assert(not any(fragment in fn_source for fragment in ("read_csv", "to_csv", "savefig", "Path(")), f"function performs I/O: {name}")
    _assert(not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)), "student cells import libraries")
    lowered = student_code.lower().replace(" ", "")
    for fragment in ("requests", "urlopen", "http://", "https://", "/content", "drive.mount", "files.upload", "random.", "xgboost", "randomforest", "cross_val", "gridsearch", "ridge(", "lasso(", ".score(", "pvalues", ".aic", "add_constant"):
        _assert(fragment not in lowered, f"out-of-scope source: {fragment}")
    _assert("smf.ols" in student_code and '" ~ "' in student_code, "argument-derived formula OLS missing")
    _assert("DummyRegressor" in student_code and "Pipeline" in student_code and "StandardScaler" in student_code and "LinearRegression" in student_code, "candidate mechanisms differ")
    _assert("zero_division=0" in student_code and "kind=\"stable\"" in student_code, "required metric/sort mechanisms differ")
    _assert(not any(".predict(" in _source(by_id[cell_id]) for cell_id in STUDENT_CODE_IDS), "direct student predict bypasses recorder")
    return notebook, by_id, tree


ALTERNATE_CELL = r'''# Instructor-disclosed alternate checks
import json as _grader_json

assert [(row["kind"], row["count"], row["index"]) for row in _GRADER_FIT_CALLS] == [
    ("mean_baseline", 29, list(range(29))),
    ("linear_pipeline", 29, list(range(29))),
]
assert [(row["kind"], row["count"], row["index"]) for row in _GRADER_PREDICT_CALLS] == [
    ("mean_baseline", 8, [f"B{row:03d}" for row in range(30, 38)]),
    ("linear_pipeline", 8, [f"B{row:03d}" for row in range(30, 38)]),
    ("linear_pipeline", 11, [f"B{row:03d}" for row in range(38, 49)]),
]
assert _GRADER_FIT_CALLS[0]["estimator_id"] == _GRADER_PREDICT_CALLS[0]["estimator_id"]
assert _GRADER_FIT_CALLS[1]["estimator_id"] == _GRADER_PREDICT_CALLS[1]["estimator_id"] == _GRADER_PREDICT_CALLS[2]["estimator_id"]
np.testing.assert_allclose(prediction_candidates["mean_baseline"].constant_, [[train_table[prediction_contract["target"]].mean()]], rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(prediction_candidates["linear_pipeline"].named_steps["scale"].mean_, train_table[feature_columns].mean().to_numpy(), rtol=1e-12, atol=1e-12)
assert FROZEN_SELECTED_APPROACH == "linear_pipeline" and TEST_GATE_OPEN is False
assert len(fig.axes) == 1 and fig.axes[0].get_title() == "Residuals versus fitted values"
assert fig.axes[0].get_xlabel() == "Fitted finish quality score" and fig.axes[0].get_ylabel() == "Residual"
assert tuple(np.round(fig.get_size_inches(), 8)) == (7.2, 4.8) and len(fig.axes[0].collections[0].get_offsets()) == 18
_GRADER_CANONICAL_FIT_CALLS = [dict(row) for row in _GRADER_FIT_CALLS]
_GRADER_CANONICAL_PREDICT_CALLS = [dict(row) for row in _GRADER_PREDICT_CALLS]

_alt_ols = pd.DataFrame({
    "dose": [0.,1.,2.,3.,4.,5.], "temperature": [2.,0.,1.,3.,2.,4.],
    "quality": [9.,12.,13.5,14.5,17.,18.],
}, index=[41,7,90,3,55,12])
_alt_ols_snapshot = _alt_ols.copy(deep=True)
_alt_model = fit_bounded_ols(_alt_ols, ["dose", "temperature"], "quality")
assert _alt_model.model.formula == "quality ~ dose + temperature"
np.testing.assert_allclose(_alt_model.params.to_numpy(), [10.,2.,-.5], rtol=1e-12, atol=1e-12)
pd.testing.assert_frame_equal(_alt_ols, _alt_ols_snapshot)
_renamed = _alt_ols.rename(columns={"dose":"x_one", "temperature":"x_two", "quality":"response_y"})
assert fit_bounded_ols(_renamed, ["x_one", "x_two"], "response_y").model.formula == "response_y ~ x_one + x_two"

_alt_avail = pd.DataFrame({"candidate_feature": pd.Series(["late_8","at_issue","past_2","late_1","past_24"], dtype="string"), "latest_required_offset_hours": np.array([8,0,-2,1,-24], dtype="int64")}, index=[8,3,21,1,13])
_alt_avail["candidate_feature"] = pd.Series(["late_8","at_issue","past_2","late_1","past_24"], index=_alt_avail.index, dtype="string")
_alt_avail_snapshot = _alt_avail.copy(deep=True)
_alt_avail_out = audit_feature_availability(_alt_avail)
assert _alt_avail_out.index.tolist() == [8,3,21,1,13]
assert _alt_avail_out["decision"].tolist() == ["exclude","keep","keep","exclude","keep"]
pd.testing.assert_frame_equal(_alt_avail, _alt_avail_snapshot)

_rows = [
(44,"X07","2027-02-06T18:00:00Z","2027-02-07T00:00:00Z"),(7,"X02","2027-02-01T18:00:00Z","2027-02-02T00:00:00Z"),(105,"X19","2027-02-18T18:00:00Z","2027-02-19T00:00:00Z"),(3,"X12","2027-02-11T18:00:00Z","2027-02-12T00:00:00Z"),(91,"X01","2027-01-31T18:00:00Z","2027-02-01T00:00:00Z"),(18,"X15","2027-02-14T18:00:00Z","2027-02-15T00:00:00Z"),(62,"X09","2027-02-08T18:00:00Z","2027-02-09T00:00:00Z"),(5,"X04","2027-02-03T18:00:00Z","2027-02-04T00:00:00Z"),(77,"X14","2027-02-13T18:00:00Z","2027-02-14T00:00:00Z"),(22,"X06","2027-02-05T18:00:00Z","2027-02-06T00:00:00Z"),(130,"X18","2027-02-17T18:00:00Z","2027-02-18T00:00:00Z"),(11,"X03","2027-02-02T18:00:00Z","2027-02-03T00:00:00Z"),(58,"X10","2027-02-09T18:00:00Z","2027-02-10T00:00:00Z"),(9,"X16","2027-02-15T18:00:00Z","2027-02-16T00:00:00Z"),(73,"X05","2027-02-04T18:00:00Z","2027-02-05T00:00:00Z"),(31,"X13","2027-02-12T18:00:00Z","2027-02-13T00:00:00Z"),(99,"X08","2027-02-07T18:00:00Z","2027-02-08T00:00:00Z"),(14,"X17","2027-02-16T18:00:00Z","2027-02-17T00:00:00Z"),(66,"X11","2027-02-10T18:00:00Z","2027-02-11T00:00:00Z")]
_alt_split = pd.DataFrame(_rows, columns=["_index","batch_id","prediction_timestamp","target_timestamp"]).set_index("_index")
_alt_split["batch_id"] = _alt_split["batch_id"].astype("string")
for _column in ["prediction_timestamp", "target_timestamp"]: _alt_split[_column] = pd.to_datetime(_alt_split[_column], utc=True)
_split_snapshot = _alt_split.copy(deep=True)
_parts, _manifest = build_chronological_splits(_alt_split, pd.Timestamp("2027-02-13T00:00:00Z"), pd.Timestamp("2027-02-16T00:00:00Z"))
assert _parts["train"]["batch_id"].tolist() == [f"X{row:02d}" for row in range(1,13)]
assert _parts["validation"]["batch_id"].tolist() == ["X13","X14","X15"]
assert _parts["test"]["batch_id"].tolist() == ["X16","X17","X18","X19"]
assert _manifest["row_count"].tolist() == [12,3,4]
pd.testing.assert_frame_equal(_alt_split, _split_snapshot)
for _kind in ["equal", "reversed", "naive", "missing", "duplicate", "late"]:
    _bad = _alt_split.copy(deep=True); _v = pd.Timestamp("2027-02-13T00:00:00Z"); _t = pd.Timestamp("2027-02-16T00:00:00Z")
    if _kind == "equal": _v = _t
    elif _kind == "reversed": _v, _t = _t, _v
    elif _kind == "naive": _v = pd.Timestamp("2027-02-13")
    elif _kind == "missing": _bad.iloc[0, _bad.columns.get_loc("target_timestamp")] = pd.NaT
    elif _kind == "duplicate": _bad.iloc[0, _bad.columns.get_loc("batch_id")] = _bad.iloc[1]["batch_id"]
    elif _kind == "late": _bad.iloc[0, _bad.columns.get_loc("prediction_timestamp")] = _bad.iloc[0]["target_timestamp"]
    try: build_chronological_splits(_bad, _v, _t); raise AssertionError(_kind)
    except ValueError: pass

_metrics = regression_metrics([1.,2.,4.], [1.,3.,2.])
np.testing.assert_allclose(list(_metrics.values()), [1.,1.2909944487358056,-1/14], rtol=1e-12, atol=1e-12)
for _a, _p in [([],[]), ([1,2],[1]), ([[1,2],[3,4]], [[1,2],[3,4]]), ([1,np.nan],[1,2])]:
    try: regression_metrics(_a, _p); raise AssertionError("metrics invalid")
    except ValueError: pass

_alt_train = pd.DataFrame({"u":[2.,-1.,4.,0.,3.,1.,5.],"v":[10.,14.,8.,12.,9.,11.,7.],"z":[5.2,-1.1,8.7,.4,7.1,2.6,10.5]}, index=[81,5,44,12,99,3,70])
_train_snapshot = _alt_train.copy(deep=True)
_candidates = fit_prediction_candidates(_alt_train, ["u","v"], "z")
assert list(_candidates) == ["mean_baseline","linear_pipeline"]
np.testing.assert_allclose(_candidates["mean_baseline"].constant_, [[4.771428571428571]], rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(_candidates["linear_pipeline"].named_steps["scale"].mean_, [2.,10.142857142857142], rtol=1e-12, atol=1e-12)
pd.testing.assert_frame_equal(_alt_train, _train_snapshot)

_winner = pd.DataFrame({"approach":["zeta","alpha","beta","ignored_nan"],"validation_loss":[.40,.25,.30,np.nan]}, index=[12,4,99,7])
assert choose_validation_winner(_winner, "validation_loss") == "alpha"
assert choose_validation_winner(pd.DataFrame({"approach":["zeta","alpha","beta"],"validation_loss":[.25,.25,.4]}, index=[8,2,30]), "validation_loss") == "alpha"

_alt_binary = pd.DataFrame({"actual":[1,1,0,0],"model_prediction":[1,0,1,0],"dummy_prediction":[0,0,0,0]})
_binary_out = compute_binary_metrics(_alt_binary, "actual", {"model_alt":"model_prediction","dummy_alt":"dummy_prediction"})
assert _binary_out["approach"].tolist() == ["model_alt","dummy_alt"]
np.testing.assert_allclose(_binary_out[["accuracy","precision","recall"]], [[.5,.5,.5],[.5,0.,0.]], rtol=1e-12, atol=1e-12)

(ASSIGNMENT_ROOT / "_grader_evidence.json").write_text(_grader_json.dumps({"alternates": 7, "status": "pass", "canonical_fit_calls": _GRADER_CANONICAL_FIT_CALLS, "canonical_predict_calls": _GRADER_CANONICAL_PREDICT_CALLS}), encoding="utf-8")
'''


INSTRUMENTATION_CELL = r'''# Independent central fit/predict instrumentation
_GRADER_FIT_CALLS = []
_GRADER_PREDICT_CALLS = []
_GRADER_ORIGINAL_DUMMY_FIT = DummyRegressor.fit
_GRADER_ORIGINAL_PIPELINE_FIT = Pipeline.fit
_GRADER_ORIGINAL_DUMMY_PREDICT = DummyRegressor.predict
_GRADER_ORIGINAL_PIPELINE_PREDICT = Pipeline.predict

def _grader_dummy_fit(self, X, y, *args, **kwargs):
    _GRADER_FIT_CALLS.append({"kind": "mean_baseline", "estimator_id": id(self), "count": len(X), "index": list(X.index)})
    return _GRADER_ORIGINAL_DUMMY_FIT(self, X, y, *args, **kwargs)

def _grader_pipeline_fit(self, X, y=None, **kwargs):
    _GRADER_FIT_CALLS.append({"kind": "linear_pipeline", "estimator_id": id(self), "count": len(X), "index": list(X.index)})
    return _GRADER_ORIGINAL_PIPELINE_FIT(self, X, y, **kwargs)

def _grader_dummy_predict(self, X, *args, **kwargs):
    _GRADER_PREDICT_CALLS.append({"kind": "mean_baseline", "estimator_id": id(self), "count": len(X), "index": list(X.index)})
    return _GRADER_ORIGINAL_DUMMY_PREDICT(self, X, *args, **kwargs)

def _grader_pipeline_predict(self, X, **kwargs):
    _GRADER_PREDICT_CALLS.append({"kind": "linear_pipeline", "estimator_id": id(self), "count": len(X), "index": list(X.index)})
    return _GRADER_ORIGINAL_PIPELINE_PREDICT(self, X, **kwargs)

DummyRegressor.fit = _grader_dummy_fit
Pipeline.fit = _grader_pipeline_fit
DummyRegressor.predict = _grader_dummy_predict
Pipeline.predict = _grader_pipeline_predict
'''


def _png(path: Path) -> tuple[int, int, int]:
    _assert(path.is_file() and not path.is_symlink(), "residual PNG missing")
    with Image.open(path) as image:
        image.load()
        _assert(image.size == (720, 480) and image.mode in {"RGB", "RGBA"}, "residual PNG dimensions/mode differ")
        colors = image.convert("RGB").getcolors(maxcolors=1_000_000)
        _assert(colors is not None and len(colors) >= 16, "residual PNG is visually trivial")
        _assert(any(color != (255, 255, 255) for _, color in colors), "residual PNG is blank")
        return image.width, image.height, len(colors)


def _clear_outputs(root: Path) -> None:
    output = root / "output"
    output.mkdir(exist_ok=True)
    for path in output.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)


def _execute(root: Path, *, alternates: bool) -> tuple[dict[str, bytes], bytes, dict]:
    _clear_outputs(root)
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.insert(2, nbformat.v4.new_code_cell(INSTRUMENTATION_CELL, id="a10-grader-instrumentation"))
    if alternates:
        notebook.cells.append(nbformat.v4.new_code_cell(ALTERNATE_CELL, id="a10-grader-alternates"))
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}})
    client.execute()
    if alternates:
        evidence_path = root / "_grader_evidence.json"
        _assert(evidence_path.is_file(), "alternate evidence missing")
        evidence = json.loads(evidence_path.read_text())
        evidence_path.unlink()
    else:
        evidence = {"alternates": 0, "status": "not-run"}
    csvs = {name: (root / "output" / name).read_bytes() for name in CSV_HASHES}
    png = (root / "output/inference_residuals.png").read_bytes()
    _png(root / "output/inference_residuals.png")
    return csvs, png, evidence


def _copy_learner(root: Path, destination: Path) -> Path:
    copied = destination / "arbitrary course" / "nested" / "submission with spaces"
    copied.mkdir(parents=True)
    for relative in sorted(BASE_FILES):
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    shutil.copytree(root / "output", copied / "output")
    return copied


def grade_root(root: Path) -> tuple[list[dict], dict]:
    tests = []
    diagnostics = {"instructor_environment": True}
    committed_csvs = {}
    committed_png = b""
    try:
        notebook, by_id, tree = _validate_template(root)
        for name in CSV_HASHES:
            path = root / "output" / name
            _assert(path.is_file() and not path.is_symlink(), f"committed artifact missing: {name}")
            committed_csvs[name] = path.read_bytes()
        committed_png = (root / "output/inference_residuals.png").read_bytes()
        _png(root / "output/inference_residuals.png")
        tests.append({"name": TEST_NAMES[0], "score": 10, "max-score": 10, "output": "candidate runtime, fixtures, inventory, topology, and completed scaffolds pass"})
    except Exception as error:
        tests.append({"name": TEST_NAMES[0], "score": 0, "max-score": 10, "output": str(error)})
        for name, points in zip(TEST_NAMES[1:], POINTS[1:]):
            tests.append({"name": name, "score": 0, "max-score": points, "output": "blocked by template/inventory failure"})
        return tests, diagnostics

    try:
        with tempfile.TemporaryDirectory() as temporary:
            copy = _copy_learner(root, Path(temporary))
            first_csvs, first_png, evidence = _execute(copy, alternates=True)
            diagnostics["alternate_checks"] = evidence["alternates"]
            _assert(all(sha256(value).hexdigest() == CSV_HASHES[name] for name, value in first_csvs.items()), "fresh CSV value/bytes differ")
            _assert(first_csvs["inference_summary.csv"] == committed_csvs["inference_summary.csv"] and first_csvs["inference_case_intervals.csv"] == committed_csvs["inference_case_intervals.csv"], "Task 1 committed/fresh CSV differs")
            tests.append({"name": TEST_NAMES[1], "score": 20, "max-score": 20, "output": "formula OLS, intervals, alternate, and decoded/live residual evidence pass"})
            _assert(first_csvs["availability_decisions.csv"] == committed_csvs["availability_decisions.csv"] and first_csvs["split_manifest.csv"] == committed_csvs["split_manifest.csv"], "Task 2 committed/fresh CSV differs")
            tests.append({"name": TEST_NAMES[2], "score": 25, "max-score": 25, "output": "contract, availability, chronological split, alternates, and boundary errors pass"})
            for name in ("validation_metrics.csv", "final_test_metrics.csv", "final_predictions.csv", "binary_metrics.csv"):
                _assert(first_csvs[name] == committed_csvs[name], f"Task 3 committed/fresh CSV differs: {name}")
            tests.append({"name": TEST_NAMES[3], "score": 30, "max-score": 30, "output": "train-only candidates, validation freeze, one test call, and binary alternates pass"})
            second_csvs, second_png, _ = _execute(copy, alternates=False)
            _assert(first_csvs == second_csvs and first_png == second_png, "second clean candidate run differs")
            _assert(set(path.name for path in (copy / "output").iterdir()) == {".gitkeep", *CSV_HASHES, "inference_residuals.png"}, "output inventory differs after rerun")
            tests.append({"name": TEST_NAMES[4], "score": 5, "max-score": 5, "output": "arbitrary root, fresh repeatability, committed CSV equality, and PNG semantics pass"})
            diagnostics.update({"fresh_runs": 2, "csv_artifacts": 8, "png_fresh_bytes_equal": True, "committed_png_bytes_equal_not_required": committed_png == first_png})
    except Exception as error:
        while len(tests) < 5:
            index = len(tests)
            tests.append({"name": TEST_NAMES[index], "score": 0, "max-score": POINTS[index], "output": str(error)})
    return tests, diagnostics


def _context() -> dict[str, str]:
    mapping = {
        "assignment": "ASSIGNMENT",
        "submission": "SUBMISSION_TAG",
        "commit": "COMMIT_URL",
        "release": "RELEASE_URL",
    }
    result = {}
    for field, variable in mapping.items():
        value = os.environ.get(variable, "").strip()
        if not value:
            raise InfrastructureError(f"missing required grader context: {variable}")
        result[field] = value
    result["review"] = os.environ.get("REVIEW_URL", "").strip() or result["commit"]
    result["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def main() -> int:
    result_path = Path.cwd() / "result.json"
    try:
        if result_path.is_file() or result_path.is_symlink():
            result_path.unlink()
        elif result_path.exists():
            raise InfrastructureError("result.json destination is not a writable regular-file path")
        context = _context()
        root = _resolve_root(Path.cwd())
        tests, diagnostics = grade_root(root)
        result = {
            "schema": "datasci217/grading-result/v1",
            **context,
            "score": sum(test["score"] for test in tests),
            "max-score": 90,
            "tests": tests,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"score": result["score"], "max-score": 90, **diagnostics}, sort_keys=True))
        return 0
    except InfrastructureError as error:
        if result_path.is_file() or result_path.is_symlink():
            result_path.unlink()
        print(f"[INFRASTRUCTURE] {error}", file=sys.stderr)
        return 2
    except Exception as error:
        if result_path.is_file() or result_path.is_symlink():
            result_path.unlink()
        print(f"[INFRASTRUCTURE] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
