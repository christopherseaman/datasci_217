# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==6.29.5",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.5",
# ]
# ///

"""Teacher-controlled Classroom50 central-grader reference for Assignment 09.

The grader never imports the student-editable public checker. It validates
protected sources, clears state and owned output in disposable copies,
fresh-executes the notebook, tests all five functions on disclosed alternate
data, and emits only the official automated 90-point result topology.
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
    "a09-header": "131ea3ecea5c880816109cc7c1b03980dcd3c0c2e4cd9d17b34d68a5af3e9163",
    "a09-setup": "46d251a68a1b2b740e64bdb90e892500cade7bb6235b69e0c2876670cd4f837d",
    "a09-terms-data": "f8ad0129811ecd03ed7c0dc60b860b68a1be545003e4961cdaea6b43489dfa59",
    "a09-task2-prompt": "d1d9d124b7fd9904ee9a47256745def60937945d1bfe40194239314199bedc1b",
    "a09-task3-prompt": "90f1797b0b65472c3567451467e85e144a2add9110a3d11a1714d3d31fe75966",
    "a09-final-verify": "015c87bbc3742a61961efbf806ed513f67363191766ad8bcb7d1bf7200fb996a",
}
PROTECTED_FILE_SHA256 = {
    ".python-version": "aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b",
    "requirements.txt": "90933f178a0a459399ff6696e8fe9407463cc65bbffd567f3e7b44cc9230ee21",
    ".gitignore": "835739aa7952d6845749187c103a4942aa441d5e8bcbfcb3006de7b1d0924c95",
    "README.md": "03026ba8f1d57e57b4a030c2ec1cd3cf0358a24df8829b735a099b47881654ff",
    "PLATFORM_CHECK.md": "019a5c52b6c7adca37c0c95300a633d15c594258928a9080dab24d6b6026952c",
    "check_assignment.py": "511fd29f063829b2bb799398be411ae2e7c77aafc37aac34db8fb6af3a2e9824",
    "data/fixture.json": "27558bc4da7738775879501a6f11a0a9d874f3948823e54bb5e82ab91a02d703",
    "data/zone_co2_readings.csv": "c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4",
}
ARTIFACT_SHA256 = {
    "prepared_panel.csv": "e29aa2ac53cffe29c2f412170100e0725ce4b6a2e0cfdd09e5f1cb92fd5fcd64",
    "hourly_grid.csv": "d4de5178f7ca56960061efb1d263ff4022a9608bc44a6f679c397cb814c150c0",
    "two_hour_summary.csv": "0805cb42799880b85afdee35b0af36c53d606311e1a83a0a995509c647b9d999",
    "temporal_features.csv": "a83d7b858bdcf0203f211cd4dbfc907f0530d132a989ee6ead0fc46e6401d0bb",
    "availability_decisions.csv": "07128f0c67a5765d115c8feb7f3a5ee547450b985b48647c3dcc2324d27a4607",
    "chronological_blocks.csv": "ddde39feea7e3b864088919675fe279dee82021d514980bfd9271cfacf9ec0d2",
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/zone_co2_readings.csv",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}
DELIVERY_FILES = {".classroom50.yaml", ".github/workflows/autograde.yaml"}
FUNCTION_SIGNATURES = {
    "prepare_temporal_panel": ["reading_table", "source_timezone"],
    "build_hourly_grid": ["prepared_table"],
    "build_two_hour_summary": ["prepared_table"],
    "build_past_features": ["prepared_table"],
    "build_chronological_blocks": ["prepared_table", "holdout_start"],
}
BANNED_IMPORT_ROOTS = {
    "matplotlib", "seaborn", "altair", "bokeh", "plotly", "scipy",
    "statsmodels", "sklearn", "xgboost", "tensorflow", "torch", "requests",
    "urllib", "dask", "multiprocessing", "joblib", "numba",
}


class InfrastructureError(RuntimeError):
    """A runner/grader failure for which no student grade is valid."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _context() -> dict[str, str]:
    mapping = {
        "classroom": "CLASSROOM",
        "assignment": "ASSIGNMENT",
        "submission": "SUBMISSION_TAG",
        "commit": "COMMIT_URL",
        "release": "RELEASE_URL",
    }
    result = {}
    for field, variable in mapping.items():
        value = os.environ.get(variable, "").strip()
        if not value:
            raise InfrastructureError(f"missing required Classroom50 runner context: {variable}")
        result[field] = value
    result["review"] = os.environ.get("REVIEW_URL", "").strip() or result["commit"]
    result["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [item for item in ast.walk(node) if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == name]


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


def _load_and_validate_template(root: Path) -> tuple[dict[str, dict], ast.Module]:
    _assert(sys.version_info[:3] == (3, 12, 13), "grader Python differs from 3.12.13")
    _assert(np.__version__ == "2.0.2" and pd.__version__ == "3.0.5", "grader NumPy/pandas pins differ")
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(root).parts[0] != ".git"
        and path.relative_to(root).parts[0] != "output"
    }
    expected_files = STUDENT_PACKAGE_FILES | (actual_files & DELIVERY_FILES)
    _assert(actual_files == expected_files, "student package inventory differs")
    _assert(not any((root / relative).is_symlink() for relative in actual_files & DELIVERY_FILES), "delivery metadata must be regular files")
    for relative, digest in PROTECTED_FILE_SHA256.items():
        path = root / relative
        _assert(path.is_file() and sha256(path.read_bytes()).hexdigest() == digest, f"protected file changed: {relative}")
    gitignore = (root / ".gitignore").read_text()
    _assert("output/" not in gitignore and "*.csv" not in gitignore, "required output is hidden from Git")
    try:
        notebook = json.loads((root / "assignment.ipynb").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"invalid notebook JSON: {error}") from error
    _assert(notebook.get("nbformat") == 4 and notebook.get("nbformat_minor") == 5, "notebook format differs")
    cells = notebook.get("cells", [])
    ids = [cell.get("id") for cell in cells]
    _assert(ids == CELL_IDS and len(ids) == len(set(ids)), "notebook cell IDs/order differ")
    for cell in cells:
        expected_type = "markdown" if cell["id"] in MARKDOWN_IDS else "code"
        _assert(cell.get("cell_type") == expected_type, f"cell type differs: {cell['id']}")
    _assert(notebook.get("metadata", {}).get("kernelspec") == {"display_name": "Python 3", "language": "python", "name": "python3"}, "kernelspec differs")
    by_id = {cell["id"]: cell for cell in cells}
    for cell_id, digest in PROTECTED_CELL_SHA256.items():
        _assert(sha256(_source(by_id[cell_id]).encode()).hexdigest() == digest, f"protected cell changed: {cell_id}")
    student_ids = set(CELL_IDS) - MARKDOWN_IDS - set(PROTECTED_CELL_SHA256)
    student_code = "\n".join(_source(by_id[cell_id]) for cell_id in student_ids)
    student_markdown = "\n".join(_source(by_id[cell_id]) for cell_id in MARKDOWN_IDS - set(PROTECTED_CELL_SHA256))
    _assert("TODO" not in student_code + student_markdown and "NotImplementedError" not in student_code, "submission contains unfinished scaffolds")
    try:
        tree = ast.parse(student_code)
    except SyntaxError as error:
        raise AssertionError(f"student code syntax error: {error}") from error
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = {alias.name.split(".")[0] for alias in node.names}
            _assert(not roots.intersection(BANNED_IMPORT_ROOTS), "out-of-scope import")
            raise AssertionError("student cells must use supplied imports")
    _assert(set(FUNCTION_SIGNATURES).issubset(functions), "required functions missing")
    for name, expected in FUNCTION_SIGNATURES.items():
        _assert([argument.arg for argument in functions[name].args.args] == expected, f"wrong signature: {name}")
        source = ast.unparse(functions[name])
        for canonical in ("atrium", "studio", "2026-01-20", "America/New_York"):
            _assert(canonical not in source, f"{name} hard-codes canonical data")
        _assert(not _calls(functions[name], "read_csv") and not _calls(functions[name], "to_csv"), f"{name} performs file I/O")
    lowered = student_code.lower()
    for fragment in ("shift(-1", "center=true", ".fillna(", ".ffill(", ".bfill(", ".interpolate(", ".expanding(", ".ewm(", ".apply(", ".plot(", "sklearn", "statsmodels", "requests.", "urlopen", "/content", "drive.mount", "random.", "timestamp.now", "datetime.now", "date.today"):
        _assert(fragment not in lowered.replace(" ", ""), f"out-of-scope source: {fragment}")
    return by_id, tree


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _check_task1_source(tree: ast.Module) -> None:
    node = _functions(tree)["prepare_temporal_panel"]
    _assert(len(_calls(node, "to_datetime")) == 1 and _keyword(_calls(node, "to_datetime")[0], "format") == "%Y-%m-%d %H:%M", "Task 1 exact parsing missing")
    _assert(len(_calls(node, "tz_localize")) == 1 and len(_calls(node, "tz_convert")) == 1, "Task 1 timezone sequence differs")
    _assert(len(_calls(node, "sort_values")) == 1 and _keyword(_calls(node, "sort_values")[0], "kind") == "stable" and _literal_argument(_calls(node, "sort_values")[0]) == ["zone", "recorded_at"], "Task 1 stable zone/time sort missing")
    _assert(_calls(node, "copy"), "Task 1 input copy missing")


def _check_task2_source(tree: ast.Module) -> None:
    functions = _functions(tree)
    hourly = functions["build_hourly_grid"]
    summary = functions["build_two_hour_summary"]
    _assert(len(_calls(hourly, "groupby")) == 1 and len(_calls(hourly, "resample")) == 1 and len(_calls(hourly, "asfreq")) == 1, "hourly entity-grid mechanism differs")
    _assert(_calls(hourly, "floor") and "ValueError" in ast.unparse(hourly), "off-grid rejection missing")
    hourly_source = ast.unparse(hourly).replace("'", '"')
    for evidence in ('hourly["source_row"].isna()', 'hourly["source_row"].eq(1)', 'hourly["co2_ppm"].isna()'):
        _assert(evidence in hourly_source, f"hourly provenance rule missing {evidence}")
    _assert(_first_literal(_calls(hourly, "resample")[0]) == "h", "lowercase hourly alias missing")
    _assert(len(_calls(summary, "groupby")) == 1 and len(_calls(summary, "resample")) == 1 and len(_calls(summary, "agg")) == 1, "summary mechanism differs")
    summary_resample = _calls(summary, "resample")[0]
    _assert(_first_literal(summary_resample) == "2h" and _keyword(summary_resample, "closed") == "left" and _keyword(summary_resample, "label") == "left", "two-hour bin boundaries differ")
    named_aggregation = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in _calls(summary, "agg")[0].keywords
        if keyword.arg is not None
    }
    _assert(named_aggregation == {"mean_co2_ppm": ("co2_ppm", "mean"), "reading_count": ("source_row", "sum")}, "summary aggregation roles differ")
    for node in (hourly, summary):
        for policy in ("observed", "sort", "dropna"):
            _assert(_keyword(_calls(node, "groupby")[0], policy) is True, f"groupby {policy}=True missing")


def _check_task3_source(tree: ast.Module) -> None:
    node = _functions(tree)["build_past_features"]
    _assert(len(_calls(node, "shift")) == 2 and len(_calls(node, "diff")) == 1 and len(_calls(node, "rolling")) == 2, "past-only operations differ")
    feature_groups = _calls(node, "groupby")
    _assert(len(feature_groups) == 3, "past-feature entity scoping differs")
    for group in feature_groups:
        _assert(_first_literal(group) == "zone", "past-feature group key differs")
        for policy in ("observed", "sort", "dropna"):
            _assert(_keyword(group, policy) is True, f"past-feature groupby {policy}=True missing")
    _assert(len(_calls(node, "merge")) == 1 and _keyword(_calls(node, "merge")[0], "validate") == "one_to_one", "validated elapsed-window merge missing")
    elapsed_rolls = [call for call in _calls(node, "rolling") if _first_literal(call) == "2h"]
    _assert(len(elapsed_rolls) == 1 and _keyword(elapsed_rolls[0], "closed") == "left" and _keyword(elapsed_rolls[0], "min_periods") == 1, "elapsed window boundary differs")


def _copy_submission(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {"_grader_selftest", ".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", "result.json"}.intersection(names)
    shutil.copytree(source, destination, ignore=ignore)


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {name: (root / "output" / name).read_bytes() for name in ARTIFACT_SHA256}


ALTERNATE_CHECKS = r'''
central_results = {}

def _central_record(name, function):
    try:
        function()
    except Exception as error:
        central_results[name] = {'passed': False, 'detail': f'{type(error).__name__}: {error}'}
    else:
        central_results[name] = {'passed': True, 'detail': 'canonical and disclosed alternate behavior passed'}

def _alternate_raw():
    table = pd.DataFrame(
        [
            ['lab', '2026-02-10 11:00', 850.0],
            ['gallery', '2026-02-10 06:00', 700.0],
            ['lab', '2026-02-10 06:00', 800.0],
            ['gallery', '2026-02-10 12:00', 750.0],
            ['lab', '2026-02-10 08:00', 820.0],
            ['gallery', '2026-02-10 09:00', np.nan],
            ['lab', '2026-02-10 12:00', 860.0],
            ['gallery', '2026-02-10 07:00', 710.0],
            ['lab', '2026-02-10 09:00', 830.0],
            ['gallery', '2026-02-10 10:00', 730.0],
        ],
        index=[31, 4, 88, 12, 57, 6, 73, 20, 2, 45],
        columns=['zone', 'recorded_at', 'co2_ppm'],
    )
    table['zone'] = table['zone'].astype('string')
    table['recorded_at'] = table['recorded_at'].astype('string')
    table['co2_ppm'] = table['co2_ppm'].astype('float64')
    return table

def _task1():
    assert temporal_representation == 'timestamp'
    assert input_row_grain == 'one recorded CO2 reading for one zone and local timestamp'
    assert entity_key == ['zone'] and row_key == ['zone', 'recorded_at'] and sort_keys == ['zone', 'recorded_at']
    assert series_structure == 'panel' and predicted_entities == ['atrium', 'studio']
    assert source_timezone == 'America/New_York' and output_timezone == 'UTC' and predicted_source_rows == 12
    assert predicted_gap_hours == {'atrium': [1.0, 2.0, 1.0, 2.0, 1.0], 'studio': [2.0, 1.0, 2.0, 1.0, 1.0]}
    assert predicted_regularity == {'atrium': 'irregular', 'studio': 'irregular'}
    raw = _alternate_raw(); snapshot = raw.copy(deep=True)
    prepared = prepare_temporal_panel(raw, 'America/Chicago')
    pd.testing.assert_frame_equal(raw, snapshot)
    assert prepared.index.tolist() == list(range(10))
    assert prepared['zone'].tolist() == ['gallery'] * 5 + ['lab'] * 5
    assert str(prepared['recorded_at'].dtype) == 'datetime64[us, UTC]'
    assert prepared['recorded_at'].dt.strftime('%H:%M').tolist() == ['12:00','13:00','15:00','16:00','18:00','12:00','14:00','15:00','17:00','18:00']
    gaps = {name: group['recorded_at'].diff().dropna().dt.total_seconds().div(3600).tolist() for name, group in prepared.groupby('zone', sort=False)}
    assert gaps == {'gallery': [1.0,2.0,1.0,2.0], 'lab': [2.0,1.0,2.0,1.0]}

def _task2():
    prepared = prepare_temporal_panel(_alternate_raw(), 'America/Chicago')
    snapshot = prepared.copy(deep=True)
    hourly = build_hourly_grid(prepared); summary = build_two_hour_summary(prepared)
    pd.testing.assert_frame_equal(prepared, snapshot)
    assert hourly.shape == (14, 6) and int(hourly['grid_created_row'].sum()) == 4 and int(hourly['source_value_missing'].sum()) == 1
    assert int(hourly['source_row'].notna().sum()) == 10
    created = hourly.loc[hourly['grid_created_row'], ['zone','recorded_at']]
    assert created['zone'].tolist() == ['gallery','gallery','lab','lab']
    assert created['recorded_at'].dt.strftime('%H:%M').tolist() == ['14:00','17:00','13:00','16:00']
    assert summary[['zone','recorded_at']].values.tolist() == [[zone, pd.Timestamp(f'2026-02-10 {hour}:00', tz='UTC')] for zone in ['gallery','lab'] for hour in ['12','14','16','18']]
    assert summary['mean_co2_ppm'].tolist()[:1] == [705.0] and pd.isna(summary['mean_co2_ppm'].iloc[1])
    assert summary['mean_co2_ppm'].tolist()[2:] == [730.0,750.0,800.0,825.0,850.0,860.0]
    assert summary['reading_count'].tolist() == [2,1,1,1,1,2,1,1]
    off_grid = prepared.copy(deep=True); off_grid.loc[off_grid.index[0], 'recorded_at'] += pd.Timedelta(minutes=30)
    try:
        build_hourly_grid(off_grid)
    except ValueError as error:
        assert 'whole UTC hour' in str(error)
    else:
        raise AssertionError('off-grid source timestamp was not rejected')

def _task3():
    prepared = prepare_temporal_panel(_alternate_raw(), 'America/Chicago')
    snapshot = prepared.copy(deep=True)
    features = build_past_features(prepared)
    blocks = build_chronological_blocks(prepared, pd.Timestamp('2026-02-10 17:00', tz='UTC'))
    pd.testing.assert_frame_equal(prepared, snapshot)
    row = features.loc[features['zone'].eq('lab') & features['recorded_at'].eq(pd.Timestamp('2026-02-10 17:00', tz='UTC'))].iloc[0]
    assert [row['co2_lag_1'], row['co2_difference'], row['mean_previous_2_observations'], row['mean_previous_2h']] == [830.0,20.0,825.0,830.0]
    assert features.groupby('zone', sort=False).head(1)['co2_lag_1'].isna().all()
    assert blocks['block'].value_counts().to_dict() == {'earlier': 7, 'later_holdout': 3}
    earlier = blocks.loc[blocks['block'].eq('earlier')]; later = blocks.loc[blocks['block'].eq('later_holdout')]
    assert earlier['recorded_at'].max() < later['recorded_at'].min()
    assert set(earlier['zone']) == {'gallery','lab'} and set(later['zone']) == {'gallery','lab'}

_central_record('task1', _task1)
_central_record('task2', _task2)
_central_record('task3', _task3)
(ASSIGNMENT_ROOT / '__central_checks.json').write_text(json.dumps(central_results), encoding='utf-8')
'''


def _execute(root: Path, cwd: Path, extra_source: str | None = None) -> None:
    notebook = nbformat.read(root / "assignment.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    if extra_source:
        notebook.cells.append(nbformat.v4.new_code_cell(extra_source, id="a09-central-checks"))
    previous = os.environ.get("PYTHONDONTWRITEBYTECODE")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}, allow_errors=False).execute()
    except CellExecutionError as error:
        lines = str(error).strip().splitlines()
        raise AssertionError("fresh notebook execution failed: " + " | ".join(lines[-12:])) from error
    finally:
        if previous is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = previous


def _check_artifacts(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "missing regular output directory")
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(actual == set(ARTIFACT_SHA256) | {".gitkeep"}, "exact output inventory differs")
    for name, digest in ARTIFACT_SHA256.items():
        data = (output / name).read_bytes()
        _assert(data.endswith(b"\n") and b"\r" not in data and sha256(data).hexdigest() == digest, f"artifact differs: {name}")


def _check_canonical_values(root: Path) -> None:
    prepared = pd.read_csv(root / "output/prepared_panel.csv", dtype={"zone": "string", "co2_ppm": "float64", "source_row": "int64"}, parse_dates=["recorded_at"])
    _assert(prepared.shape == (12, 4) and str(prepared["recorded_at"].dtype) == "datetime64[us, UTC]", "prepared schema differs")
    hourly = pd.read_csv(root / "output/hourly_grid.csv", dtype={"zone": "string", "co2_ppm": "float64", "source_row": "float64", "grid_created_row": "bool", "source_value_missing": "bool"}, parse_dates=["recorded_at"])
    _assert(hourly.shape == (16, 6) and int(hourly["grid_created_row"].sum()) == 4 and int(hourly["source_value_missing"].sum()) == 1 and int(hourly["source_row"].notna().sum()) == 12, "hourly provenance differs")
    summary = pd.read_csv(root / "output/two_hour_summary.csv", dtype={"zone": "string", "mean_co2_ppm": "float64", "reading_count": "int64"}, parse_dates=["recorded_at"])
    _assert(summary["reading_count"].tolist() == [2,1,1,2,1,2,1,2] and int(summary["reading_count"].sum()) == 12, "resample counts differ")
    features = pd.read_csv(root / "output/temporal_features.csv", dtype={name: "float64" for name in ["co2_ppm","co2_lag_1","co2_difference","mean_previous_2_observations","mean_previous_2h"]} | {"zone": "string"}, parse_dates=["recorded_at"])
    row = features.loc[features["zone"].eq("studio") & features["recorded_at"].eq(pd.Timestamp("2026-01-20 17:00", tz="UTC"))].iloc[0]
    _assert([row["co2_lag_1"], row["co2_difference"], row["mean_previous_2_observations"], row["mean_previous_2h"]] == [540.0,20.0,530.0,540.0], "canonical past features differ")
    availability = pd.read_csv(root / "output/availability_decisions.csv", dtype={"candidate": "string", "available_by_prediction_time": "bool", "decision": "string"}, parse_dates=["latest_required_timestamp"])
    _assert(availability["decision"].tolist() == ["keep","keep","reject","reject"], "availability differs")
    blocks = pd.read_csv(root / "output/chronological_blocks.csv", dtype={"zone": "string", "co2_ppm": "float64", "source_row": "int64", "block": "string"}, parse_dates=["recorded_at"])
    _assert(blocks["block"].value_counts().to_dict() == {"earlier": 8, "later_holdout": 4}, "block counts differ")


def _result_test(name: str, maximum: int, error: Exception | None) -> dict:
    passed = error is None
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {'all automated checks passed' if passed else error}")
    return {"test-name": name, "passed": passed, "score": maximum if passed else 0, "max-score": maximum}


def grade_submission(submission_root: str | Path) -> dict:
    """Grade a submission and return an official Classroom50 result object."""
    context = _context()
    root = Path(submission_root).resolve()
    specs = (
        ("Template, environment, fixture, notebook, and protected integrity", 10),
        ("Task 1 temporal structure and preparation", 20),
        ("Task 2 entity-scoped frequency and provenance", 25),
        ("Task 3 past-only evidence and chronological blocks", 30),
        ("Portability, visible outputs, repeatability, and resubmission", 5),
    )
    errors: dict[str, Exception | None] = {name: None for name, _ in specs}
    template_name, task1_name, task2_name, task3_name, portability_name = [name for name, _ in specs]
    try:
        _by_id, tree = _load_and_validate_template(root)
    except Exception as error:
        for name in errors:
            errors[name] = error
    else:
        for name, check in ((task1_name, _check_task1_source), (task2_name, _check_task2_source), (task3_name, _check_task3_source)):
            try:
                check(tree)
            except Exception as error:
                errors[name] = error
        try:
            _check_artifacts(root)
        except Exception as error:
            errors[portability_name] = error
        with tempfile.TemporaryDirectory(prefix="a09-central-") as temporary_name:
            temporary = Path(temporary_name)
            flat = temporary / "flattened submission with spaces"
            _copy_submission(root, flat)
            output = flat / "output"
            output.mkdir(exist_ok=True)
            for name in ARTIFACT_SHA256:
                (output / name).write_text("stale,artifact\n", encoding="utf-8")
            try:
                _execute(flat, flat, ALTERNATE_CHECKS)
                alternate_path = flat / "__central_checks.json"
                alternate = json.loads(alternate_path.read_text())
                alternate_path.unlink()
            except Exception as error:
                for name in (task1_name, task2_name, task3_name, portability_name):
                    if errors[name] is None:
                        errors[name] = error
            else:
                for key, name in (("task1", task1_name), ("task2", task2_name), ("task3", task3_name)):
                    if errors[name] is None and not alternate[key]["passed"]:
                        errors[name] = AssertionError(alternate[key]["detail"])
                for name, check in ((task1_name, _check_canonical_values), (task2_name, _check_canonical_values), (task3_name, _check_canonical_values)):
                    if errors[name] is None:
                        try:
                            check(flat)
                        except Exception as error:
                            errors[name] = error
                if errors[portability_name] is None:
                    try:
                        _check_artifacts(flat)
                        fresh = _artifact_bytes(flat)
                        _assert(fresh == _artifact_bytes(root), "committed output differs from fresh execution")
                        nested_cwd = flat / "nested" / "working directory"
                        nested_cwd.mkdir(parents=True)
                        _execute(flat, nested_cwd)
                        _assert(_artifact_bytes(flat) == fresh, "nested repeat differs")
                        course_root = temporary / "relocated course"
                        course_assignment = course_root / "09" / "assignment"
                        course_assignment.parent.mkdir(parents=True)
                        _copy_submission(root, course_assignment)
                        for artifact in ARTIFACT_SHA256:
                            (course_assignment / "output" / artifact).unlink(missing_ok=True)
                        _execute(course_assignment, course_root)
                        _assert(_artifact_bytes(course_assignment) == fresh, "course-root layout differs")
                        relocated = temporary / "relocated standalone"
                        _copy_submission(root, relocated)
                        for artifact in ARTIFACT_SHA256:
                            (relocated / "output" / artifact).unlink(missing_ok=True)
                        _execute(relocated, relocated / "data")
                        _assert(_artifact_bytes(relocated) == fresh, "nested-within-assignment layout differs")
                    except Exception as error:
                        errors[portability_name] = error
    tests = [_result_test(name, maximum, errors[name]) for name, maximum in specs]
    return {"schema": "classroom50/result/v1", **context, "score": sum(test["score"] for test in tests), "max-score": 90, "tests": tests}


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    result_path = Path("result.json")
    try:
        if result_path.exists():
            if not result_path.is_file():
                raise InfrastructureError("result.json path is not a regular file")
            result_path.unlink()
        result = grade_submission(target)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except Exception as error:
        if result_path.is_file():
            result_path.unlink()
        print(f"Grader infrastructure failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
