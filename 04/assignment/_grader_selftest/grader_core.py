"""Instructor-owned disposable grader core for Assignment 04.

This module intentionally does not import or execute the student-editable
public checker. It validates the written contract in temporary copies, clears
stored notebook evidence, deletes prior CSV outputs, injects instructor-owned
verification, and executes fresh kernels.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import pandas as pd


BUNDLE_DIR = Path(__file__).resolve().parent
ALTERNATE_FIXTURE_DIR = BUNDLE_DIR / "alternate_fixture"
PROTECTED_HASHES_PATH = BUNDLE_DIR / "protected_files.json"
REQUIRED_COPY_PATHS = (
    ".gitignore",
    ".python-version",
    "PLATFORM_CHECK.md",
    "README.md",
    "assignment.ipynb",
    "check_assignment.py",
    "requirements.txt",
    "data/fixture.json",
    "data/purchases.csv",
    ".github/test/requirements.txt",
    ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
)
DELIVERY_FILES = {".classroom50.yaml", ".github/workflows/autograde.yaml"}
REQUIRED_CELL_IDS = {
    "a04-header",
    "a04-supplied-setup",
    "a04-task1-heading",
    "a04-task1-dependent",
    "a04-task1-producer",
    "a04-task1-explanation",
    "a04-task2-heading",
    "a04-task2-objects",
    "a04-task3-heading",
    "a04-task3-roundtrip",
    "a04-final-heading",
    "a04-final-verification",
}
EXPECTED_SETUP_SHA256 = (
    "f0bbad6712f084881860ed0f64d1d2067a3045ac69aadffdc9457da62def2138"
)
EXPECTED_PRODUCER = 'base_rate = 3\nprint("base_rate:", base_rate)'
EXPECTED_DEPENDENT = (
    'adjusted_rate = base_rate + 2\nprint("adjusted_rate:", adjusted_rate)'
)


@dataclass(frozen=True)
class GradeTest:
    name: str
    max_score: int
    passed: bool
    detail: str

    @property
    def score(self) -> int:
        return self.max_score if self.passed else 0


@dataclass(frozen=True)
class VariantResult:
    execution_error: str | None
    artifact_error: str | None


def _cell_source(cell) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source if isinstance(source, str) else ""


def _check_protected_files(submission: Path) -> str | None:
    try:
        protected = json.loads(PROTECTED_HASHES_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Could not load instructor protected-file manifest: {error}") from error

    errors = []
    for relative, expected_hash in protected.items():
        path = submission / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            errors.append(f"edited {relative}")
    return "; ".join(errors) or None


def _check_submission_inventory(submission: Path) -> str | None:
    actual = {
        path.relative_to(submission).as_posix()
        for path in submission.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(submission).parts[0] != ".git"
        and path.relative_to(submission).parts[0] != "output"
    }
    expected = set(REQUIRED_COPY_PATHS) | (actual & DELIVERY_FILES)
    if actual != expected:
        return "student package inventory differs"
    if any((submission / relative).is_symlink() for relative in actual & DELIVERY_FILES):
        return "delivery metadata must be regular files"
    return None


def _load_notebook(path: Path):
    if not path.is_file():
        raise AssertionError("Missing assignment.ipynb.")
    try:
        return nbformat.read(path, as_version=4)
    except Exception as error:
        raise AssertionError(f"assignment.ipynb is not valid notebook JSON: {error}") from error


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _name_assignments(tree: ast.Module, name: str) -> list[ast.AST]:
    values = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ):
            values.append(node.value)
    return values


def _indexer_subscripts(node: ast.AST, owner: str, attribute: str) -> list[ast.Subscript]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Subscript)
        and isinstance(candidate.value, ast.Attribute)
        and candidate.value.attr == attribute
        and isinstance(candidate.value.value, ast.Name)
        and candidate.value.value.id == owner
    ]


def _slice_bounds(node: ast.AST, lower, upper) -> bool:
    return (
        isinstance(node, ast.Slice)
        and isinstance(node.lower, ast.Constant)
        and node.lower.value == lower
        and isinstance(node.upper, ast.Constant)
        and node.upper.value == upper
        and node.step is None
    )


def _literal_list(node: ast.AST) -> list | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = []
    for item in node.elts:
        if not isinstance(item, ast.Constant):
            return None
        values.append(item.value)
    return values


def _call_is(node: ast.AST, dotted_name: str) -> bool:
    return isinstance(node, ast.Call) and _attribute_name(node.func) == dotted_name


def _subscript(node: ast.AST, owner: str, key) -> bool:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return False
    if node.value.id != owner:
        return False
    slice_node = node.slice
    if isinstance(key, list):
        return _literal_list(slice_node) == key
    return isinstance(slice_node, ast.Constant) and slice_node.value == key


def _check_task2_source(source: str) -> None:
    if "TODO" in source:
        raise AssertionError("Replace every Task 2 TODO in the supplied task cell.")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AssertionError(f"Task 2 has a SyntaxError at line {error.lineno}: {error.msg}.") from error

    reading_values = _name_assignments(tree, "reading_by_site")
    if not any(_call_is(value, "pd.Series") for value in reading_values):
        raise AssertionError("Create reading_by_site with pd.Series(...).")
    table_values = _name_assignments(tree, "measurement_table")
    if not any(_call_is(value, "pd.DataFrame") for value in table_values):
        raise AssertionError("Create measurement_table with pd.DataFrame(...).")

    baseline_series = _name_assignments(tree, "baseline_series")
    if not any(_subscript(value, "measurement_table", "baseline_c") for value in baseline_series):
        raise AssertionError('Create baseline_series with measurement_table["baseline_c"].')
    baseline_table = _name_assignments(tree, "baseline_table")
    if not any(_subscript(value, "measurement_table", ["baseline_c"]) for value in baseline_table):
        raise AssertionError('Create baseline_table with measurement_table[["baseline_c"]].')

    label_values = _name_assignments(tree, "label_block")
    correct_label_selection = False
    for value in label_values:
        for selection in _indexer_subscripts(value, "measurement_table", "loc"):
            if (
                isinstance(selection.slice, ast.Tuple)
                and len(selection.slice.elts) == 2
                and _slice_bounds(selection.slice.elts[0], "site-102", "site-103")
                and _literal_list(selection.slice.elts[1])
                == ["baseline_c", "follow_up_c"]
            ):
                correct_label_selection = True
    if not correct_label_selection:
        raise AssertionError(
            "Create label_block with .loc from site-102 through site-103 and both columns."
        )

    position_values = _name_assignments(tree, "position_block")
    correct_position_selection = False
    for value in position_values:
        for selection in _indexer_subscripts(value, "measurement_table", "iloc"):
            if (
                isinstance(selection.slice, ast.Tuple)
                and len(selection.slice.elts) == 2
                and _slice_bounds(selection.slice.elts[0], 1, 3)
                and _slice_bounds(selection.slice.elts[1], 0, 2)
            ):
                correct_position_selection = True
    if not correct_position_selection:
        raise AssertionError("Create position_block with measurement_table.iloc[1:3, 0:2].")

    index_name_assignment = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "name"
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "index"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "measurement_table"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value == "record_id"
        for node in ast.walk(tree)
    )
    if not index_name_assignment:
        raise AssertionError('Set measurement_table.index.name = "record_id".')

    label_writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_csv"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "label_block"
    ]
    if not label_writes:
        raise AssertionError("Write label_block to LABELED_OUTPUT_PATH with to_csv().")


def _check_task3_source(source: str) -> None:
    if "TODO" in source:
        raise AssertionError("Replace every Task 3 TODO in the supplied task cell.")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AssertionError(f"Task 3 has a SyntaxError at line {error.lineno}: {error.msg}.") from error

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise AssertionError("Task 3 needs no additional imports; use the supplied setup names.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if node.value.startswith("/") or "/content" in lowered or "drive" in lowered:
                raise AssertionError("Task code may not use an absolute, /content, or Drive-dependent path.")

    purchases_values = _name_assignments(tree, "purchases")
    if not any(
        _call_is(value, "pd.read_csv")
        and value.args
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "DATA_PATH"
        for value in purchases_values
    ):
        raise AssertionError("Read purchases with pd.read_csv(DATA_PATH).")

    mask_values = _name_assignments(tree, "quantity_at_least_two")
    correct_mask = any(
        isinstance(value, ast.Compare)
        and _subscript(value.left, "purchases", "quantity")
        and len(value.ops) == 1
        and isinstance(value.ops[0], ast.GtE)
        and len(value.comparators) == 1
        and isinstance(value.comparators[0], ast.Constant)
        and value.comparators[0].value == 2
        for value in mask_values
    )
    if not correct_mask:
        raise AssertionError('Use exactly quantity_at_least_two = purchases["quantity"] >= 2.')

    selected_values = _name_assignments(tree, "selected_purchases")
    correct_selected = False
    for value in selected_values:
        for selection in _indexer_subscripts(value, "purchases", "loc"):
            if (
                isinstance(selection.slice, ast.Tuple)
                and len(selection.slice.elts) == 2
                and isinstance(selection.slice.elts[0], ast.Name)
                and selection.slice.elts[0].id == "quantity_at_least_two"
                and _literal_list(selection.slice.elts[1])
                == ["purchase_id", "item", "quantity", "unit_price"]
            ):
                correct_selected = True
    if not correct_selected:
        raise AssertionError(
            "Create selected_purchases with purchases.loc[quantity_at_least_two, explicit source columns]."
        )

    line_total_assignment = any(
        isinstance(node, ast.Assign)
        and any(_subscript(target, "selected_purchases", "line_total") for target in node.targets)
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Mult)
        and _subscript(node.value.left, "selected_purchases", "quantity")
        and _subscript(node.value.right, "selected_purchases", "unit_price")
        for node in ast.walk(tree)
    )
    if not line_total_assignment:
        raise AssertionError("Create line_total as selected quantity * selected unit_price.")

    correct_sort = False
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sort_values"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if (
            _literal_list(keywords.get("by")) == ["line_total", "purchase_id"]
            and _literal_list(keywords.get("ascending")) == [False, True]
        ):
            correct_sort = True
    if not correct_sort:
        raise AssertionError(
            "Sort with by=['line_total', 'purchase_id'] and ascending=[False, True]."
        )

    correct_write = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_csv"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "selected_purchases"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "SELECTED_OUTPUT_PATH"
        and any(
            keyword.arg == "index"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )
    if not correct_write:
        raise AssertionError("Write selected_purchases to SELECTED_OUTPUT_PATH with index=False.")

    round_trip_values = _name_assignments(tree, "round_trip")
    if not any(
        _call_is(value, "pd.read_csv")
        and value.args
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "SELECTED_OUTPUT_PATH"
        for value in round_trip_values
    ):
        raise AssertionError("Read SELECTED_OUTPUT_PATH back into round_trip with pd.read_csv().")


def _check_notebook_contract(submission: Path) -> tuple[str | None, str | None]:
    try:
        notebook = _load_notebook(submission / "assignment.ipynb")
    except AssertionError as error:
        return str(error), str(error)

    try:
        cells = notebook.cells
        ids = [cell.get("id") for cell in cells]
        if not all(isinstance(cell_id, str) for cell_id in ids):
            raise AssertionError("Every notebook cell must retain a stable string ID.")
        if len(ids) != len(set(ids)):
            raise AssertionError("Notebook cell IDs must remain unique.")
        missing = sorted(REQUIRED_CELL_IDS - set(ids))
        if missing:
            raise AssertionError(f"Missing supplied notebook cells: {', '.join(missing)}.")
        kernelspec = notebook.metadata.get("kernelspec", {})
        if not (
            kernelspec.get("name") == "python3"
            and kernelspec.get("display_name") == "Python 3"
            and kernelspec.get("language") == "python"
        ):
            raise AssertionError("Keep the portable Python 3 kernelspec.")
        by_id = {cell["id"]: cell for cell in cells}
        setup = _cell_source(by_id["a04-supplied-setup"])
        if sha256(setup.encode("utf-8")).hexdigest() != EXPECTED_SETUP_SHA256:
            raise AssertionError("The supplied setup cell was edited.")
    except AssertionError as error:
        return str(error), str(error)

    try:
        if _cell_source(by_id["a04-task1-producer"]) != EXPECTED_PRODUCER:
            raise AssertionError("Restore the supplied producer cell and move it without rewriting it.")
        if _cell_source(by_id["a04-task1-dependent"]) != EXPECTED_DEPENDENT:
            raise AssertionError("Restore the supplied dependent cell and move it without rewriting it.")
        if ids.index("a04-task1-producer") >= ids.index("a04-task1-dependent"):
            raise AssertionError("Move the producer cell above the dependent cell.")
        _check_task2_source(_cell_source(by_id["a04-task2-objects"]))
        _check_task3_source(_cell_source(by_id["a04-task3-roundtrip"]))
    except AssertionError as error:
        return None, str(error)
    return None, None


def _copy_submission(submission: Path, project: Path) -> None:
    project.mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_COPY_PATHS:
        source = submission / relative
        if not source.is_file():
            continue
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (project / "output").mkdir(parents=True, exist_ok=True)


def _replace_with_alternate_fixture(project: Path) -> None:
    for filename in ("purchases.csv", "fixture.json"):
        shutil.copy2(
            ALTERNATE_FIXTURE_DIR / filename,
            project / "data" / filename,
        )


def _remove_generated_outputs(project: Path) -> None:
    for filename in ("labeled_block.csv", "selected_purchases.csv"):
        path = project / "output" / filename
        if path.exists():
            path.unlink()


INJECTED_VERIFICATION = r'''# Instructor-owned verification appended only to the disposable grader copy.
assert base_rate == 3
assert adjusted_rate == 5

assert isinstance(reading_by_site, pd.Series)
assert reading_by_site.index.tolist() == ["north", "south", "east", "west"]
assert reading_by_site.name == "reading_c"
np.testing.assert_allclose(reading_by_site.to_numpy(), np.array([12.5, 15.0, 11.5, 15.5]))

_expected_measurement_table = pd.DataFrame(
    [[12, 18], [15, 23], [10, 17], [15, 23]],
    index=["site-101", "site-102", "site-103", "site-104"],
    columns=["baseline_c", "follow_up_c"],
)
_expected_measurement_table.index.name = "record_id"
assert isinstance(measurement_table, pd.DataFrame)
pd.testing.assert_frame_equal(measurement_table, _expected_measurement_table)
assert isinstance(baseline_series, pd.Series)
assert isinstance(baseline_table, pd.DataFrame)
pd.testing.assert_series_equal(baseline_series, _expected_measurement_table["baseline_c"])
pd.testing.assert_frame_equal(baseline_table, _expected_measurement_table[["baseline_c"]])

_expected_block = _expected_measurement_table.loc[
    "site-102":"site-103",
    ["baseline_c", "follow_up_c"],
]
assert isinstance(label_block, pd.DataFrame)
assert isinstance(position_block, pd.DataFrame)
pd.testing.assert_frame_equal(label_block, _expected_block)
pd.testing.assert_frame_equal(position_block, _expected_block)

assert isinstance(purchases, pd.DataFrame)
assert isinstance(quantity_at_least_two, pd.Series)
assert quantity_at_least_two.dtype == bool
assert quantity_at_least_two.index.equals(purchases.index)
pd.testing.assert_series_equal(
    quantity_at_least_two,
    purchases["quantity"] >= 2,
    check_names=False,
)

_source_columns = ["purchase_id", "item", "quantity", "unit_price"]
_expected_selected = purchases.loc[
    purchases["quantity"] >= 2,
    _source_columns,
].copy()
_expected_selected["line_total"] = (
    _expected_selected["quantity"] * _expected_selected["unit_price"]
)
_expected_selected = _expected_selected.sort_values(
    by=["line_total", "purchase_id"],
    ascending=[False, True],
)
assert isinstance(selected_purchases, pd.DataFrame)
pd.testing.assert_frame_equal(selected_purchases, _expected_selected)
np.testing.assert_allclose(
    selected_purchases["line_total"],
    selected_purchases["quantity"] * selected_purchases["unit_price"],
)

assert LABELED_OUTPUT_PATH.is_file()
assert SELECTED_OUTPUT_PATH.is_file()
_written_label = pd.read_csv(LABELED_OUTPUT_PATH, index_col="record_id")
pd.testing.assert_frame_equal(_written_label, _expected_block)
_written_selected = pd.read_csv(SELECTED_OUTPUT_PATH)
assert list(_written_selected.columns) == _source_columns + ["line_total"]
assert not any(str(column).startswith("Unnamed:") for column in _written_selected.columns)
pd.testing.assert_frame_equal(
    _written_selected.reset_index(drop=True),
    _expected_selected.reset_index(drop=True),
    check_dtype=False,
)
assert isinstance(round_trip, pd.DataFrame)
pd.testing.assert_frame_equal(
    round_trip.reset_index(drop=True),
    _written_selected.reset_index(drop=True),
    check_dtype=False,
)

print("Instructor disposable verification passed")
'''


def execute_notebook_in_place(project: Path, *, nested_launch: bool = False) -> str | None:
    """Fresh-execute one project after deleting outputs and stored notebook evidence."""
    _remove_generated_outputs(project)
    notebook_path = project / "assignment.ipynb"
    try:
        notebook = _load_notebook(notebook_path)
    except AssertionError as error:
        return str(error)

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            INJECTED_VERIFICATION,
            id="a04-instructor-verification",
        )
    )

    launch_dir = project
    if nested_launch:
        launch_dir = project / "launch" / "from" / "nested"
        launch_dir.mkdir(parents=True, exist_ok=True)
    try:
        NotebookClient(
            notebook,
            timeout=90,
            kernel_name="python3",
            resources={"metadata": {"path": str(launch_dir)}},
        ).execute()
    except CellExecutionError as error:
        lines = str(error).strip().splitlines()
        return lines[-1] if lines else "Notebook cell execution failed."
    except Exception as error:
        return f"Fresh notebook execution failed: {error}"
    return None


def _validate_artifacts(project: Path) -> str | None:
    try:
        labeled = pd.read_csv(project / "output" / "labeled_block.csv")
        expected_labeled = pd.DataFrame(
            {
                "record_id": ["site-102", "site-103"],
                "baseline_c": [15, 10],
                "follow_up_c": [23, 17],
            }
        )
        pd.testing.assert_frame_equal(labeled, expected_labeled, check_dtype=False)

        purchases = pd.read_csv(project / "data" / "purchases.csv")
        source_columns = ["purchase_id", "item", "quantity", "unit_price"]
        expected = purchases.loc[purchases["quantity"] >= 2, source_columns].copy()
        expected["line_total"] = expected["quantity"] * expected["unit_price"]
        expected = expected.sort_values(
            by=["line_total", "purchase_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        selected = pd.read_csv(project / "output" / "selected_purchases.csv")
        if list(selected.columns) != source_columns + ["line_total"]:
            raise AssertionError("selected_purchases.csv has the wrong schema or a serialized index.")
        pd.testing.assert_frame_equal(
            selected.reset_index(drop=True),
            expected,
            check_dtype=False,
        )
    except Exception as error:
        return str(error) or error.__class__.__name__
    return None


def _run_variant(
    submission: Path,
    *,
    prefix: str,
    relocated: bool = False,
    alternate: bool = False,
) -> VariantResult:
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        temporary_root = Path(temporary)
        project = temporary_root / "project"
        if relocated:
            project = temporary_root / "unrelated" / "deep" / "checkout" / "a04"
        _copy_submission(submission, project)
        if alternate:
            _replace_with_alternate_fixture(project)
        execution_error = execute_notebook_in_place(project, nested_launch=relocated)
        artifact_error = _validate_artifacts(project)
        return VariantResult(execution_error, artifact_error)


def grade_submission(submission: Path) -> list[GradeTest]:
    submission = submission.resolve()
    protected_error = _check_protected_files(submission)
    inventory_error = _check_submission_inventory(submission)
    json_error, source_error = _check_notebook_contract(submission)

    canonical = _run_variant(
        submission,
        prefix="ds217-a04-canonical-",
    )
    relocated = _run_variant(
        submission,
        prefix="ds217-a04-relocated-",
        relocated=True,
    )
    alternate = _run_variant(
        submission,
        prefix="ds217-a04-alternate-",
        relocated=True,
        alternate=True,
    )

    package_detail = "; ".join(
        detail for detail in (inventory_error, protected_error, json_error) if detail
    )
    source_detail = source_error or ""
    canonical_detail = canonical.execution_error or ""
    canonical_artifact_detail = canonical.artifact_error or ""
    relocated_detail = "; ".join(
        detail for detail in (relocated.execution_error, relocated.artifact_error) if detail
    )
    alternate_detail = "; ".join(
        detail for detail in (alternate.execution_error, alternate.artifact_error) if detail
    )

    return [
        GradeTest(
            "protected package and valid notebook JSON",
            1,
            not package_detail,
            package_detail,
        ),
        GradeTest(
            "task source contract and repaired state order",
            2,
            json_error is None and not source_detail,
            json_error or source_detail,
        ),
        GradeTest(
            "fresh canonical notebook execution and injected state checks",
            2,
            canonical.execution_error is None,
            canonical_detail,
        ),
        GradeTest(
            "new canonical CSV artifacts",
            1,
            canonical.execution_error is None and canonical.artifact_error is None,
            canonical.execution_error or canonical_artifact_detail,
        ),
        GradeTest(
            "relocated checkout and nested launch",
            2,
            not relocated_detail,
            relocated_detail,
        ),
        GradeTest(
            "alternate valid fixture and deterministic tie handling",
            2,
            not alternate_detail,
            alternate_detail,
        ),
    ]
