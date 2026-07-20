# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "numpy==2.0.2",
#   "pandas==3.0.3",
#   "nbclient==0.11.0",
#   "nbformat==5.10.4",
#   "ipykernel==7.3.0",
# ]
# ///

"""Materialize submissions and adversarially validate Assignment 04 grading."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
SELFTEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSIGNMENT_DIR))
sys.path.insert(0, str(SELFTEST_DIR))

from check_assignment import run_public_checks  # noqa: E402
from grader_core import (  # noqa: E402
    ALTERNATE_FIXTURE_DIR,
    REQUIRED_COPY_PATHS,
    execute_notebook_in_place,
    grade_submission,
)


PROTECTED_TEST = "protected package and valid notebook JSON"
SOURCE_TEST = "task source contract and repaired state order"
CANONICAL_TEST = "fresh canonical notebook execution and injected state checks"
ARTIFACT_TEST = "new canonical CSV artifacts"
RELOCATED_TEST = "relocated checkout and nested launch"
ALTERNATE_TEST = "alternate valid fixture and deterministic tie handling"
REQUIRED_RUNNER_ENV = (
    "CLASSROOM",
    "ASSIGNMENT",
    "SUBMISSION_TAG",
    "COMMIT_URL",
    "RELEASE_URL",
)
UTC_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

CORRECT_TASK2 = '''reading_values = np.array([12.5, 15.0, 11.5, 15.5])

reading_by_site = pd.Series(
    reading_values,
    index=["north", "south", "east", "west"],
    name="reading_c",
)

measurement_values = np.array(
    [
        [12, 18],
        [15, 23],
        [10, 17],
        [15, 23],
    ]
)

measurement_table = pd.DataFrame(
    measurement_values,
    index=["site-101", "site-102", "site-103", "site-104"],
    columns=["baseline_c", "follow_up_c"],
)
measurement_table.index.name = "record_id"

baseline_series = measurement_table["baseline_c"]
baseline_table = measurement_table[["baseline_c"]]

label_block = measurement_table.loc[
    "site-102":"site-103",
    ["baseline_c", "follow_up_c"],
]
position_block = measurement_table.iloc[1:3, 0:2]

print("Series metadata:", reading_by_site.index, reading_by_site.dtype, reading_by_site.name)
print("DataFrame metadata:", measurement_table.index, measurement_table.columns)
print("shape:", measurement_table.shape)
print("dtypes:")
print(measurement_table.dtypes)
print("bracket return types:", type(baseline_series), type(baseline_table))

pd.testing.assert_frame_equal(label_block, position_block)
label_block.to_csv(LABELED_OUTPUT_PATH)
print("wrote:", LABELED_OUTPUT_PATH)'''

CORRECT_TASK3 = '''purchases = pd.read_csv(DATA_PATH)

print("shape:", purchases.shape)
print("columns:", purchases.columns)
print("dtypes:")
print(purchases.dtypes)
print(purchases.head())

quantity_at_least_two = purchases["quantity"] >= 2

selected_purchases = purchases.loc[
    quantity_at_least_two,
    ["purchase_id", "item", "quantity", "unit_price"],
].copy()
selected_purchases["line_total"] = (
    selected_purchases["quantity"] * selected_purchases["unit_price"]
)
selected_purchases = selected_purchases.sort_values(
    by=["line_total", "purchase_id"],
    ascending=[False, True],
)
selected_purchases.to_csv(SELECTED_OUTPUT_PATH, index=False)

round_trip = pd.read_csv(SELECTED_OUTPUT_PATH)
round_trip'''

CORRECT_EXPLANATION = '''### State explanation

The notebook source has a visible top-to-bottom cell order, while execution order records which cells the kernel actually ran. The kernel can retain names from an earlier order, so a value or stored output may remain visible even when the current source would fail in a fresh kernel. I moved the complete `base_rate` producer cell above its dependent cell, then restarted the kernel and ran all cells. That fresh run, not the stored output, shows that the repaired visible order is reproducible.'''


def _source(cell) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _set_source(cell, source: str) -> None:
    cell["source"] = source


def _load_notebook(root: Path) -> dict:
    return json.loads((root / "assignment.ipynb").read_text(encoding="utf-8"))


def _write_notebook(root: Path, notebook: dict) -> None:
    (root / "assignment.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n",
        encoding="utf-8",
    )


def copy_student_package(root: Path) -> None:
    for relative in REQUIRED_COPY_PATHS:
        source = ASSIGNMENT_DIR / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (root / "output").mkdir(parents=True, exist_ok=True)
    (root / "output" / ".gitkeep").write_text("", encoding="utf-8")


def complete_notebook(root: Path) -> None:
    notebook = _load_notebook(root)
    cells = notebook["cells"]
    by_id = {cell["id"]: cell for cell in cells}

    producer = by_id["a04-task1-producer"]
    cells.remove(producer)
    dependent_index = next(
        index for index, cell in enumerate(cells) if cell["id"] == "a04-task1-dependent"
    )
    cells.insert(dependent_index, producer)

    _set_source(by_id["a04-task1-explanation"], CORRECT_EXPLANATION)
    _set_source(by_id["a04-task2-objects"], CORRECT_TASK2)
    _set_source(by_id["a04-task3-roundtrip"], CORRECT_TASK3)
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    _write_notebook(root, notebook)


def public_failures(root: Path) -> dict[str, str]:
    return {
        name: error
        for name, error in run_public_checks(root)
        if error is not None
    }


def central_results(root: Path):
    return {test.name: test for test in grade_submission(root)}


def assert_all_central_pass(root: Path, label: str) -> None:
    results = central_results(root)
    failures = {name: test.detail for name, test in results.items() if not test.passed}
    if failures:
        raise AssertionError(f"{label}: central grader failures: {failures}")


def assert_delivery_inventory(correct: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ds217-a04-inventory-") as temporary:
        accepted = Path(temporary) / "accepted"
        shutil.copytree(correct, accepted)
        (accepted / ".classroom50.yaml").write_text("version: 1\n", encoding="utf-8")
        workflow = accepted / ".github/workflows/autograde.yaml"
        workflow.parent.mkdir(parents=True, exist_ok=True)
        workflow.write_text("name: autograde\n", encoding="utf-8")
        git_config = accepted / ".git/config"
        git_config.parent.mkdir(parents=True)
        git_config.write_text("[core]\n", encoding="utf-8")
        if public_failures(accepted):
            raise AssertionError("delivery metadata or .git files failed public readiness")
        assert_all_central_pass(accepted, "accepted delivery metadata")

        for label, relative in (
            ("extra-root", "notes.txt"),
            ("extra-workflow", ".github/workflows/extra.yaml"),
            ("grader-tree", "_grader_selftest/copied.py"),
            ("nested-git", "ordinary/.git/nested.txt"),
        ):
            rejected = Path(temporary) / label
            shutil.copytree(accepted, rejected)
            path = rejected / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unexpected\n", encoding="utf-8")
            if not public_failures(rejected):
                raise AssertionError(f"public readiness accepted {label}")
            if central_results(rejected)[PROTECTED_TEST].passed:
                raise AssertionError(f"central grader accepted {label}")
    print("[PASS] delivery metadata and top-level .git accepted; inventory bypasses rejected")


def replace_cell_source(root: Path, cell_id: str, old: str, new: str) -> None:
    notebook = _load_notebook(root)
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)
    source = _source(cell)
    if old not in source:
        raise AssertionError(f"Could not prepare mutation in {cell_id}: {old!r}")
    _set_source(cell, source.replace(old, new, 1))
    _write_notebook(root, notebook)


def assert_rejected(
    case: str,
    expected_test: str,
    mutate,
    *,
    prepare_outputs: bool = False,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"ds217-a04-{case}-") as temporary:
        root = Path(temporary)
        copy_student_package(root)
        complete_notebook(root)
        if prepare_outputs:
            error = execute_notebook_in_place(root)
            if error is not None:
                raise AssertionError(f"{case}: could not prepare correct outputs: {error}")
        mutate(root)
        results = central_results(root)
        if results[expected_test].passed:
            raise AssertionError(
                f"{case}: expected `{expected_test}` to fail; results were "
                f"{[(name, test.passed) for name, test in results.items()]}"
            )
        print(f"[PASS] rejected {case} through {expected_test}")


with tempfile.TemporaryDirectory(prefix="ds217-a04-starter-") as temporary:
    starter = Path(temporary)
    copy_student_package(starter)
    failures = public_failures(starter)
    if len(failures) != 3 or not {
        "notebook JSON, supplied setup, and repaired state order",
        "labeled-block CSV schema, index, and values",
        "selected-purchases membership, arithmetic, order, and index=False",
    }.issubset(failures):
        raise AssertionError(f"starter public failure surface changed: {failures}")
    starter_results = central_results(starter)
    if starter_results[SOURCE_TEST].passed or starter_results[CANONICAL_TEST].passed:
        raise AssertionError("untouched starter unexpectedly passed source or fresh execution")
    print("[PASS] untouched starter has three stable public fixes and fails fresh grading")


with tempfile.TemporaryDirectory(prefix="ds217-a04-correct-") as temporary:
    correct = Path(temporary)
    copy_student_package(correct)
    complete_notebook(correct)

    initial_public = public_failures(correct)
    if set(initial_public) != {
        "labeled-block CSV schema, index, and values",
        "selected-purchases membership, arithmetic, order, and index=False",
    }:
        raise AssertionError(f"correct source without generated files changed: {initial_public}")
    assert_all_central_pass(correct, "correct source without submitted outputs")
    print("[PASS] central grader regenerates missing CSVs from correct source")

    execution_error = execute_notebook_in_place(correct)
    if execution_error is not None:
        raise AssertionError(f"correct notebook failed direct fresh execution: {execution_error}")
    if public_failures(correct):
        raise AssertionError(f"public checker rejected correct generated artifacts: {public_failures(correct)}")
    print("[PASS] correct notebook and newly generated artifacts pass public checks")
    assert_delivery_inventory(correct)

    with tempfile.TemporaryDirectory(prefix="ds217-a04-public-cwd-") as other_cwd:
        checker = subprocess.run(
            [sys.executable, "-B", str(correct / "check_assignment.py")],
            cwd=other_cwd,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    if checker.returncode != 0 or "All public checks passed." not in checker.stdout:
        raise AssertionError(f"relocated public checker failed: {checker.stdout}{checker.stderr}")
    print("[PASS] public checker resolves its package from an unrelated working directory")

    assert_all_central_pass(correct, "correct source with submitted outputs")
    print("[PASS] correct submission passes canonical, relocated, and alternate central runs")

    autograder_env = {
        **os.environ,
        "A04_SKIP_INSTALL": "1",
        "CLASSROOM": "datasci-217-local",
        "ASSIGNMENT": "assignment-04",
        "SUBMISSION_TAG": "submit/local-correct",
        "COMMIT_URL": "https://example.invalid/commit/correct",
        "RELEASE_URL": "https://example.invalid/release/correct",
        "REVIEW_URL": "https://example.invalid/review/correct",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    autograder = subprocess.run(
        [sys.executable, "-B", str(SELFTEST_DIR / "autograder.py")],
        cwd=correct,
        capture_output=True,
        text=True,
        timeout=300,
        env=autograder_env,
    )
    if autograder.returncode != 0:
        raise AssertionError(f"autograder prototype crashed: {autograder.stdout}{autograder.stderr}")
    result = json.loads((correct / "result.json").read_text(encoding="utf-8"))
    if not (
        result["schema"] == "classroom50/result/v1"
        and result["score"] == result["max-score"] == 10
        and len(result["tests"]) == 6
        and all(test["passed"] for test in result["tests"])
    ):
        raise AssertionError(f"invalid Classroom50 result payload: {result}")
    if result["review"] != autograder_env["REVIEW_URL"] or not UTC_DATETIME.fullmatch(
        result["datetime"]
    ):
        raise AssertionError(f"runner context was not emitted exactly: {result}")
    print("[PASS] autograder emits a full-score classroom50/result/v1 payload")

    with tempfile.TemporaryDirectory(prefix="ds217-a04-runner-contract-") as contract_name:
        contract_root = Path(contract_name)

        for label, review_value in (("missing", None), ("empty", "   ")):
            fallback = contract_root / f"review-{label}"
            shutil.copytree(correct, fallback, ignore=shutil.ignore_patterns("result.json"))
            fallback_env = dict(autograder_env)
            if review_value is None:
                fallback_env.pop("REVIEW_URL", None)
            else:
                fallback_env["REVIEW_URL"] = review_value
            completed = subprocess.run(
                [sys.executable, "-B", str(SELFTEST_DIR / "autograder.py")],
                cwd=fallback,
                env=fallback_env,
                text=True,
                capture_output=True,
                timeout=300,
                check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(completed.stdout + completed.stderr)
            fallback_result = json.loads((fallback / "result.json").read_text())
            if fallback_result["review"] != fallback_result["commit"]:
                raise AssertionError(f"REVIEW_URL {label} did not fall back to COMMIT_URL")

        captured_failure = contract_root / "captured-student-failure"
        copy_student_package(captured_failure)
        failed = subprocess.run(
            [sys.executable, "-B", str(SELFTEST_DIR / "autograder.py")],
            cwd=captured_failure,
            env=autograder_env,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        failed_result = json.loads((captured_failure / "result.json").read_text())
        if failed.returncode != 0 or failed_result["score"] >= failed_result["max-score"]:
            raise AssertionError("completed student failure did not exit zero with a failing result")

        for environment_name in REQUIRED_RUNNER_ENV:
            for label, replacement in (("missing", None), ("empty", "   ")):
                broken = contract_root / f"{environment_name.lower()}-{label}"
                shutil.copytree(correct, broken, ignore=shutil.ignore_patterns("result.json"))
                broken_env = dict(autograder_env)
                if replacement is None:
                    broken_env.pop(environment_name, None)
                else:
                    broken_env[environment_name] = replacement
                infrastructure = subprocess.run(
                    [sys.executable, "-B", str(SELFTEST_DIR / "autograder.py")],
                    cwd=broken,
                    env=broken_env,
                    text=True,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
                if infrastructure.returncode == 0 or (broken / "result.json").exists():
                    raise AssertionError(
                        f"{environment_name} {label} did not fail without result.json"
                    )
    print("[PASS] official context, review fallback, UTC datetime, and infrastructure exits")


def stored_output_broken(root: Path) -> None:
    notebook = _load_notebook(root)
    dependent = next(
        cell for cell in notebook["cells"] if cell.get("id") == "a04-task1-dependent"
    )
    _set_source(dependent, 'adjusted_rate = 999\nprint("adjusted_rate:", adjusted_rate)')
    for number, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] == "code":
            cell["execution_count"] = number
            cell["outputs"] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": ["Assignment 04 fresh-run verification passed\n"],
                }
            ]
    _write_notebook(root, notebook)


assert_rejected(
    "stored_output_with_broken_source",
    CANONICAL_TEST,
    stored_output_broken,
    prepare_outputs=True,
)

assert_rejected(
    "loc_inclusive_stop",
    CANONICAL_TEST,
    lambda root: replace_cell_source(
        root,
        "a04-task2-objects",
        '"site-102":"site-103"',
        '"site-102":"site-102"',
    ),
)
assert_rejected(
    "iloc_exclusive_stop",
    CANONICAL_TEST,
    lambda root: replace_cell_source(
        root,
        "a04-task2-objects",
        "measurement_table.iloc[1:3, 0:2]",
        "measurement_table.iloc[1:2, 0:2]",
    ),
)
assert_rejected(
    "series_dataframe_return_type",
    SOURCE_TEST,
    lambda root: replace_cell_source(
        root,
        "a04-task2-objects",
        'measurement_table[["baseline_c"]]',
        'measurement_table["baseline_c"]',
    ),
)
assert_rejected(
    "wrong_arithmetic",
    SOURCE_TEST,
    lambda root: replace_cell_source(
        root,
        "a04-task3-roundtrip",
        'selected_purchases["quantity"] * selected_purchases["unit_price"]',
        'selected_purchases["quantity"] + selected_purchases["unit_price"]',
    ),
)
assert_rejected(
    "strict_mask_boundary",
    SOURCE_TEST,
    lambda root: replace_cell_source(
        root,
        "a04-task3-roundtrip",
        'purchases["quantity"] >= 2',
        'purchases["quantity"] > 2',
    ),
)
assert_rejected(
    "missing_unique_tie_breaker",
    SOURCE_TEST,
    lambda root: (
        replace_cell_source(
            root,
            "a04-task3-roundtrip",
            'by=["line_total", "purchase_id"]',
            'by=["line_total"]',
        ),
        replace_cell_source(
            root,
            "a04-task3-roundtrip",
            "ascending=[False, True]",
            "ascending=[False]",
        ),
    ),
)
assert_rejected(
    "serialized_dataframe_index",
    SOURCE_TEST,
    lambda root: replace_cell_source(
        root,
        "a04-task3-roundtrip",
        "to_csv(SELECTED_OUTPUT_PATH, index=False)",
        "to_csv(SELECTED_OUTPUT_PATH, index=True)",
    ),
)


def hardcode_reference_membership(root: Path) -> None:
    notebook = _load_notebook(root)
    cell = next(
        cell for cell in notebook["cells"] if cell.get("id") == "a04-task3-roundtrip"
    )
    source = _source(cell)
    marker = "selected_purchases.to_csv(SELECTED_OUTPUT_PATH, index=False)"
    inserted = '''selected_purchases = selected_purchases.loc[
    selected_purchases["purchase_id"].isin(
        ["P008", "P003", "P004", "P006", "P001", "P011", "P007", "P009", "P012"]
    )
].copy()
selected_purchases.to_csv(SELECTED_OUTPUT_PATH, index=False)'''
    _set_source(cell, source.replace(marker, inserted, 1))
    _write_notebook(root, notebook)


with tempfile.TemporaryDirectory(prefix="ds217-a04-hardcoded-") as temporary:
    hardcoded = Path(temporary)
    copy_student_package(hardcoded)
    complete_notebook(hardcoded)
    hardcode_reference_membership(hardcoded)
    results = central_results(hardcoded)
    if not results[CANONICAL_TEST].passed or results[ALTERNATE_TEST].passed:
        raise AssertionError(
            "hard-coded reference membership must pass canonical execution but fail alternate fixture"
        )
    print("[PASS] alternate fixture rejects canonical-row hard-coding")

assert_rejected(
    "absolute_content_path",
    SOURCE_TEST,
    lambda root: replace_cell_source(
        root,
        "a04-task3-roundtrip",
        "pd.read_csv(DATA_PATH)",
        'pd.read_csv("/content/data/purchases.csv")',
    ),
)


def edit_fixture(root: Path) -> None:
    path = root / "data" / "purchases.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


assert_rejected("edited_fixture", PROTECTED_TEST, edit_fixture)


def edit_manifest(root: Path) -> None:
    path = root / "data" / "fixture.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["row_count"] = 11
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


assert_rejected("edited_manifest", PROTECTED_TEST, edit_manifest)
assert_rejected(
    "edited_public_checker",
    PROTECTED_TEST,
    lambda root: (root / "check_assignment.py").write_text(
        "print('All public checks passed.')\n",
        encoding="utf-8",
    ),
)
assert_rejected(
    "malformed_notebook_json",
    PROTECTED_TEST,
    lambda root: (root / "assignment.ipynb").write_text("{not json\n", encoding="utf-8"),
)


def remove_required_cell(root: Path) -> None:
    notebook = _load_notebook(root)
    notebook["cells"] = [
        cell for cell in notebook["cells"] if cell.get("id") != "a04-task2-objects"
    ]
    _write_notebook(root, notebook)


assert_rejected("missing_required_cell", PROTECTED_TEST, remove_required_cell)


with tempfile.TemporaryDirectory(prefix="ds217-a04-alternate-direct-") as temporary:
    alternate = Path(temporary)
    copy_student_package(alternate)
    complete_notebook(alternate)
    for filename in ("purchases.csv", "fixture.json"):
        shutil.copy2(ALTERNATE_FIXTURE_DIR / filename, alternate / "data" / filename)
    error = execute_notebook_in_place(alternate, nested_launch=True)
    if error is not None:
        raise AssertionError(f"correct notebook failed direct alternate execution: {error}")
    print("[PASS] correct notebook directly executes against the alternate manifest and fixture")


with tempfile.TemporaryDirectory(prefix="ds217-a04-resubmit-") as temporary:
    resubmission = Path(temporary)
    copy_student_package(resubmission)
    complete_notebook(resubmission)
    replace_cell_source(
        resubmission,
        "a04-task3-roundtrip",
        'purchases["quantity"] >= 2',
        'purchases["quantity"] > 2',
    )
    first = central_results(resubmission)
    if first[SOURCE_TEST].passed:
        raise AssertionError("broken first submission unexpectedly passed")
    complete_notebook(resubmission)
    assert_all_central_pass(resubmission, "corrected resubmission")
    print("[PASS] corrected resubmission passes after actionable feedback")

print("All Assignment 04 grader self-tests passed.")
