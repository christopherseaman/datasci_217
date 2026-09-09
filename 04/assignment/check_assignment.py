"""Run the discoverable public Assignment 04 artifact checks.

Checks read committed artifacts; student notebook code is never executed or inspected.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from math import isclose
from pathlib import Path


ASSIGNMENT_DIR = Path(__file__).resolve().parent
EXPECTED_PYTHON = "3.12.13\n"
EXPECTED_REQUIREMENTS = "numpy==2.0.2\npandas==3.0.5\n"
EXPECTED_GITIGNORE = (
    ".venv/\n"
    ".ipynb_checkpoints/\n"
    "__pycache__/\n"
    "*.pyc\n"
    ".pytest_cache/\n"
)
EXPECTED_FIXTURE = {
    "fixture_id": "a04-purchases-v1",
    "provenance": "course-authored synthetic teaching data",
    "row_count": 12,
    "columns": ["purchase_id", "item", "quantity", "unit_price"],
    "sha256": "0e86448f20a071552f8456075b8decef7541669b21345949a505aa93c78a07c9",
}
STUDENT_PACKAGE_FILES = {
    ".gitignore", ".python-version", "PLATFORM_CHECK.md", "README.md",
    "assignment.ipynb", "check_assignment.py", "requirements.txt",
    "data/fixture.json", "data/purchases.csv",
    ".github/test/requirements.txt", ".github/test/test_assignment.py",
    ".github/workflows/tests.yml",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_text(path: Path, label: str) -> str:
    _assert(path.is_file(), f"Missing {label}.")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError(f"{label} must be UTF-8 text.") from error


def _load_json(path: Path, label: str):
    source = _read_text(path, label)
    try:
        return json.loads(source)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"{label} is not valid JSON at line {error.lineno}: {error.msg}."
        ) from error


def _check_submission_inventory(root: Path) -> None:
    ignored_roots = {
        ".git", "output", "_grader_selftest", ".venv", "venv",
        "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "result.json",
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not any(part in ignored_roots for part in path.relative_to(root).parts)
    }
    _assert(STUDENT_PACKAGE_FILES <= actual, "Required submission files are missing.")


def check_environment_and_fixture(root: Path) -> None:
    _check_submission_inventory(root)
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Missing regular output/ directory.")
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert({".gitkeep", "labeled_block.csv", "selected_purchases.csv"} <= actual, "Required output artifacts are missing.")
    _assert(
        _read_text(root / ".python-version", ".python-version") == EXPECTED_PYTHON,
        "Restore .python-version to exactly `3.12.13` and one final newline.",
    )
    _assert(
        _read_text(root / "requirements.txt", "requirements.txt")
        == EXPECTED_REQUIREMENTS,
        "Restore requirements.txt to the exact NumPy 2.0.2 and pandas 3.0.5 records.",
    )
    _assert(
        _read_text(root / ".gitignore", ".gitignore") == EXPECTED_GITIGNORE,
        "Restore the supplied notebook, environment, and cache exclusions in .gitignore.",
    )

    manifest = _load_json(root / "data" / "fixture.json", "data/fixture.json")
    _assert(manifest == EXPECTED_FIXTURE, "Restore the supplied canonical data/fixture.json manifest.")
    data_path = root / "data" / "purchases.csv"
    data_bytes = data_path.read_bytes() if data_path.is_file() else b""
    _assert(data_bytes, "Missing data/purchases.csv.")
    _assert(
        sha256(data_bytes).hexdigest() == EXPECTED_FIXTURE["sha256"],
        "Restore the immutable data/purchases.csv bytes; do not edit the fixture.",
    )
    with data_path.open("r", encoding="utf-8", newline="") as data_file:
        reader = csv.DictReader(data_file)
        rows = list(reader)
    _assert(
        reader.fieldnames == EXPECTED_FIXTURE["columns"],
        "data/purchases.csv columns do not match its manifest.",
    )
    _assert(
        len(rows) == EXPECTED_FIXTURE["row_count"],
        "data/purchases.csv row count does not match its manifest.",
    )


def _decimal(value: str, contract: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise AssertionError(f"{contract} must contain numeric values; found {value!r}.") from error


def check_labeled_block(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Missing regular output/ directory.")
    path = root / "output" / "labeled_block.csv"
    _assert(
        path.is_file() and not path.is_symlink(),
        "Missing output/labeled_block.csv; complete Task 2 and create the committed artifact.",
    )
    with path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        rows = list(reader)
    _assert(
        reader.fieldnames == ["record_id", "baseline_c", "follow_up_c"],
        "output/labeled_block.csv must preserve the named record_id index and the two measurement columns.",
    )
    _assert(
        [row["record_id"] for row in rows] == ["site-102", "site-103"],
        "output/labeled_block.csv must contain the inclusive site-102 through site-103 label block.",
    )
    expected_values = [(Decimal("15"), Decimal("23")), (Decimal("10"), Decimal("17"))]
    actual_values = [
        (
            _decimal(row["baseline_c"], "output/labeled_block.csv"),
            _decimal(row["follow_up_c"], "output/labeled_block.csv"),
        )
        for row in rows
    ]
    _assert(
        all(isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9) for pair, wanted in zip(actual_values, expected_values) for actual, expected in zip(pair, wanted)),
        "output/labeled_block.csv values do not match the required .loc/.iloc block.",
    )


def _source_rows(root: Path) -> list[dict]:
    path = root / "data" / "purchases.csv"
    with path.open("r", encoding="utf-8", newline="") as data_file:
        return list(csv.DictReader(data_file))


def _expected_selected(root: Path) -> list[dict]:
    selected = []
    for row in _source_rows(root):
        quantity = _decimal(row["quantity"], "data/purchases.csv")
        unit_price = _decimal(row["unit_price"], "data/purchases.csv")
        if quantity >= 2:
            selected.append(
                {
                    "purchase_id": row["purchase_id"],
                    "item": row["item"],
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "line_total": quantity * unit_price,
                }
            )
    return sorted(selected, key=lambda row: (-row["line_total"], row["purchase_id"]))


def check_selected_purchases(root: Path) -> None:
    path = root / "output" / "selected_purchases.csv"
    _assert(
        path.is_file() and not path.is_symlink(),
        "Missing output/selected_purchases.csv; complete Task 3 and create the committed artifact.",
    )
    with path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        rows = list(reader)

    expected_columns = [
        "purchase_id",
        "item",
        "quantity",
        "unit_price",
        "line_total",
    ]
    _assert(
        reader.fieldnames == expected_columns,
        "output/selected_purchases.csv must have exactly the five required columns; use index=False.",
    )
    _assert(
        not any((name or "").startswith("Unnamed:") for name in reader.fieldnames or []),
        "output/selected_purchases.csv contains a serialized DataFrame index; write it with index=False.",
    )

    expected = _expected_selected(root)
    _assert(
        len(rows) == len(expected) == 9,
        "output/selected_purchases.csv must contain the nine purchases with quantity at least two.",
    )
    _assert(
        [row["purchase_id"] for row in rows] == [row["purchase_id"] for row in expected],
        "Sort selected purchases by line_total descending and purchase_id ascending as the unique tie-breaker.",
    )

    for actual, wanted in zip(rows, expected, strict=True):
        _assert(
            actual["item"] == wanted["item"],
            f"Item membership is incorrect for {wanted['purchase_id']}.",
        )
        actual_quantity = _decimal(actual["quantity"], "output/selected_purchases.csv")
        actual_unit_price = _decimal(actual["unit_price"], "output/selected_purchases.csv")
        actual_total = _decimal(actual["line_total"], "output/selected_purchases.csv")
        _assert(
            actual_quantity == wanted["quantity"] and actual_quantity >= 2,
            f"Quantity selection is incorrect for {wanted['purchase_id']}.",
        )
        _assert(
            isclose(actual_unit_price, wanted["unit_price"], rel_tol=1e-9, abs_tol=1e-9),
            f"Unit price is incorrect for {wanted['purchase_id']}.",
        )
        _assert(
            isclose(actual_total, actual_quantity * actual_unit_price, rel_tol=1e-9, abs_tol=1e-9)
            and isclose(actual_total, wanted["line_total"], rel_tol=1e-9, abs_tol=1e-9),
            f"line_total must equal quantity * unit_price for {wanted['purchase_id']}.",
        )


PUBLIC_CHECKS = (
    ("candidate environment records and immutable fixture", check_environment_and_fixture),
    ("labeled-block CSV schema, index, and values", check_labeled_block),
    ("selected-purchases membership, arithmetic, order, and index=False", check_selected_purchases),
)


def run_public_checks(root: Path) -> list[tuple[str, str | None]]:
    results = []
    for name, action in PUBLIC_CHECKS:
        try:
            action(root)
        except Exception as error:  # continue so students receive every actionable message
            results.append((name, str(error) or error.__class__.__name__))
        else:
            results.append((name, None))
    return results


def main() -> None:
    results = run_public_checks(ASSIGNMENT_DIR)
    failure_count = 0
    for name, error in results:
        if error is None:
            print(f"[PASS] {name}")
        else:
            failure_count += 1
            print(f"[FIX]  {name}: {error}")

    if failure_count:
        print(f"\n{failure_count} public check(s) still need attention.")
        raise SystemExit(1)
    print("\nAll public checks passed.")


if __name__ == "__main__":
    main()
