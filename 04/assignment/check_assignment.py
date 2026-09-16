"""Run the Assignment 04 artifact checks.

Checks read committed artifacts; student notebook code is never executed or inspected.
"""

from __future__ import annotations

import csv
import argparse
from decimal import Decimal, InvalidOperation
import json
from math import isclose
from pathlib import Path


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_text(path: Path, label: str) -> str:
    _assert(path.is_file(), f"Missing {label}.")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError(f"{label} must be UTF-8 text.") from error


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


def _source_rows() -> list[dict]:
    path = ASSIGNMENT_DIR / "data" / "purchases.csv"
    with path.open("r", encoding="utf-8", newline="") as data_file:
        return list(csv.DictReader(data_file))


def _expected_selected() -> list[dict]:
    selected = []
    for row in _source_rows():
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

    expected = _expected_selected()
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", nargs="?", type=Path, default=ASSIGNMENT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    from grading import grade_submission

    result = grade_submission(args.submission_dir)
    if args.json:
        print(json.dumps(result))
    else:
        for test in result["tests"]:
            print(f"[{'PASS' if test['passed'] else 'FIX'}]  {test['test-name']}" + (f": {test['detail']}" if test["detail"] else ""))
        print(f"\nScore: {result['score']}/{result['max-score']}")
        if result["score"] == result["max-score"]:
            print("All checks passed.")
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
