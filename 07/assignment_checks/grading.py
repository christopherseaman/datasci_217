"""Artifact-only grading rules for Assignment 07.

Each check is scored on its own, so a partly complete submission earns the
points for what it got right and the report names what to fix.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


# One value per check in _value_checks.CHECKS, in the same order.
POINTS = (
    # Task 1, output/exploratory_spec.json: point mark, embedded rows, then x, y, color, and shape.
    4, 5, 4, 4, 4, 4,
    # Task 2: the redesign PNG, then one critique entry per category.
    12, 5, 5, 5, 5, 5,
    # Task 3: the explanatory PNG, the supporting CSV's columns and rows, the six contract
    # strings, the three data types, and the text alternative file.
    8, 4, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
)


def grade_submission(submission_dir: Path) -> dict:
    """Grade saved artifacts without importing or executing submitted student code."""
    diagnostics = run_checks(Path(submission_dir))
    tests = []
    for (name, detail), max_score in zip(diagnostics, POINTS, strict=True):
        passed = detail is None
        tests.append({"test-name": name, "passed": passed, "score": max_score if passed else 0,
                      "max-score": max_score, "detail": detail or ""})
    return {"schema": "datasci217/grading-result/v1", "score": sum(test["score"] for test in tests),
            "max-score": sum(test["max-score"] for test in tests), "tests": tests}
