"""Artifact-only grading rules for Assignment 04.

Each check is scored on its own, so a partly correct CSV earns the points for
what it got right and the report names what to fix.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


# One value per check in _value_checks.CHECKS, in the same order: the four
# fridge-block checks (Task 2), then the four selected-supplies checks (Task 3).
POINTS = (10, 10, 10, 10, 15, 15, 15, 15)


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
