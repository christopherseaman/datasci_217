"""Artifact-only grading rules for Assignment 06.

Each check is scored on its own, so a partly correct CSV earns the points for
what it got right and the report names what to fix.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


# One value per check in _value_checks.CHECKS, in the same order: the merge
# audit (Task 1.2), the stacked batches (Task 2.1), the aligned features
# (Task 2.3), the long readings (Task 3.1), then the round trip (Task 3.2).
POINTS = (
    8, 8, 8, 8, 8,
    3, 4, 4, 4,
    3, 4, 4, 4,
    5, 5, 5,
    3, 4, 4, 4,
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
