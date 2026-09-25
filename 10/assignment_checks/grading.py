"""Artifact-only grading rules for Assignment 10.

Each check is scored on its own, so a partly correct file earns the points for
what it got right and the report names what to fix.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


# One value per check in _value_checks.CHECKS, in the same order: the coefficient
# table (Task 1.1), the new-patient intervals (Task 1.2), the residuals and their
# plot (Task 1.3), the availability audit (Task 2.1), the split summary (Task 2.2),
# the validation metrics (Task 3.1), the test metrics and predictions (Task 3.2),
# then the readmission metrics (Task 3.3).
POINTS = (
    2, 2, 3, 2, 3,
    2, 1, 2, 2, 2,
    2, 2, 1, 2, 2,
    5,
    2, 2, 1, 3, 3,
    2, 2, 4, 3, 3,
    2, 2, 3, 3, 3,
    2, 1, 2, 2, 2,
    2, 2, 2, 3,
    2, 1, 2, 2, 2,
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
