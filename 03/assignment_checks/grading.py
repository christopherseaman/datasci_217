"""Artifact-only grading rules for Assignment 03.

Each check is scored on its own so the report says which artifact to fix, and
each of the fourteen answers is a check of its own so a partly correct summary
scores what it earned.
"""

from __future__ import annotations

from pathlib import Path

from _public_checks import run_public_checks


# One value per check in _public_checks.PUBLIC_CHECKS, in the same order:
# environment probe, record count, monitor counts,
# summary format, then the fourteen answers in README order.
POINTS = (
    13, 10, 15, 12,
    4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 3, 3,
)


def grade_submission(submission_dir: Path) -> dict:
    """Grade committed artifacts without importing or executing submitted code."""
    diagnostics = run_public_checks(Path(submission_dir))
    tests = []
    for (name, detail), max_score in zip(diagnostics, POINTS, strict=True):
        passed = detail is None
        tests.append({"test-name": name, "passed": passed, "score": max_score if passed else 0,
                      "max-score": max_score, "detail": detail or ""})
    return {"schema": "datasci217/grading-result/v1", "score": sum(test["score"] for test in tests),
            "max-score": sum(test["max-score"] for test in tests), "tests": tests}
