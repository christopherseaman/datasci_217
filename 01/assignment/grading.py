"""Artifact-only grading rules for Assignment 01.

Each check is scored on its own: 10 points for each terminal-practice file,
5 for each graded line of the readiness report, and 15 for the identity hash.
A wrong report line costs its own 5 points and nothing else.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


# In the order of _value_checks.CHECKS: two practice files, thirteen report lines, and the identity hash.
POINTS = (10, 10) + (5,) * 13 + (15,)


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
