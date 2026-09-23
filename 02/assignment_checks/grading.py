"""Artifact-only grading rules for Assignment 02.

Each requirement is scored on its own, so a partial submission reports exactly
which value to fix, and every expected value is recomputed from the supplied
encounter file.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


POINTS = (5, 5, 5, 10, 8, 7, 10, 15, 5, 5, 5, 5, 15)


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
