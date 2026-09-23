"""Artifact-only grading rules for Assignment 01: the value checks.

This is the half that decides the grade. The saved readiness report is compared
with the documented report, and the saved identity hash with the course roster;
the report and the identity share their points, so both have to pass.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


POINTS = (20, 80)


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
