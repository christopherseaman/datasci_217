"""Artifact-only shape rules for Assignment 02.

This half of the checks travels in your repository and answers one question per
artifact: is it well formed? The values themselves are checked by the course
checks the GitHub Actions run downloads on every push, which is where your
grade comes from. Both halves report the same result format, so the same
tooling runs either one.
"""

from __future__ import annotations

from pathlib import Path

from _shape_checks import run_checks


POINTS = (5, 5, 5, 10, 8, 7, 10, 15, 5, 5, 5, 5, 15)

REPORT_NOTE = (
    "These checks confirm the shape of your artifacts; your values are checked when you push."
)


def grade_submission(submission_dir: Path) -> dict:
    """Check saved artifacts without importing or executing submitted student code."""
    diagnostics = run_checks(Path(submission_dir))
    tests = []
    for (name, detail), max_score in zip(diagnostics, POINTS, strict=True):
        passed = detail is None
        tests.append({"test-name": name, "passed": passed, "score": max_score if passed else 0,
                      "max-score": max_score, "detail": detail or ""})
    return {"schema": "datasci217/grading-result/v1", "score": sum(test["score"] for test in tests),
            "max-score": sum(test["max-score"] for test in tests), "tests": tests}
