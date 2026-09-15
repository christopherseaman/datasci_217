"""Public, artifact-only grading rules for Assignment 03."""

from __future__ import annotations

from pathlib import Path

from _public_checks import run_public_checks


POINTS = (20, 40, 40)


def grade_submission(submission_dir: Path) -> dict:
    """Grade artifacts without importing or executing submitted student code."""
    diagnostics = run_public_checks(Path(submission_dir))
    tests = []
    for (name, detail), max_score in zip(diagnostics, POINTS, strict=True):
        passed = detail is None
        tests.append({"test-name": name, "passed": passed, "score": max_score if passed else 0,
                      "max-score": max_score, "detail": detail or ""})
    return {"schema": "datasci217/grading-result/v1", "score": sum(test["score"] for test in tests),
            "max-score": sum(test["max-score"] for test in tests), "tests": tests}
