"""Public, artifact-only grading rules for Assignment 04."""

from __future__ import annotations

from pathlib import Path

from check_assignment import run_public_checks


POINTS = (40, 60)


def grade_submission(submission_dir: Path) -> dict:
    """Grade artifacts without importing or executing submitted notebook code."""
    tests = []
    for (name, detail), max_score in zip(run_public_checks(Path(submission_dir)), POINTS, strict=True):
        passed = detail is None
        tests.append({"test-name": name, "passed": passed, "score": max_score if passed else 0,
                      "max-score": max_score, "detail": detail or ""})
    return {"schema": "datasci217/grading-result/v1", "score": sum(test["score"] for test in tests),
            "max-score": sum(test["max-score"] for test in tests), "tests": tests}
