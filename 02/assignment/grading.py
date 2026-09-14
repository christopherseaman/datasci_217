"""Public, artifact-only grading rules for Assignment 02."""

from __future__ import annotations

from pathlib import Path

from _public_checks import run_public_checks


def grade_submission(submission_dir: Path) -> dict:
    """Grade artifacts without importing or executing submitted student code."""
    diagnostics = run_public_checks(Path(submission_dir))
    tests = []
    for name, detail in diagnostics:
        passed = detail is None
        tests.append({"test-name": name, "passed": passed, "score": 0, "max-score": 0, "detail": detail or ""})
    complete = all(test["passed"] for test in tests)
    tests.append({"test-name": "complete required artifacts", "passed": complete,
                  "score": 100 if complete else 0, "max-score": 100,
                  "detail": "" if complete else "Complete every public artifact check."})
    return {"schema": "datasci217/grading-result/v1", "score": 100 if complete else 0,
            "max-score": 100, "tests": tests}
