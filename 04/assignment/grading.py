"""Artifact-only grading rules for Assignment 04.

Each check is scored on its own, so a partly correct CSV earns the points for
what it got right and the report names what to fix.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


# One value per check in _value_checks.CHECKS, in the same order: the Task 1.2 explanation, the loaded
# table (Task 2), the visit summary and clinic counts (Task 3), the follow-up
# list and its Parquet copy (Task 4), and the white-coat gap (Task 5).
POINTS = (
    4,
    4, 4, 4, 4, 4, 4,
    3, 5, 5, 5,
    4, 2,
    3, 6, 2, 2, 4, 4, 2, 3, 2,
    4, 4,
    6, 6,
)


def grade_submission(submission_dir: Path) -> dict:
    """Grade saved artifacts without importing or executing submitted student code."""
    diagnostics = run_checks(Path(submission_dir))
    # Compared by hand rather than with zip(strict=True), which Python 3.9 lacks, so a
    # mismatched pair of files still stops the run.
    if len(diagnostics) != len(POINTS):
        raise ValueError(
            f"_value_checks.py runs {len(diagnostics)} checks but grading.py scores {len(POINTS)}; "
            "update both files together."
        )
    tests = []
    for (name, detail), max_score in zip(diagnostics, POINTS):
        passed = detail is None
        tests.append({"test-name": name, "passed": passed, "score": max_score if passed else 0,
                      "max-score": max_score, "detail": detail or ""})
    return {"schema": "datasci217/grading-result/v1", "score": sum(test["score"] for test in tests),
            "max-score": sum(test["max-score"] for test in tests), "tests": tests}
