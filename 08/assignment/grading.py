"""Artifact-only grading rules for Assignment 08.

Each check is scored on its own, so a partly correct CSV earns the points for
what it got right and the report names what to fix.
"""

from __future__ import annotations

from pathlib import Path

from _value_checks import run_checks


# One value per check in _value_checks.CHECKS, in the same order: the site summary
# (Task 1.2), the site and modality summary (Task 2.1), the delayed STAT exams
# (Task 2.2), the CT dose review (Task 2.3), the slowest STAT reports (Task 2.4),
# then the pivot (Task 3.1).
POINTS = (
    4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4,
    4, 4, 4, 4, 4,
    2, 4, 2,
    2, 4, 2,
    4, 4, 4, 4,
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
