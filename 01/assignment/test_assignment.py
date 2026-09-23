"""Pytest contract for Assignment 01: one test per artifact check.

The handout ships a byte-identical copy of this file, so the local run and the
GitHub Actions run, which downloads the course's current copy, go through the
same entrypoint.
"""

from pathlib import Path
import sys

import pytest


ASSIGNMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSIGNMENT_DIR))

import grading  # noqa: E402


# Graded once at collection so each check reports as its own test.
REPORT = grading.grade_submission(ASSIGNMENT_DIR)
NOTE = getattr(grading, "REPORT_NOTE", "")


@pytest.mark.parametrize("check_name", [test["test-name"] for test in REPORT["tests"]])
def test_artifact_check(check_name):
    test = next(test for test in REPORT["tests"] if test["test-name"] == check_name)
    assert test["passed"], "\n\n".join(
        part for part in (test["detail"], f"Score so far: {REPORT['score']}/{REPORT['max-score']}", NOTE) if part
    )
