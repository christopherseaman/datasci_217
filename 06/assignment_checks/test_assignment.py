"""Pytest contract for Assignment 06: one test per artifact check.

The handout ships a byte-identical copy of this file, so the local run and the
GitHub Actions run, which downloads the course's current copy, go through the
same entrypoint. The checks read the saved CSV files only; no submitted code
is imported or executed.
"""

from pathlib import Path
import sys

import pytest


ASSIGNMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSIGNMENT_DIR))

from grading import grade_submission  # noqa: E402


RESULT = grade_submission(ASSIGNMENT_DIR)


@pytest.mark.parametrize("check", RESULT["tests"], ids=[check["test-name"] for check in RESULT["tests"]])
def test_assignment_artifact(check):
    assert check["passed"], (
        f"{check['detail']} "
        f"[{check['score']}/{check['max-score']} points; run `python check_assignment.py` for the full report]"
    )
