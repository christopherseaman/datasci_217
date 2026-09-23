"""Public pytest contract for Assignment 03.

One test per check, so a failing run names the artifact to fix. The checks read
committed files only; no submitted code is imported or executed.
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
