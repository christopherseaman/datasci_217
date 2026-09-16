"""Portable pytest entrypoint for Assignment 06.

Invoke the assignment CLI so pytest shares the student and grader rules.
"""

from pathlib import Path
import subprocess
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[2]


def test_assignment_artifacts():
    result = subprocess.run(
        [sys.executable, "-B", str(ASSIGNMENT_DIR / "check_assignment.py")],
        cwd=ASSIGNMENT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    detail = (result.stdout + "\n" + result.stderr).strip()
    assert result.returncode == 0, detail or "checker exited unsuccessfully"
