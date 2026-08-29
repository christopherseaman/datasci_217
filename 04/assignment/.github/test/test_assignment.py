"""Portable pytest entrypoint for Assignment 04.

The public checker is intentionally invoked as a subprocess so the exported
assignment repository has one visible pytest contract without importing the
instructor-only grader bundle.
"""

from pathlib import Path
import subprocess
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[2]


def test_public_checker():
    result = subprocess.run(
        [sys.executable, "-B", str(ASSIGNMENT_DIR / "check_assignment.py")],
        cwd=ASSIGNMENT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    detail = (result.stdout + "\n" + result.stderr).strip()
    assert result.returncode == 0, detail or "public checker exited unsuccessfully"
