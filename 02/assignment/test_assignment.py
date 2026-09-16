"""Public pytest contract for Assignment 02."""

from pathlib import Path
import subprocess
import sys


ASSIGNMENT_DIR = Path(__file__).resolve().parent


def test_assignment_artifacts():
    result = subprocess.run([sys.executable, "-B", "check_assignment.py"], cwd=ASSIGNMENT_DIR, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
