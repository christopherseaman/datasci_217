"""Public pytest contract for the canonical artifact grader."""

import json
from pathlib import Path
import subprocess
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[2]


def test_canonical_cli_json_schema():
    result = subprocess.run(
        [sys.executable, "-B", str(ASSIGNMENT_DIR / "check_assignment.py"), "--json"],
        cwd=ASSIGNMENT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "datasci217/grading-result/v1"
    assert payload["score"] == sum(test["score"] for test in payload["tests"])
