"""Check every public grader's empty/starter contract without executing submissions.

Run with the assignment grading dependencies installed:
    python scripts/test_assignment_grading.py
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    repo = Path(__file__).resolve().parents[1]
    scratch = repo / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="grading-contract-") as directory:
        target = Path(directory)
        for name in ("grading.py", "check_assignment.py", "_assignment_checks.py"):
            (target / name).write_text("raise RuntimeError('submission code must not run')\n")
        for number in range(1, 12):
            assignment = repo / f"{number:02}" / "assignment"
            for submission in (target, assignment):
                result = subprocess.run(
                    [sys.executable, "-B", str(assignment / "check_assignment.py"), str(submission), "--json"],
                    cwd=target, capture_output=True, text=True, check=False,
                )
                assert result.returncode == 1, (number, result.stdout, result.stderr)
                report = json.loads(result.stdout)
                assert report["schema"] == "datasci217/grading-result/v1", report
                assert report["score"] == 0, (number, submission, report)
                assert report["max-score"] == (85 if number in (5, 11) else 100), report
                assert sum(test["max-score"] for test in report["tests"]) == report["max-score"], report
                assert sum(test["score"] for test in report["tests"]) == 0, report
                assert "submission code must not run" not in result.stdout + result.stderr, report
            print(f"Assignment {number:02}: empty and starter earn zero; point total and trusted CLI pass")


if __name__ == "__main__":
    main()
