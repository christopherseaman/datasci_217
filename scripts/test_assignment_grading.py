# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.3.3", "pandas==3.0.5", "scikit-learn==1.9.0"]
# ///
"""Check every grader's empty/scaffold contract without executing submissions.

Runs each assignment's handout checker and, where it exists, its course-owned
NN/assignment_checks/ checker, against an empty submission and the scaffold.
Where the handout ships a byte-identical copy of every file the workflow
downloads from NN/assignment_checks/, as homework handouts do so students can
test locally, both copies must print the same report.

Assignment 11's checks refuse any other versions of these dependencies:
    uv run scripts/test_assignment_grading.py
"""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def ships_course_checks(assignment: Path, course_owned: Path) -> bool:
    """Whether the handout carries every file its workflow downloads, byte for byte."""
    workflow = assignment / ".github" / "workflows" / "tests.yml"
    if not workflow.is_file():
        return False
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow.read_text(encoding="utf-8"), re.M)
    return listed is not None and all(
        (assignment / name).is_file() and (course_owned / name).is_file()
        and (assignment / name).read_bytes() == (course_owned / name).read_bytes()
        for name in listed.group(1).split())


def main():
    repo = Path(__file__).resolve().parents[1]
    scratch = repo / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="grading-contract-") as directory:
        target = Path(directory)
        for name in ("grading.py", "check_assignment.py", "_assignment_checks.py", "_public_checks.py",
                     "_shape_checks.py", "_value_checks.py"):
            (target / name).write_text("raise RuntimeError('submission code must not run')\n")
        for number in range(1, 12):
            assignment = repo / f"{number:02}" / "assignment"
            course_owned = repo / f"{number:02}" / "assignment_checks"
            checkers = [assignment] + ([course_owned] if (course_owned / "check_assignment.py").is_file() else [])
            reports = {}
            for checks in checkers:
                for submission in (target, assignment):
                    result = subprocess.run(
                        [sys.executable, "-B", str(checks / "check_assignment.py"), str(submission), "--json"],
                        cwd=target, capture_output=True, text=True, check=False,
                    )
                    assert result.returncode == 1, (number, checks, result.stdout, result.stderr)
                    report = json.loads(result.stdout)
                    assert report["schema"] == "datasci217/grading-result/v1", report
                    assert report["score"] == 0, (number, checks, submission, report)
                    assert report["max-score"] == (85 if number in (5, 11) else 100), report
                    assert sum(test["max-score"] for test in report["tests"]) == report["max-score"], report
                    assert sum(test["score"] for test in report["tests"]) == 0, report
                    assert "submission code must not run" not in result.stdout + result.stderr, report
                    reports[checks, submission] = report
            halves = " and course-owned" if len(checkers) == 2 else ""
            identical = len(checkers) == 2 and ships_course_checks(assignment, course_owned)
            if identical:
                for submission in (target, assignment):
                    assert reports[assignment, submission] == reports[course_owned, submission], (number, submission)
            print(f"Assignment {number:02}: empty and scaffold earn zero under the handout{halves} checks; "
                  "point total and trusted CLI pass"
                  + ("; the handout ships the course checks unchanged and reports the same" if identical else ""))


if __name__ == "__main__":
    main()
