"""Grade one Assignment 11 (final exam) submission from its committed files.

    python3 11/assignment_checks/check_assignment.py <submission_dir> [--json] [--local-checks]

By default it downloads the latest grading.py from the course repository's
main into .checks/ and falls back to the bundled copy; --local-checks or
DS217_LOCAL_CHECKS=1 uses the bundled copy.

Course-side only: the exam handout ships no checks. Only the files in the
submission's `output/` and its `report.md` are read; no submitted code runs.
Exits 0 when every check passes, 1 otherwise, and 2 when nothing could be
graded: the supplied release in `11/assignment/data/` is missing or changed, or
the Python running this file lacks the packages the checks need.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
LATEST_URL = os.environ.get(
    "DS217_CHECKS_URL", "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/assignment_checks"
)
LATEST_FOLDER = HERE / ".checks"
DATA_DIR = HERE.parent / "assignment" / "data"

try:
    import grading
    from grading import InfrastructureError, SCHEMA
except ModuleNotFoundError as missing:
    # Reported in main() instead of as a traceback, so a run from a Python without
    # NumPy, pandas, or scikit-learn, such as macOS's system python3, says how to run it instead.
    MISSING_PACKAGE = missing.name
    SCHEMA = "datasci217/grading-result/v1"
else:
    MISSING_PACKAGE = None


HUMAN_REVIEW_POINTS = 25
RUN_COMMAND = ("uv run --isolated --project 11/assignment --locked "
               "python3 11/assignment_checks/check_assignment.py <submission_dir>")


def load_grader():
    """Return (grade_submission, note): the latest grading.py when it downloads and runs, else the bundled copy."""
    if not os.environ.get("DS217_LOCAL_CHECKS"):
        try:
            LATEST_FOLDER.mkdir(exist_ok=True)
            with urllib.request.urlopen(f"{LATEST_URL}/grading.py", timeout=10) as response:
                text = response.read().decode("utf-8")
            if not text.strip():
                raise ValueError("grading.py is empty")
            (LATEST_FOLDER / "grading.py").write_text(text, encoding="utf-8")
            sys.modules.pop("grading", None)
            sys.path.insert(0, str(LATEST_FOLDER))
            try:
                latest = importlib.import_module("grading")
            finally:
                sys.path.remove(str(LATEST_FOLDER))
            # The fetched copy sits in .checks/, so point it at this repository's release.
            latest.DATA_DIR = DATA_DIR
            # Run once on an empty folder: grading.py raises when its points and checks disagree.
            latest.grade_submission(Path(tempfile.mkdtemp()))
            return latest, "Checks: latest from the course repository (main)."
        except Exception as error:  # offline, HTTP error, or a set that does not run
            sys.modules["grading"] = grading
            note = f"Checks: used the bundled copy because the latest could not be fetched ({type(error).__name__}: {error})."
    else:
        note = "Checks: bundled copy (--local-checks)."
    return grading, note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path.cwd(),
                        help="the submission's assignment directory (default: the current directory)")
    parser.add_argument("--json", action="store_true", help="print the datasci217/grading-result/v1 report")
    parser.add_argument("--local-checks", action="store_true", help="grade with the bundled checks, without downloading")
    args = parser.parse_args()
    if args.local_checks:
        os.environ["DS217_LOCAL_CHECKS"] = "1"
    if MISSING_PACKAGE:
        message = (f"this Python ({sys.executable}) has no {MISSING_PACKAGE} module, which the checks need. "
                   f"Run them with the packages 11/assignment/pyproject.toml pins, from the course repository: "
                   f"{RUN_COMMAND}")
        if args.json:
            print(json.dumps({"schema": SCHEMA, "error": message}))
        else:
            print(f"Cannot grade: {message}", file=sys.stderr)
        return 2
    grader, note = load_grader()
    try:
        result = grader.grade_submission(args.submission_dir)
    except grader.InfrastructureError as error:
        if args.json:
            print(json.dumps({"schema": SCHEMA, "error": str(error)}))
        else:
            print(f"Cannot grade: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(note)
        print(f"Assignment 11 final exam: {result['max-score']} points from the committed files "
              f"({HUMAN_REVIEW_POINTS} more come from human review)\n")
        for test in result["tests"]:
            status = "PASS" if test["passed"] else "PART" if test["score"] else "FIX "
            print(f"[{status}] {test['score']:>3}/{test['max-score']:<3} {test['test-name']}")
            if test["detail"]:
                print(f"         {test['detail']}")
        print(f"\nScore: {result['score']}/{result['max-score']}")
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
