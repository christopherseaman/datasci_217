"""Grade one Assignment 05 (midterm) submission from its committed files.

    python3 05/assignment_checks/check_assignment.py <submission_dir> [--json]

Course-side only: the midterm handout ships no checks. Only the files in the
submission's `output/` are read; no submitted code runs. Exits 0 when every
check passes and 1 otherwise. By default the latest `grading.py` is fetched from
the course repository; `--local-checks` or DS217_LOCAL_CHECKS=1 uses this folder's
copy, and DS217_CHECKS_URL points the fetch elsewhere.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LATEST_FILES = ("grading.py",)
LATEST_URL = os.environ.get(
    "DS217_CHECKS_URL", "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/05/assignment_checks"
)
LATEST_FOLDER = HERE / ".checks"

HUMAN_REVIEW_POINTS = 25


def _import_grader(folder: Path):
    """Import `grading` from `folder`, forgetting any earlier copy, and return the module."""
    sys.modules.pop("grading", None)
    sys.path.insert(0, str(folder))
    try:
        return importlib.import_module("grading")
    finally:
        sys.path.remove(str(folder))


def load_grader(use_latest: bool = True):
    """Return (grade_submission, note): the latest course checks when they download and run, else the bundled copy."""
    if use_latest and not os.environ.get("DS217_LOCAL_CHECKS"):
        try:
            LATEST_FOLDER.mkdir(exist_ok=True)
            for name in LATEST_FILES:
                with urllib.request.urlopen(f"{LATEST_URL}/{name}", timeout=10) as response:
                    text = response.read().decode("utf-8")
                if not text.strip():
                    raise ValueError(f"{name} is empty")
                (LATEST_FOLDER / name).write_text(text, encoding="utf-8")
            grader = _import_grader(LATEST_FOLDER)
            # Run once on an empty folder: grading.py raises when its points and checks disagree.
            grader.grade_submission(Path(tempfile.mkdtemp()))
            return grader.grade_submission, "Checks: latest from the course repository (main)."
        except Exception as error:  # offline, HTTP error, or a set that does not run
            reason = f"{type(error).__name__}: {error}"
            note = f"Checks: used the bundled copy because the latest could not be fetched ({reason})."
    else:
        note = "Checks: bundled copy (--local-checks)."
    return _import_grader(HERE).grade_submission, note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path.cwd(),
                        help="the submission's assignment directory (default: the current directory)")
    parser.add_argument("--json", action="store_true", help="print the datasci217/grading-result/v1 report")
    parser.add_argument("--local-checks", action="store_true", help="grade with the bundled checks, without downloading")
    args = parser.parse_args()
    if args.local_checks:
        os.environ["DS217_LOCAL_CHECKS"] = "1"
    grade_submission, note = load_grader()
    result = grade_submission(args.submission_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(note)
        print(f"Assignment 05 midterm: {result['max-score']} points from the committed files "
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
