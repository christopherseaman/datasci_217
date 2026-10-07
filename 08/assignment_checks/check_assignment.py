"""Run the Assignment 08 checks against the saved CSV files and say what to fix."""

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
# The course-owned files that decide the grade; the pytest entrypoint and this script are not needed to grade.
LATEST_FILES = ("_value_checks.py", "grading.py")
LATEST_URL = os.environ.get(
    "DS217_CHECKS_URL", "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/assignment_checks"
)
LATEST_FOLDER = HERE / ".checks"


def _import_grader(folder: Path):
    """Import `grading` from `folder`, forgetting any earlier copy, and return the module."""
    for name in ("grading", "_value_checks"):
        sys.modules.pop(name, None)
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


def left_to_fix(tests: list[dict]) -> str:
    """The failing checks, grouped by the part of their name before the colon.

    A group whose checks all fail reads "clinic counts (all 5 checks)"; otherwise its
    failing checks are named, as in "mean wait pivot: columns and mean waits".
    """
    groups: dict[str, list[dict]] = {}
    for test in tests:
        groups.setdefault(test["test-name"].partition(": ")[0], []).append(test)
    parts = []
    for group, members in groups.items():
        failing = [test for test in members if not test["passed"]]
        if len(failing) == 1:
            parts.append(failing[0]["test-name"])
        elif failing and len(failing) == len(members):
            parts.append(f"{group} (all {len(members)} checks)")
        elif failing:
            names = [test["test-name"].partition(": ")[2] for test in failing]
            if len(names) > 4:
                names = names[:3] + [f"{len(names) - 3} more"]
            parts.append(f"{group}: {', '.join(names[:-1])} and {names[-1]}")
    return "; ".join(parts)


def run_checks(submission_dir=".") -> dict:
    """Grade the submission, print the human-readable report, and return the report dict."""
    grade_submission, note = load_grader()
    print(note)
    result = grade_submission(Path(submission_dir))
    # A fix shared by the checks right after it, such as a missing file, is printed once.
    previous = None
    for test in result["tests"]:
        status = "PASS" if test["passed"] else "FIX "
        line = f"[{status}] {test['score']:>2}/{test['max-score']:<2} {test['test-name']}"
        if test["detail"] and test["detail"] == previous:
            print(f"{line}  (same fix as above)")
        else:
            print(line)
            if test["detail"]:
                print(f"         {test['detail']}")
        previous = test["detail"] or None
    print(f"\nScore: {result['score']}/{result['max-score']}")
    if result["score"] == result["max-score"]:
        print("All checks passed.")
    else:
        lost = result["max-score"] - result["score"]
        print(f"Left to fix ({lost} point{'' if lost == 1 else 's'}): {left_to_fix(result['tests'])}.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--local-checks", action="store_true", help="grade with the bundled checks, without downloading")
    args = parser.parse_args()
    if args.local_checks:
        os.environ["DS217_LOCAL_CHECKS"] = "1"
    if args.json:
        grade_submission, _ = load_grader()
        result = grade_submission(args.submission_dir)
        print(json.dumps(result))
    else:
        result = run_checks(args.submission_dir)
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
