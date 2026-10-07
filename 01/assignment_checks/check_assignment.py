"""Run the Assignment 01 checks against saved artifacts and say what to fix."""

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
    "DS217_CHECKS_URL", "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/01/assignment_checks"
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
    """The failing checks, grouped by the file they read.

    A file whose checks all fail reads "output/readiness.txt (all 13 checks)"; otherwise its
    failing checks are named, as in "output/readiness.txt: Total and Review count".
    """
    # The file each check reads and the check's name within it, from the checks that were loaded.
    where = {check.name: (check.artifact, check.label) for check in sys.modules["_value_checks"].CHECKS}
    groups: dict[str, list[tuple[str, bool]]] = {}
    for test in tests:
        artifact, label = where.get(test["test-name"], (test["test-name"], test["test-name"]))
        groups.setdefault(artifact, []).append((label, test["passed"]))
    parts = []
    for artifact, members in groups.items():
        failing = [label for label, passed in members if not passed]
        if not failing:
            continue
        if len(members) > 1 and len(failing) == len(members):
            parts.append(f"{artifact} ({'both' if len(members) == 2 else f'all {len(members)}'} checks)")
        elif failing == [artifact]:
            parts.append(artifact)
        else:
            if len(failing) > 4:
                failing = failing[:3] + [f"{len(failing) - 3} more"]
            named = failing[0] if len(failing) == 1 else f"{', '.join(failing[:-1])} and {failing[-1]}"
            parts.append(f"{artifact}: {named}")
    return "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--local-checks", action="store_true", help="grade with the bundled checks, without downloading")
    args = parser.parse_args()
    if args.local_checks:
        os.environ["DS217_LOCAL_CHECKS"] = "1"
    grade_submission, note = load_grader()
    result = grade_submission(args.submission_dir)
    if args.json:
        print(json.dumps(result))
    else:
        print(note)
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
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
