"""Grade one Assignment 11 (final exam) submission from its committed files.

    python3 11/assignment_checks/check_assignment.py <submission_dir> [--json]

Course-side only: the exam handout ships no checks. Only the files in the
submission's `output/` and its `report.md` are read; no submitted code runs.
Exits 0 when every check passes, 1 otherwise, and 2 when the supplied release in
`11/assignment/data/` is missing or changed, so nothing could be graded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from grading import InfrastructureError, SCHEMA, grade_submission


HUMAN_REVIEW_POINTS = 25


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path.cwd(),
                        help="the submission's assignment directory (default: the current directory)")
    parser.add_argument("--json", action="store_true", help="print the datasci217/grading-result/v1 report")
    args = parser.parse_args()
    try:
        result = grade_submission(args.submission_dir)
    except InfrastructureError as error:
        if args.json:
            print(json.dumps({"schema": SCHEMA, "error": str(error)}))
        else:
            print(f"Cannot grade: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
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
