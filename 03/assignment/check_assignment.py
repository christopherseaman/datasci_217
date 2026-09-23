"""Run the artifact-only Assignment 03 checks and print the score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _public_checks import COMPLETE_NOTE, SCOPE_NOTE, SCORE_LABEL
from grading import grade_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = grade_submission(args.submission_dir)
    if args.json:
        print(json.dumps(result))
    else:
        # Report a count, never a score: these checks have not looked at a value,
        # so printing anything out of 100 would read as a grade.
        passed = 0
        for test in result["tests"]:
            if test["passed"]:
                passed += 1
                print(f"[ OK ] {test['test-name']}")
            else:
                print(f"[ FIX ] {test['test-name']}")
                if test["detail"]:
                    print(f"        {test['detail']}")
        print(f"\n{passed} of {len(result['tests'])} shape checks passed.")
        print(SCOPE_NOTE)
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
