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
        print(SCOPE_NOTE)
        print()
        for test in result["tests"]:
            status = "PASS" if test["passed"] else "FIX "
            print(f"[{status}] {test['score']:>3}/{test['max-score']:<3} {test['test-name']}")
            if test["detail"]:
                print(f"         {test['detail']}")
        print(f"\n{SCORE_LABEL}: {result['score']}/{result['max-score']}")
        if result["score"] == result["max-score"]:
            print(COMPLETE_NOTE)
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
