"""Run the Assignment 08 checks against the saved CSV files and say what to fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        for test in result["tests"]:
            status = "PASS" if test["passed"] else "FIX "
            print(f"[{status}] {test['score']:>2}/{test['max-score']:<2} {test['test-name']}")
            if test["detail"]:
                print(f"         {test['detail']}")
        print(f"\nScore: {result['score']}/{result['max-score']}")
        if result["score"] == result["max-score"]:
            print("All checks passed.")
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
