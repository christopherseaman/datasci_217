"""Run the public, artifact-only Assignment 01 grader."""

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
            print(f"[{'PASS' if test['passed'] else 'FIX'}]  {test['test-name']}" + (f": {test['detail']}" if test["detail"] else ""))
        print(f"\nScore: {result['score']}/{result['max-score']}")
        if result["score"] == result["max-score"]:
            print("All public checks passed.")
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
