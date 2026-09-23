"""Check the shape of your Assignment 01 artifacts before you push."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grading import REPORT_NOTE, grade_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = grade_submission(args.submission_dir)
    if args.json:
        print(json.dumps(result))
        return 0 if all(test["passed"] for test in result["tests"]) else 1

    passed = 0
    for test in result["tests"]:
        if test["passed"]:
            passed += 1
            print(f"[ OK ] {test['test-name']}")
        else:
            print(f"[ FIX ] {test['test-name']}")
            print(f"        {test['detail']}")
    print(f"\n{passed} of {len(result['tests'])} shape checks passed.")
    print(REPORT_NOTE)
    return 0 if passed == len(result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
